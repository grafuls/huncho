//! `huncho serve` — run the HTTP API.

use std::sync::Arc;

use clap::Args;
use huncho_api::{AppState, Metrics, ModelRegistry, ServerConfig};
use huncho_core::manifest::{BackendId, CalibrationStatus, Family};

use crate::load::{engine_from_ref, engine_from_resolved_manifest, mock_engine, BackendChoice};

#[derive(Args)]
pub struct ServeArgs {
    /// Address to bind (host:port). Also read from `HUNCHO_BIND`.
    #[arg(long, default_value = "127.0.0.1:8080", env = "HUNCHO_BIND")]
    pub bind: String,

    /// Require this bearer token on requests (API-03). Also read from the
    /// `HUNCHO_AUTH_TOKEN` env var, so secrets can be supplied via an
    /// `EnvironmentFile=` (systemd) or a `.env` without appearing in the
    /// command line.
    #[arg(long, env = "HUNCHO_AUTH_TOKEN")]
    pub auth_token: Option<String>,

    /// Serve a built-in mock model (no weights required).
    #[arg(long, default_value_t = false)]
    pub mock: bool,

    /// Name(s) to register the mock model under (repeatable). If omitted, the
    /// mock is registered under both `mock` and `jev-latest` so that clients
    /// pointing at `model="jev-latest"` work with only a base-URL change.
    #[arg(long = "mock-model")]
    pub mock_model: Vec<String>,

    /// Additional model manifest paths (huncho-model.json) to serve.
    #[arg(long)]
    pub manifest: Vec<String>,

    /// Model reference(s) to serve: a local package dir/path or an HF repo id
    /// (`owner/repo`). Hub references require `hf` (included by `clef`).
    #[arg(long)]
    pub model: Vec<String>,

    /// Git revision to resolve HF model references at (default: repo default branch).
    #[arg(long)]
    pub revision: Option<String>,

    /// Hugging Face access token (defaults to HF_TOKEN / login cache).
    #[arg(long)]
    pub token: Option<String>,

    /// Backend override (auto|onnx|candle|clef|llamacpp|mock). Auto selects a runtime per model.
    #[arg(long, default_value = "auto", env = "HUNCHO_BACKEND")]
    pub backend: String,

    /// Override the dtype for manifest/manually-loaded models. Also read from `HUNCHO_DTYPE`.
    #[arg(long, env = "HUNCHO_DTYPE")]
    pub dtype: Option<String>,

    /// Serve engine extensions by default (API-05).
    #[arg(long, default_value_t = false)]
    pub extensions: bool,

    /// Maximum waiting requests per model; overload returns HTTP 503. Zero disables waiting.
    #[arg(long, default_value = "32", env = "HUNCHO_MAX_QUEUED_PER_MODEL")]
    pub max_queued_per_model: u16,

    /// Independent CPU contexts sharing immutable weights (1–8). Every context
    /// must pass complete labeled startup qualification against pinned goldens.
    #[arg(long, default_value = "1", env = "HUNCHO_REPLICAS", value_parser = clap::value_parser!(u8).range(1..=8))]
    pub replicas: u8,

    /// Bound requests preparing/holding prompts while another request executes (0 disables).
    #[arg(long, default_value = "0", env = "HUNCHO_MAX_PREPARED_PER_MODEL")]
    pub max_prepared_per_model: u16,

    /// Per-model charged-byte budget for exact successful response reuse (0 disables).
    /// Retains input text in memory until eviction or model unload.
    #[arg(long, default_value = "0", env = "HUNCHO_RESULT_CACHE_BYTES")]
    pub result_cache_bytes: usize,

    /// Per-model metadata budget for identical in-flight request sharing (0 disables).
    #[arg(long, default_value = "0", env = "HUNCHO_COALESCE_BYTES")]
    pub coalesce_bytes: usize,

    /// Enable Kev prefix reuse after qualifying the served model/device.
    #[arg(long, default_value_t = false, env = "HUNCHO_PREFIX_CACHE")]
    pub prefix_cache: bool,

    /// Yield CPU Kev execution after each prefix chunk/question (one context).
    /// Requires HUNCHO_PREFILL_CHUNK_TOKENS and interleaved startup conformance.
    #[arg(
        long,
        default_value_t = false,
        env = "HUNCHO_COOPERATIVE_PREFILL",
        requires = "prefix_cache"
    )]
    pub cooperative_prefill: bool,

    /// Charged budget for exact retained native prefixes (0 disables).
    #[arg(long, default_value = "0", env = "HUNCHO_PERSISTENT_PREFIX_BYTES")]
    pub persistent_prefix_bytes: usize,

    /// Enable native equal-length question batching for qualified models/devices.
    #[arg(long, env = "HUNCHO_MAX_BATCH_TOKENS", conflicts_with = "prefix_cache")]
    pub max_batch_tokens: Option<usize>,

    /// Collate up to this many prepared requests (2–64); requires a token budget.
    #[arg(
        long,
        requires = "max_batch_tokens",
        conflicts_with = "prefix_cache",
        env = "HUNCHO_BATCH_MAX_REQUESTS"
    )]
    pub batch_max_requests: Option<u16>,

    /// Maximum cross-request collation wait after enqueue, in milliseconds.
    #[arg(long, default_value = "2", env = "HUNCHO_BATCH_WAIT_MS")]
    pub batch_wait_ms: u16,

    /// Enable qualified F3 candidate-only vocabulary projection.
    #[arg(long, default_value_t = false, env = "HUNCHO_CANDIDATE_READOUT")]
    pub candidate_readout: bool,

    /// Pinned labeled conformance suite for every real runtime, MODEL=PATH.
    /// Startup checks both external goldens and independent-forward parity.
    #[arg(long = "qualification-golden")]
    pub qualification_golden: Vec<String>,

    /// Require a matching retained receipt in addition to fresh conformance.
    /// MODEL=PATH (repeatable); requires the `qualification` build feature.
    #[cfg(feature = "qualification")]
    #[arg(long = "qualification-record")]
    pub qualification_record: Vec<String>,

    /// Model cache directory (also used by HF resolution; OPS-04). Also read from `HUNCHO_CACHE_DIR`.
    #[arg(long, env = "HUNCHO_CACHE_DIR")]
    pub cache_dir: Option<String>,
}

fn load_models(args: &ServeArgs) -> anyhow::Result<ModelRegistry> {
    anyhow::ensure!(!args.cooperative_prefill || (args.replicas == 1 && args.max_queued_per_model <= 62),
        "cooperative prefill requires one context and at most 62 queued requests (64 native cache handles)");
    anyhow::ensure!(
        args.replicas <= 1 || args.batch_max_requests.is_none(),
        "replicas and cross-request collation cannot be combined yet"
    );
    anyhow::ensure!(
        args.max_batch_tokens != Some(0),
        "max batch tokens must be positive"
    );
    anyhow::ensure!(
        match args.batch_max_requests {
            Some(rows) => (2..=64).contains(&rows),
            None => true,
        },
        "batch max requests must be between 2 and 64"
    );
    anyhow::ensure!(
        !(args.prefix_cache && args.max_batch_tokens.is_some()),
        "prefix reuse and batching cannot be combined yet"
    );
    anyhow::ensure!(
        args.persistent_prefix_bytes == 0 || args.prefix_cache,
        "persistent prefixes require --prefix-cache"
    );
    anyhow::ensure!(
        args.persistent_prefix_bytes == 0
            || args.persistent_prefix_bytes >= usize::from(args.replicas),
        "persistent prefix budget must provide at least one byte per replica"
    );
    let mut registry = ModelRegistry::new();
    #[cfg(feature = "qualification")]
    let mut records = RecordBindings::load(&args.qualification_record)?;

    if args.mock {
        let names = if args.mock_model.is_empty() {
            vec!["mock".to_string(), "jev-latest".to_string()]
        } else {
            args.mock_model.clone()
        };
        for name in &names {
            let engine = mock_engine(name, Family::F1, BackendId::Onnx, "fp32", 1.0)?;
            let device = engine.device().to_owned();
            registry.insert(
                name.clone(),
                engine.with_result_cache(args.result_cache_bytes),
            );
            tracing::info!("registered mock model `{name}` on {device} (F1 / onnx / fp32)");
        }
    }

    let backend = BackendChoice::parse(&args.backend)?;

    for path in &args.manifest {
        tracing::info!("loading model manifest {path}");
        let path = std::path::Path::new(path);
        #[cfg(feature = "qualification")]
        let engine = records.load_engine(path, backend, args.dtype.as_deref())?;
        #[cfg(not(feature = "qualification"))]
        let engine = engine_from_resolved_manifest(path, backend, args.dtype.as_deref())?;
        let name = engine.manifest().name.clone();
        let device = engine.device().to_owned();
        registry.insert(
            name.clone(),
            engine.with_result_cache(args.result_cache_bytes),
        );
        tracing::info!("registered model `{name}` on {device}");
    }

    if !args.model.is_empty() {
        // The serve listener binds only after every model is loaded, so make it
        // clear that a first-time Hub resolution may be downloading weights.
        tracing::info!(
            "resolving {} model reference(s) from the Hugging Face Hub; the serve listener comes up only after they are ready",
            args.model.len()
        );
    }

    for model in &args.model {
        tracing::info!(
            "loading model `{model}` (fetching base weights if needed, then materializing; first load can take a couple of minutes)..."
        );
        #[cfg(feature = "qualification")]
        let recorded_path = if records.records.is_empty() {
            None
        } else {
            Some(crate::load::resolve_model(
                model,
                backend,
                args.dtype.as_deref(),
                args.revision.clone(),
                args.token.clone(),
                args.cache_dir.clone(),
                false,
            )?)
        };
        #[cfg(feature = "qualification")]
        let engine = if let Some(path) = recorded_path {
            records.load_engine(&path, backend, args.dtype.as_deref())?
        } else {
            engine_from_ref(
                model,
                backend,
                args.dtype.as_deref(),
                args.revision.clone(),
                args.token.clone(),
                args.cache_dir.clone(),
            )?
        };
        #[cfg(not(feature = "qualification"))]
        let engine = engine_from_ref(
            model,
            backend,
            args.dtype.as_deref(),
            args.revision.clone(),
            args.token.clone(),
            args.cache_dir.clone(),
        )?;
        let name = engine.manifest().name.clone();
        let device = engine.device().to_owned();
        registry.insert(
            name.clone(),
            engine.with_result_cache(args.result_cache_bytes),
        );
        tracing::info!("registered model `{name}` on {device} (from `{model}`)");
    }

    if registry.is_empty() {
        tracing::warn!("no models registered; /v1/systemone will return 422 for every model");
    }
    registry.set_replicas(usize::from(args.replicas))?;
    qualify_optimizations(&registry, args)?;
    #[cfg(feature = "qualification")]
    records.verify(&registry, args)?;
    Ok(registry)
}

fn evaluation_options(
    engine: &huncho_core::engine::Engine,
    args: &ServeArgs,
) -> huncho_core::engine::EvalOptions {
    huncho_core::engine::EvalOptions {
        prefix_cache: args.prefix_cache && engine.supports_prefix_cache(),
        cooperative_prefill: args.cooperative_prefill,
        persistent_prefix_bytes: if args.prefix_cache && engine.supports_prefix_cache() {
            args.persistent_prefix_bytes / usize::from(args.replicas)
        } else {
            0
        },
        max_batch_tokens: args.max_batch_tokens.filter(|_| engine.supports_batch()),
        reference_readout: !args.candidate_readout,
        prepare_all: (args.cooperative_prefill
            || args.max_prepared_per_model > 0
            || (args.batch_max_requests.is_some() && engine.supports_batch()))
            && engine.family() != Family::F5,
        ..Default::default()
    }
}

#[cfg(feature = "qualification")]
struct RecordBindings {
    records: std::collections::BTreeMap<String, crate::qualification::QualificationRecord>,
    inputs: std::collections::BTreeMap<String, crate::qualification::InputSnapshot>,
}

#[cfg(feature = "qualification")]
impl RecordBindings {
    fn load(bindings: &[String]) -> anyhow::Result<Self> {
        let mut records = std::collections::BTreeMap::new();
        for binding in bindings {
            let (name, path) = binding
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("qualification record must be MODEL=PATH"))?;
            anyhow::ensure!(
                !name.is_empty() && !path.is_empty(),
                "qualification record must be MODEL=PATH"
            );
            anyhow::ensure!(
                !records.contains_key(name),
                "duplicate qualification record for `{name}`"
            );
            records.insert(
                name.into(),
                crate::qualification::QualificationRecord::load(std::path::Path::new(path))?,
            );
        }
        Ok(Self {
            records,
            inputs: Default::default(),
        })
    }

    fn load_engine(
        &mut self,
        path: &std::path::Path,
        backend: BackendChoice,
        dtype: Option<&str>,
    ) -> huncho_core::error::Result<huncho_core::engine::Engine> {
        if self.records.is_empty() {
            return engine_from_resolved_manifest(path, backend, dtype);
        }
        crate::load::engine_from_resolved_manifest_observed(
            path,
            backend,
            dtype,
            |manifest, backend, dtype, dir| {
                if self.records.contains_key(&manifest.name) {
                    self.inputs.insert(
                        manifest.name.clone(),
                        crate::qualification::InputSnapshot::capture(
                            path, manifest, backend, dtype, dir,
                        )?,
                    );
                }
                Ok(())
            },
        )
    }

    fn verify(&self, registry: &ModelRegistry, args: &ServeArgs) -> anyhow::Result<()> {
        for (name, record) in &self.records {
            let engine = registry
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("qualification model `{name}` is not registered"))?;
            let inputs = self.inputs.get(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "qualification record requires observed real-artifact loading for `{name}`"
                )
            })?;
            let golden = args
                .qualification_golden
                .iter()
                .filter_map(|binding| binding.split_once('='))
                .find(|(model, _)| *model == name)
                .map(|(_, path)| path)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "qualification records require fresh --qualification-golden {name}=PATH"
                    )
                })?;
            let require_outcomes = engine.calibration().status == CalibrationStatus::Refit
                || requires_outcome_qualification(&engine)
                || engine.replica_engines().len() > 1;
            record.verify(
                &engine,
                inputs,
                &evaluation_options(&engine, args),
                std::path::Path::new(golden),
                args.batch_max_requests
                    .filter(|_| engine.supports_batch())
                    .map(usize::from),
                require_outcomes,
            )?;
        }
        Ok(())
    }
}

fn requires_outcome_qualification(engine: &huncho_core::engine::Engine) -> bool {
    [
        "native_execution",
        "projection_chunk_rows",
        "attention_compute_dtype",
        "device_path",
        "joint_head_execution",
        "joint_pool_execution",
        "onnx_execution_provider",
        "onnx_intra_threads",
        "onnx_native_batch",
        "onnx_initializer_residency",
        "delta_rule_execution",
        "causal_conv_execution",
        "mlp_gate_execution",
        "prefill_chunk_tokens",
        "weight_quantization",
        "cpu_kernel_build",
        "cpu_blas_execution",
        "llamacpp_execution",
    ]
    .iter()
    .any(|key| engine.execution_metadata().contains_key(*key))
}

fn qualify_optimizations(registry: &ModelRegistry, args: &ServeArgs) -> anyhow::Result<()> {
    use huncho_core::conformance::{
        load_suite, run_suite_with_cross_request_batches, run_suite_with_options,
        ConformanceThresholds,
    };
    let mut paths = std::collections::BTreeMap::new();
    for binding in &args.qualification_golden {
        let (name, path) = binding
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("qualification golden must be MODEL=PATH"))?;
        anyhow::ensure!(
            !name.is_empty() && !path.is_empty(),
            "qualification golden must be MODEL=PATH"
        );
        anyhow::ensure!(
            registry.get(name).is_some(),
            "qualification model `{name}` is not registered"
        );
        anyhow::ensure!(
            paths.insert(name, path).is_none(),
            "duplicate qualification binding for `{name}`"
        );
    }
    for (name, engine) in registry.models() {
        anyhow::ensure!(
            engine.calibration().status != CalibrationStatus::Pending,
            "serving `{name}` requires fitted calibration; {}:{} is pending",
            engine.backend_id(),
            engine.dtype()
        );
        let refit = engine.calibration().status == CalibrationStatus::Refit;
        if engine
            .execution_metadata()
            .contains_key("weight_quantization")
        {
            let exact = engine.manifest().calibration.entries.get(
                &engine
                    .manifest()
                    .calibration_key(&engine.backend_id().to_string(), engine.dtype()),
            );
            anyhow::ensure!(exact.is_some_and(|entry| entry.status == CalibrationStatus::Refit),
                "quantized serving for `{name}` requires an explicit backend:dtype refit; an inherited or fitted source entry is insufficient");
        }
        let outcome_profile = requires_outcome_qualification(engine);
        let replicas = engine.replica_engines();
        let replicated = replicas.len() > 1;
        let opts = evaluation_options(engine, args);
        anyhow::ensure!(!args.cooperative_prefill || engine.supports_resumable_prefill(),
            "cooperative prefill requires CPU Kev with HUNCHO_PREFILL_CHUNK_TOKENS configured for every model");
        anyhow::ensure!(!engine.execution_metadata().contains_key("prefill_chunk_tokens") || opts.prefix_cache,
            "chunked prefill serving for `{name}` requires --prefix-cache and an actual split-prefix qualification");
        let readout_storage_profile = [
            "onnx_readout",
            "onnx_output_buffer_bytes",
            "base_weight_cache",
        ]
        .iter()
        .any(|key| engine.execution_metadata().contains_key(*key));
        if !opts.prefix_cache
            && opts.max_batch_tokens.is_none()
            && !(args.candidate_readout && engine.family() == Family::F3)
            && !paths.contains_key(name.as_str())
            && !refit
            && !outcome_profile
            && !readout_storage_profile
            && !opts.prepare_all
            && !replicated
        {
            continue;
        }
        let path=paths.get(name.as_str()).ok_or_else(||anyhow::anyhow!("serving a native, refitted or optimized variant of `{name}` requires --qualification-golden {name}=/path/to/pinned-golden.json"))?;
        let suite = load_suite(path)?;
        anyhow::ensure!(!(refit || outcome_profile || replicated) || suite.cases.iter().any(|case| !case.targets.is_empty()),
            "native, refitted or changed-profile serving for `{name}` requires held-out golden vectors with observed target labels");
        let evaluate = |engine: &huncho_core::engine::Engine| -> huncho_core::Result<_> {
            if let Some(rows) = args.batch_max_requests.filter(|_| engine.supports_batch()) {
                run_suite_with_cross_request_batches(
                    engine,
                    &suite,
                    &ConformanceThresholds::default(),
                    &opts,
                    usize::from(rows),
                )
            } else {
                run_suite_with_options(engine, &suite, &ConformanceThresholds::default(), &opts)
            }
        };
        // Qualify the actual independently locked contexts concurrently, with
        // one complete fixed suite per context. A subset or a primary-only pass
        // cannot silently authorize the remainder of the pool.
        let reports = std::thread::scope(|scope| {
            let jobs: Vec<_> = replicas
                .iter()
                .map(|replica| {
                    let evaluate = &evaluate;
                    scope.spawn(move || evaluate(replica))
                })
                .collect();
            jobs.into_iter()
                .map(|job| {
                    job.join()
                        .map_err(|_| anyhow::anyhow!("replica qualification worker panicked"))?
                        .map_err(anyhow::Error::from)
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })?;
        for (index, report) in reports.into_iter().enumerate() {
            anyhow::ensure!(report.passed,"qualification failed for `{name}` replica {index} on {} / {}: golden delta={}, argmax={}, ECE drift={}, parity={:?}",engine.device(),engine.dtype(),report.max_prob_delta,report.argmax_agreement,report.ece,report.optimization_parity);
        }
        tracing::info!(
            model = name,
            device = engine.device(),
            dtype = engine.dtype(),
            replicas = replicas.len(),
            "qualification passed against pinned goldens (with independent parity for optimized paths)"
        );
    }
    Ok(())
}

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    let registry = load_models(&args)?;
    let config = ServerConfig {
        bind: args.bind.clone(),
        auth_token: args.auth_token.clone(),
        metrics: true,
        default_extensions: args.extensions,
        max_queued_per_model: args.max_queued_per_model,
        max_prepared_per_model: args.max_prepared_per_model,
        coalesce_bytes: args.coalesce_bytes,
        prefix_cache: args.prefix_cache,
        cooperative_prefill: args.cooperative_prefill,
        persistent_prefix_bytes: args.persistent_prefix_bytes,
        max_batch_tokens: args.max_batch_tokens,
        batch_max_requests: args.batch_max_requests,
        batch_wait_ms: args.batch_wait_ms,
        candidate_readout: args.candidate_readout,
    };

    let state = AppState::new(config, registry, Metrics::new());
    tracing::info!("huncho starting (bind={}, mock={})", args.bind, args.mock,);

    huncho_api::serve(Arc::new(state)).await?;
    Ok(())
}

#[cfg(test)]
mod qualification_tests {
    use super::*;
    use clap::Parser;
    use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
    use huncho_core::conformance::{GoldenCase, GoldenSuite};
    use huncho_core::engine::{Engine, EvalOptions};
    use huncho_core::head::HeadParams;
    use huncho_core::tensor::Tensor;
    use huncho_core::tokenizer::SimpleTokenizer;
    use std::collections::BTreeMap;

    struct BatchBackend {
        drift: f32,
        execution_metadata: BTreeMap<String, String>,
        is_replica: bool,
    }
    impl Backend for BatchBackend {
        fn replica(&self) -> huncho_core::Result<Box<dyn Backend>> {
            Ok(Box::new(Self {
                drift: self.drift,
                execution_metadata: self.execution_metadata.clone(),
                is_replica: true,
            }))
        }
        fn id(&self) -> BackendId {
            BackendId::Onnx
        }
        fn capabilities(&self) -> Capabilities {
            let mut extra = self.execution_metadata.clone();
            extra.insert("device".into(), "CPU".into());
            Capabilities {
                id: BackendId::Onnx,
                dtype: "fp32".into(),
                families: vec![Family::F1],
                extra,
                ..Default::default()
            }
        }
        fn supports_batch(&self) -> bool {
            true
        }
        fn forward(&mut self, input: ForwardInput) -> huncho_core::Result<ForwardOutput> {
            output(input, if self.is_replica { self.drift } else { 0. })
        }
        fn forward_batch(
            &mut self,
            inputs: Vec<ForwardInput>,
        ) -> huncho_core::Result<Vec<ForwardOutput>> {
            inputs
                .into_iter()
                .map(|input| output(input, self.drift))
                .collect()
        }
        fn fork(&mut self, _: CacheHandle) -> huncho_core::Result<CacheHandle> {
            unreachable!()
        }
    }
    fn output(input: ForwardInput, drift: f32) -> huncho_core::Result<ForwardOutput> {
        let values = (0..input.positions.len())
            .map(|row| if row == 0 { drift } else { 1. })
            .collect();
        Ok(ForwardOutput::Logits {
            values: Tensor::new(vec![input.positions.len(), 1], values)?,
            positions: input.positions,
        })
    }
    fn registry(drift: f32) -> ModelRegistry {
        registry_with_calibration(drift, CalibrationStatus::Fit)
    }
    fn registry_with_calibration(drift: f32, status: CalibrationStatus) -> ModelRegistry {
        registry_with_profile(drift, status, false)
    }
    fn registry_with_profile(
        drift: f32,
        status: CalibrationStatus,
        kernel_profile: bool,
    ) -> ModelRegistry {
        registry_with_execution_metadata(
            drift,
            status,
            if kernel_profile {
                BTreeMap::from([("projection_chunk_rows".into(), "64".into())])
            } else {
                BTreeMap::new()
            },
        )
    }
    fn registry_with_execution_metadata(
        drift: f32,
        status: CalibrationStatus,
        execution_metadata: BTreeMap<String, String>,
    ) -> ModelRegistry {
        let mut manifest = crate::load::mock_manifest("qual", Family::F1, "fp32", 1.).unwrap();
        manifest.calibration.default.status = status;
        for entry in manifest.calibration.entries.values_mut() {
            entry.status = status;
        }
        manifest.prompt_contract.template = "laya-v1".into();
        let engine = Engine::new(
            manifest,
            Box::new(BatchBackend {
                drift,
                execution_metadata,
                is_replica: false,
            }),
            Box::new(SimpleTokenizer::new(32768)),
            HeadParams::default(),
            BackendId::Onnx,
            "fp32",
        )
        .unwrap();
        let mut registry = ModelRegistry::new();
        registry.insert("qual", engine);
        registry
    }
    fn args() -> ServeArgs {
        let crate::Command::Serve(args) =
            crate::Cli::try_parse_from(["huncho", "serve", "--max-batch-tokens", "1024"])
                .unwrap()
                .command
        else {
            panic!("serve")
        };
        args
    }
    fn suite() -> GoldenSuite {
        // Fixed analytical probabilities for logits [0,1], not regenerated
        // native reference vectors. Two independent identical questions batch.
        let low = 1. / (1. + 1f32.exp());
        let request = serde_json::from_value(
            serde_json::json!({"model":"qual","state":"state","questions":{
                "a":{"type":"choice","instructions":"choose","criteria":{"a":null,"b":null}},
                "b":{"type":"choice","instructions":"choose","criteria":{"a":null,"b":null}}
            }}),
        )
        .unwrap();
        GoldenSuite {
            schema_version: "1.0".into(),
            family: "F1".into(),
            hash: None,
            cases: vec![GoldenCase {
                id: "batch".into(),
                request,
                expected: BTreeMap::from([
                    (
                        "a".into(),
                        BTreeMap::from([("a".into(), low), ("b".into(), 1. - low)]),
                    ),
                    (
                        "b".into(),
                        BTreeMap::from([("a".into(), low), ("b".into(), 1. - low)]),
                    ),
                ]),
                targets: Default::default(),
            }],
        }
    }

    #[test]
    fn every_replica_requires_complete_labeled_qualification_and_a_primary_pass_cannot_hide_drift()
    {
        let mut args = args();
        args.max_batch_tokens = None;
        args.replicas = 2;
        let mut models = registry(0.);
        models.set_replicas(2).unwrap();
        assert!(qualify_optimizations(&models, &args)
            .unwrap_err()
            .to_string()
            .contains("requires --qualification-golden"));
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("golden.json");
        args.qualification_golden = vec![format!("qual={}", path.display())];
        let mut golden = suite();
        huncho_core::conformance::save_suite(&golden, &path).unwrap();
        assert!(qualify_optimizations(&models, &args)
            .unwrap_err()
            .to_string()
            .contains("observed target labels"));
        golden.cases[0].targets =
            BTreeMap::from([("a".into(), "b".into()), ("b".into(), "b".into())]);
        huncho_core::conformance::save_suite(&golden, &path).unwrap();
        qualify_optimizations(&models, &args).unwrap();
        let mut drifting = registry(0.02);
        drifting.set_replicas(2).unwrap();
        assert!(
            huncho_core::conformance::run_suite(
                &drifting.get("qual").unwrap(),
                &golden,
                &Default::default()
            )
            .unwrap()
            .passed
        );
        assert!(qualify_optimizations(&drifting, &args)
            .unwrap_err()
            .to_string()
            .contains("replica 1"));
        golden.cases[0].targets.remove("b");
        huncho_core::conformance::save_suite(&golden, &path).unwrap();
        assert!(qualify_optimizations(&models, &args).is_err());
    }

    #[test]
    fn persistent_budget_is_divided_across_replica_contexts_without_expanding_the_model_limit() {
        let mut manifest = crate::load::mock_manifest("kev", Family::F2, "fp32", 1.0).unwrap();
        manifest.prompt_contract.template = "kev-v1".into();
        let engine = Engine::new(
            manifest,
            Box::new(huncho_backend::MockBackend::new()),
            Box::new(SimpleTokenizer::new(32768)),
            HeadParams::default(),
            BackendId::Onnx,
            "fp32",
        )
        .unwrap();
        let mut args = args();
        args.prefix_cache = true;
        args.max_batch_tokens = None;
        args.persistent_prefix_bytes = 1001;
        args.replicas = 3;
        let options = evaluation_options(&engine, &args);
        assert!(options.prefix_cache);
        assert_eq!(options.persistent_prefix_bytes, 333);
        assert!(
            options.persistent_prefix_bytes * usize::from(args.replicas)
                <= args.persistent_prefix_bytes
        );
    }

    #[test]
    fn cross_request_serving_requires_actual_mixed_batch_qualification() {
        let mut args = args();
        args.batch_max_requests = Some(4);
        let models = registry(0.);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("golden.json");
        let mut golden = suite();
        std::fs::write(&path, serde_json::to_vec(&golden).unwrap()).unwrap();
        args.qualification_golden = vec![format!("qual={}", path.display())];
        assert!(qualify_optimizations(&models, &args).is_err());
        golden.cases.push(golden.cases[0].clone());
        golden.cases[1].id = "distinct request".into();
        std::fs::write(&path, serde_json::to_vec(&golden).unwrap()).unwrap();
        assert!(qualify_optimizations(&models, &args).is_ok());
        assert!(qualify_optimizations(&registry(0.002), &args).is_err());
    }

    #[test]
    fn preparing_serving_requires_nonvacuous_paired_qualification() {
        let registry = registry(0.);
        let mut args = args();
        args.max_batch_tokens = None;
        args.max_prepared_per_model = 1;
        assert!(qualify_optimizations(&registry, &args)
            .unwrap_err()
            .to_string()
            .contains("requires --qualification-golden"));
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("golden.json");
        huncho_core::conformance::save_suite(&suite(), &path).unwrap();
        args.qualification_golden = vec![format!("qual={}", path.display())];
        qualify_optimizations(&registry, &args).unwrap();
        let report = huncho_core::conformance::run_suite_with_options(
            &registry.get("qual").unwrap(),
            &suite(),
            &Default::default(),
            &EvalOptions {
                prepare_all: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.work.prepared_questions, 2);
        assert_eq!(report.optimization_parity.unwrap().max_prob_delta, 0.);
    }

    #[test]
    fn optimized_serving_requires_actual_successful_qualification() {
        let registry = registry(0.);
        let mut args = args();
        assert!(qualify_optimizations(&registry, &args)
            .unwrap_err()
            .to_string()
            .contains("requires --qualification-golden"));
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("golden.json");
        huncho_core::conformance::save_suite(&suite(), &path).unwrap();
        args.qualification_golden = vec![format!("qual={}", path.display())];
        qualify_optimizations(&registry, &args).unwrap();
        let mut missing_batch = suite();
        missing_batch.cases[0].request.questions.shift_remove("b");
        missing_batch.cases[0].expected.remove("b");
        huncho_core::conformance::save_suite(&missing_batch, &path).unwrap();
        assert!(qualify_optimizations(&registry, &args).is_err());
        huncho_core::conformance::save_suite(&suite(), &path).unwrap();
        args.qualification_golden
            .push(format!("qual={}", path.display()));
        assert!(qualify_optimizations(&registry, &args).is_err());
        args.qualification_golden = vec![format!("unknown={}", path.display())];
        assert!(qualify_optimizations(&registry, &args).is_err());
    }

    #[test]
    fn explicit_baseline_qualification_is_checked_without_optimization_flags() {
        let registry = registry(0.);
        let mut args = args();
        args.max_batch_tokens = None;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("golden.json");
        args.qualification_golden = vec![format!("qual={}", path.display())];
        assert!(qualify_optimizations(&registry, &args).is_err());
        let mut golden = suite();
        huncho_core::conformance::save_suite(&golden, &path).unwrap();
        qualify_optimizations(&registry, &args).unwrap();
        golden.cases[0].expected.insert(
            "a".into(),
            BTreeMap::from([("a".into(), 0.8), ("b".into(), 0.2)]),
        );
        huncho_core::conformance::save_suite(&golden, &path).unwrap();
        assert!(qualify_optimizations(&registry, &args)
            .unwrap_err()
            .to_string()
            .contains("qualification failed"));
    }

    #[test]
    fn pending_is_rejected_and_refits_require_labeled_heldout_qualification() {
        let mut args = args();
        args.max_batch_tokens = None;
        assert!(qualify_optimizations(
            &registry_with_calibration(0., CalibrationStatus::Pending),
            &args
        )
        .unwrap_err()
        .to_string()
        .contains("is pending"));
        let registry = registry_with_calibration(0., CalibrationStatus::Refit);
        assert!(qualify_optimizations(&registry, &args)
            .unwrap_err()
            .to_string()
            .contains("requires --qualification-golden"));
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("golden.json");
        args.qualification_golden = vec![format!("qual={}", path.display())];
        let mut golden = suite();
        huncho_core::conformance::save_suite(&golden, &path).unwrap();
        assert!(qualify_optimizations(&registry, &args)
            .unwrap_err()
            .to_string()
            .contains("observed target labels"));
        golden.cases[0].targets =
            BTreeMap::from([("a".into(), "b".into()), ("b".into(), "b".into())]);
        huncho_core::conformance::save_suite(&golden, &path).unwrap();
        qualify_optimizations(&registry, &args).unwrap();
        golden.cases[0].targets.remove("b");
        huncho_core::conformance::save_suite(&golden, &path).unwrap();
        assert!(qualify_optimizations(&registry, &args).is_err());
    }

    #[test]
    fn changed_kernels_require_labeled_qualification_even_with_fit_metadata() {
        for (key, value) in [
            ("native_execution", "candle-qwen35-v1"),
            ("cpu_blas_execution", "openblas-lp64-fp32-v1"),
            ("projection_chunk_rows", "64"),
            ("attention_compute_dtype", "fp32"),
            ("device_path", "modernbert-cuda"),
            ("device_path", "qwen-f3-cuda"),
            ("joint_head_execution", "vectorized-v1"),
            ("joint_pool_execution", "grouped-spans-summary-v1"),
            ("onnx_execution_provider", "cuda-strict-tf32-off-v1"),
            ("onnx_intra_threads", "4"),
            ("onnx_native_batch", "equal-length-v1"),
            ("onnx_initializer_residency", "immutable-cpu-v1"),
            ("delta_rule_execution", "cpu-buffered-v1"),
            ("causal_conv_execution", "cpu-buffered-v1"),
            ("mlp_gate_execution", "cpu-fused-silu-mul-v1"),
            ("cpu_kernel_build", "x86_64:avx,avx2,f16c,fma"),
            ("llamacpp_execution", "cpu-qwen35-masked-prefill-v1"),
        ] {
            let registry = registry_with_execution_metadata(
                0.0,
                CalibrationStatus::Fit,
                BTreeMap::from([(key.into(), value.into())]),
            );
            let mut args = args();
            args.max_batch_tokens = None;
            assert!(qualify_optimizations(&registry, &args)
                .unwrap_err()
                .to_string()
                .contains("requires --qualification-golden"));
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("golden.json");
            args.qualification_golden = vec![format!("qual={}", path.display())];
            let mut golden = suite();
            huncho_core::conformance::save_suite(&golden, &path).unwrap();
            assert!(qualify_optimizations(&registry, &args)
                .unwrap_err()
                .to_string()
                .contains("observed target labels"));
            golden.cases[0].targets =
                BTreeMap::from([("a".into(), "b".into()), ("b".into(), "b".into())]);
            huncho_core::conformance::save_suite(&golden, &path).unwrap();
            qualify_optimizations(&registry, &args).unwrap();
            let report = huncho_core::conformance::run_suite(
                &registry.get("qual").unwrap(),
                &golden,
                &Default::default(),
            )
            .unwrap();
            assert_eq!(report.execution_metadata[key], value);
        }
    }

    #[test]
    fn readout_and_storage_profiles_require_fresh_numerical_qualification() {
        for (key, value) in [
            ("onnx_readout", "gather-v1"),
            ("onnx_output_buffer_bytes", "4096"),
            ("base_weight_cache", "content-checked-cpu-v1"),
        ] {
            let registry = registry_with_execution_metadata(
                0.,
                CalibrationStatus::Fit,
                BTreeMap::from([(key.into(), value.into())]),
            );
            let mut args = args();
            args.max_batch_tokens = None;
            assert!(qualify_optimizations(&registry, &args)
                .unwrap_err()
                .to_string()
                .contains("requires --qualification-golden"));
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("golden.json");
            huncho_core::conformance::save_suite(&suite(), &path).unwrap();
            args.qualification_golden = vec![format!("qual={}", path.display())];
            qualify_optimizations(&registry, &args).unwrap();
        }
    }

    #[test]
    fn paired_gate_rejects_drift_that_passes_the_external_golden_threshold() {
        let registry = registry(0.0008);
        let engine = registry.get("qual").unwrap();
        let report = huncho_core::conformance::run_suite_with_options(
            &engine,
            &suite(),
            &Default::default(),
            &EvalOptions {
                max_batch_tokens: Some(1024),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(report.max_prob_delta < 0.001);
        assert_eq!(report.argmax_agreement, 1.);
        assert!(report.ece < 0.02);
        assert!(report.optimization_parity.unwrap().max_prob_delta > 1e-4);
        assert!(!report.passed);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("golden.json");
        huncho_core::conformance::save_suite(&suite(), &path).unwrap();
        let mut args = args();
        args.qualification_golden = vec![format!("qual={}", path.display())];
        assert!(qualify_optimizations(&registry, &args)
            .unwrap_err()
            .to_string()
            .contains("qualification failed"));
    }
}

#[cfg(all(test, feature = "clef", feature = "onnx"))]
mod tests {
    use super::*;
    use clap::Parser;
    use huncho_core::manifest::ArtifactRef;
    use std::{fs, path::Path};

    #[test]
    fn one_server_selects_a_different_runtime_for_each_model() {
        let root = tempfile::tempdir().unwrap();
        let fixture = Path::new("../huncho-backend/tests/fixtures/tiny_modernbert");
        for file in ["config.json", "model.safetensors"] {
            fs::copy(fixture.join(file), root.path().join(file)).unwrap();
        }
        let mut manifest =
            crate::load::mock_manifest("tiny-candle", Family::F1, "fp32", 1.0).unwrap();
        manifest.backbone.hidden_size = 8;
        manifest.backbone.artifacts.insert(
            BackendId::Candle,
            vec![ArtifactRef {
                path: "model.safetensors".into(),
                dtype: "fp32".into(),
                quantization: None,
            }],
        );
        fs::write(
            root.path().join("huncho-model.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        // Routing-only test setup simulates an already-fitted package. Keep
        // the checked-in numerical fixture Pending and the startup gate strict.
        let clef_root = root.path().join("clef");
        fs::create_dir(&clef_root).unwrap();
        let fixture = Path::new("../huncho-backend/tests/fixtures/tiny_clef");
        for file in [
            "config.json",
            "model.safetensors",
            "joint_head_config.json",
            "joint_head.safetensors",
            "tokenizer.json",
        ] {
            fs::copy(fixture.join(file), clef_root.join(file)).unwrap();
        }
        let mut clef_manifest =
            huncho_core::manifest::ModelManifest::load(fixture.join("huncho-model.json")).unwrap();
        clef_manifest.calibration.default.status = huncho_core::manifest::CalibrationStatus::Fit;
        let clef_manifest_path = clef_root.join("huncho-model.json");
        fs::write(
            &clef_manifest_path,
            serde_json::to_vec(&clef_manifest).unwrap(),
        )
        .unwrap();
        let crate::Command::Serve(args) = crate::Cli::try_parse_from([
            "huncho",
            "serve",
            "--backend",
            "auto",
            "--dtype",
            "fp32",
            "--model",
            root.path().to_str().unwrap(),
            "--manifest",
            "../../examples/mock-model/huncho-model.json",
            "--manifest",
            clef_manifest_path.to_str().unwrap(),
        ])
        .unwrap()
        .command
        else {
            panic!("expected serve arguments")
        };
        let registry = load_models(&args).unwrap();
        assert_eq!(registry.len(), 3);
        assert_eq!(
            registry.get("tiny-candle").unwrap().backend_id(),
            BackendId::Candle
        );
        assert_eq!(
            registry.get("mock-laya").unwrap().backend_id(),
            BackendId::Onnx
        );
        assert_eq!(
            registry.get("tiny-clef").unwrap().backend_id(),
            BackendId::Clef
        );
    }
}
