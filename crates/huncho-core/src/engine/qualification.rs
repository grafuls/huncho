//! Fresh in-process authorization for one immutable execution context/profile.
//! Diagnostic conformance reports and serialized receipts cannot create it.
use super::{Engine, EvalOptions, EvalStats};
use crate::conformance::{
    run_suite_with_cross_request_batches, run_suite_with_options, ConformanceReport,
    ConformanceThresholds, GoldenSuite,
};
use crate::contract::{SystemOneRequest, SystemOneResponse};
use crate::error::{Error, Result};
use crate::manifest::{BackendId, CalibrationStatus};
use std::sync::{Arc, Mutex};

struct QualifiedProfile {
    key: Vec<u8>,
    proof: Arc<()>,
}

/// Opaque in-process lease for one actual context and qualification attempt.
/// It cannot be constructed/deserialized from a diagnostic report or receipt.
pub struct ServingQualificationToken {
    context: Arc<()>,
    proof: Option<Arc<()>>,
}

#[derive(Default)]
pub(super) struct ServingQualification {
    // Keep the authorization lookup short even during lengthy qualification.
    // A separate lock serializes attempts on this execution context.
    attempt: Mutex<()>,
    profile: Mutex<Option<QualifiedProfile>>,
    identity: Arc<()>,
}

fn profile_key(
    options: &EvalOptions,
    cross_request_max_requests: Option<usize>,
) -> Result<Vec<u8>> {
    let mut options = options.clone();
    // Response presentation does not change model arithmetic or scheduling.
    options.extensions = false;
    let mut environment = Vec::new();
    for name in [
        "RAYON_NUM_THREADS",
        "CANDLE_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "HUNCHO_DEVICE",
        "HUNCHO_CLEF_DEVICE",
        "HUNCHO_COMPILED_CPU_KERNELS",
        "HUNCHO_PROJECTION_CHUNK_ROWS",
        "HUNCHO_ATTENTION_FP32",
        "HUNCHO_ATTENTION_QUERY_ROWS",
        "HUNCHO_GROUPED_GQA",
        "HUNCHO_KV_PAGE_TOKENS",
        "HUNCHO_DIRECT_PAGED_ATTENTION",
        "HUNCHO_RUNTIME_LORA",
        "HUNCHO_CPU_DELTA_RULE",
        "HUNCHO_CPU_CAUSAL_CONV",
        "HUNCHO_CPU_FUSED_GATE",
        "HUNCHO_CPU_BLAS_LIBRARY",
        "HUNCHO_CPU_BLAS_THREADS",
        "HUNCHO_PREFILL_CHUNK_TOKENS",
        "HUNCHO_CLEF_VECTOR_HEAD",
        "HUNCHO_CLEF_GROUPED_POOL",
        "HUNCHO_LAYA_SELECTED_HEAD",
        "HUNCHO_BASE_CACHE_BYTES",
        "HUNCHO_ONNX_COMPACT_READOUT",
        "HUNCHO_ONNX_INTEGRATED_HEAD",
        "HUNCHO_ONNX_OUTPUT_BUFFER_BYTES",
        "HUNCHO_ONNX_DEVICE_IO_BYTES",
        "HUNCHO_ONNX_CUDA_GRAPH",
        "HUNCHO_ONNX_EP",
        "HUNCHO_ONNX_THREADS",
        "HUNCHO_ONNX_NATIVE_BATCH",
        "HUNCHO_ONNX_SHARED_INITIALIZERS",
        "HUNCHO_LLAMA_THREADS",
        "HUNCHO_LLAMA_BATCH_ROWS",
        "HUNCHO_VLLM_PYTHON",
        "HUNCHO_VLLM_THREADS",
        "HUNCHO_VLLM_BATCH_ROWS",
        "HUNCHO_VLLM_KV_BYTES",
        "HUNCHO_VLLM_TIMEOUT_SECS",
        "HUNCHO_VLLM_TENSOR_PARALLEL",
        "ORT_DYLIB_PATH",
        "ORT_PREFER_DYNAMIC_LINK",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    ] {
        let value = match std::env::var(name) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => {
                return Err(Error::Conformance(
                    "invalid serving runtime environment".into(),
                ))
            }
        };
        environment.push((name, value));
    }
    Ok(serde_json::to_vec(&(
        options,
        cross_request_max_requests,
        environment,
    ))?)
}

impl Engine {
    /// Native runtimes and changed arithmetic profiles need complete observed
    /// outcomes, including profiles with upstream `fit` metadata.
    pub fn requires_outcome_qualification(&self) -> bool {
        [
            "native_execution",
            "projection_chunk_rows",
            "attention_compute_dtype",
            "attention_execution",
            "gqa_execution",
            "kv_storage",
            "paged_attention",
            "adapter_execution",
            "device_path",
            "joint_head_execution",
            "joint_pool_execution",
            "request_batch_execution",
            "vllm_readout",
            "onnx_execution_provider",
            "onnx_intra_threads",
            "onnx_native_batch",
            "onnx_integrated_head",
            "onnx_initializer_residency",
            "onnx_device_io",
            "onnx_cuda_graph",
            "delta_rule_execution",
            "causal_conv_execution",
            "mlp_gate_execution",
            "prefill_chunk_tokens",
            "weight_quantization",
            "cpu_kernel_build",
            "cpu_blas_execution",
            "laya_head_execution",
            "llamacpp_execution",
            "llamacpp_prefix_state",
        ]
        .iter()
        .any(|key| self.execution_metadata.contains_key(*key))
    }

    fn validate_serving_calibration(&self) -> Result<()> {
        if self.calibration.status == CalibrationStatus::Pending {
            return Err(Error::Conformance(
                "serving requires fitted calibration; variant is pending".into(),
            ));
        }
        let exact = self.manifest.calibration.entries.get(
            &self
                .manifest
                .calibration_key(&self.backend_id.to_string(), &self.dtype),
        );
        if self.execution_metadata.contains_key("weight_quantization")
            && !exact.is_some_and(|entry| entry.status == CalibrationStatus::Refit)
        {
            return Err(Error::Conformance(
                "quantized serving requires an explicit backend:dtype refit".into(),
            ));
        }
        if (self.backend_id == BackendId::Vllm
            || self.execution_metadata.contains_key("onnx_integrated_head"))
            && !exact.is_some_and(|entry| entry.status != CalibrationStatus::Pending)
        {
            return Err(Error::Conformance(
                "this runtime requires an explicit fitted/refitted backend:dtype entry".into(),
            ));
        }
        Ok(())
    }

    /// Run fresh complete labeled conformance at the fixed serving thresholds.
    /// Only a passing run authorizes this exact context/options/runtime. A new
    /// attempt revokes its previous profile, including on error or rejection.
    /// Replicas and newly loaded contexts must each run their own complete suite.
    /// This is an in-process proof; an unsigned report/receipt cannot install it.
    pub fn qualify_for_serving(
        &self,
        suite: &GoldenSuite,
        options: &EvalOptions,
        cross_request_max_requests: Option<usize>,
    ) -> Result<ConformanceReport> {
        let state = &self.serving_qualification;
        let _attempt = state
            .attempt
            .lock()
            .map_err(|_| Error::Conformance("qualification attempt lock poisoned".into()))?;
        *state
            .profile
            .lock()
            .map_err(|_| Error::Conformance("qualification state poisoned".into()))? = None;
        // Diagnostic evaluation may have retained results before this fresh
        // gate, or under an earlier environment. Never promote those entries.
        if let Some(cache) = &self.response_cache {
            cache
                .lock()
                .map_err(|_| {
                    Error::Conformance("response cache poisoned during qualification".into())
                })?
                .clear()?;
        }
        self.validate_serving_calibration()?;
        if suite.cases.is_empty()
            || suite.cases.iter().any(|case| {
                case.request.questions.is_empty()
                    || case.targets.len() != case.request.questions.len()
                    || case
                        .request
                        .questions
                        .keys()
                        .any(|id| !case.targets.contains_key(id))
            })
        {
            return Err(Error::Conformance(
                "serving qualification requires complete observed target labels for every question"
                    .into(),
            ));
        }
        if (self.execution_metadata.contains_key("kv_storage")
            || self.execution_metadata.contains_key("prefill_chunk_tokens"))
            && !options.prefix_cache
        {
            return Err(Error::Conformance(
                "this prefix profile requires actual prefix/fork qualification".into(),
            ));
        }
        let key = profile_key(options, cross_request_max_requests)?;
        let thresholds = ConformanceThresholds::default();
        let report = match cross_request_max_requests {
            Some(rows) => {
                run_suite_with_cross_request_batches(self, suite, &thresholds, options, rows)?
            }
            None => run_suite_with_options(self, suite, &thresholds, options)?,
        };
        if profile_key(options, cross_request_max_requests)? != key {
            return Err(Error::Conformance(
                "runtime environment changed during qualification".into(),
            ));
        }
        if report.passed
            && report
                .outcome_calibration
                .as_ref()
                .is_some_and(|outcomes| outcomes.questions > 0)
        {
            *state
                .profile
                .lock()
                .map_err(|_| Error::Conformance("qualification state poisoned".into()))? =
                Some(QualifiedProfile {
                    key,
                    proof: Arc::new(()),
                });
        }
        Ok(report)
    }

    /// Reject unqualified native/refitted contexts before any serving work or
    /// exact-result reuse. Pure offline mock profiles retain their demo path.
    /// All built-in real runtimes declare `native_execution`; custom runtimes
    /// must declare it too when exposing calibrated production inference.
    pub fn require_serving_qualification(
        &self,
        options: &EvalOptions,
        cross_request_max_requests: Option<usize>,
    ) -> Result<()> {
        self.serving_qualification_token(options, cross_request_max_requests)
            .map(|_| ())
    }

    /// Retain before serving work; validate again before publishing its result.
    pub fn serving_qualification_token(
        &self,
        options: &EvalOptions,
        cross_request_max_requests: Option<usize>,
    ) -> Result<ServingQualificationToken> {
        self.validate_serving_calibration()?;
        let context = self.serving_qualification.identity.clone();
        if !self.requires_outcome_qualification()
            && self.calibration.status != CalibrationStatus::Refit
        {
            return Ok(ServingQualificationToken {
                context,
                proof: None,
            });
        }
        let key = profile_key(options, cross_request_max_requests)?;
        let profile = self
            .serving_qualification
            .profile
            .lock()
            .map_err(|_| Error::Conformance("qualification state poisoned".into()))?;
        if !profile.as_ref().is_some_and(|profile| profile.key == key) {
            return Err(Error::Conformance(
                "model execution context/options/runtime lack fresh labeled serving qualification"
                    .into(),
            ));
        }
        Ok(ServingQualificationToken {
            context,
            proof: profile.as_ref().map(|profile| profile.proof.clone()),
        })
    }

    /// Refuse even if a later attempt requalified the same execution options.
    pub fn validate_serving_qualification_token(
        &self,
        token: &ServingQualificationToken,
        options: &EvalOptions,
        cross_request_max_requests: Option<usize>,
    ) -> Result<()> {
        let current = self.serving_qualification_token(options, cross_request_max_requests)?;
        let same_proof = match (&token.proof, &current.proof) {
            (Some(old), Some(new)) => Arc::ptr_eq(old, new),
            (None, None) => true,
            _ => false,
        };
        if !Arc::ptr_eq(&token.context, &current.context) || !same_proof {
            return Err(Error::Conformance(
                "serving qualification changed while evaluation was in progress".into(),
            ));
        }
        Ok(())
    }

    /// Calibrated production evaluation after exact fresh context qualification.
    /// `eval` and raw/prepared APIs remain available for fitting/diagnostics.
    pub fn eval_for_serving(
        &self,
        req: &SystemOneRequest,
        options: &EvalOptions,
    ) -> Result<SystemOneResponse> {
        self.eval_for_serving_with_stats(req, options, &mut EvalStats::default())
    }

    pub fn eval_for_serving_with_stats(
        &self,
        req: &SystemOneRequest,
        options: &EvalOptions,
        stats: &mut EvalStats,
    ) -> Result<SystemOneResponse> {
        *stats = EvalStats::default();
        let token = self.serving_qualification_token(options, None)?;
        let response = self.eval_with_stats(req, options, stats)?;
        self.validate_serving_qualification_token(&token, options, None)?;
        Ok(response)
    }
}
