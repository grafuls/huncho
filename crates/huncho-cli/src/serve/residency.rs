//! Explicit CPU lazy loading. No previous pass authorizes a changed package.
use super::*;
use crate::qualification::{hash_file, InputSnapshot};
use huncho_api::ModelDescription;
use huncho_core::conformance::load_suite;
use huncho_core::error::Error;
use huncho_core::manifest::ModelManifest;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

fn environment() -> BTreeMap<OsString, OsString> {
    std::env::vars_os()
        .filter(|(name, _)| {
            let name = name.to_string_lossy();
        (name.starts_with("HUNCHO_") && name != "HUNCHO_AUTH_TOKEN")
                || name.starts_with("ORT_")
                || [
                    "RAYON_NUM_THREADS",
                    "CANDLE_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                ]
                .contains(&name.as_ref())
        })
        .collect()
}

pub(super) fn load_lazy(args: &ServeArgs) -> anyhow::Result<ModelRegistry> {
    validate_scheduling(args)?;
    anyhow::ensure!(
        !args.mock,
        "lazy serving supports real pinned packages; --mock is eager"
    );
    anyhow::ensure!(
        std::env::var("HUNCHO_DEVICE").as_deref() == Ok("cpu"),
        "lazy serving currently requires explicit HUNCHO_DEVICE=cpu"
    );
    anyhow::ensure!(
        std::env::var("HUNCHO_ONNX_EP").map_or(true, |ep| ep == "cpu"),
        "lazy serving does not support GPU execution providers"
    );
    let BackendChoice::Explicit(backend) = BackendChoice::parse(&args.backend)? else {
        anyhow::bail!("lazy serving requires an explicit --backend and --dtype");
    };
    backend.require_available(&crate::load::available_backends())?;
    let dtype = args
        .dtype
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("lazy serving requires an explicit --dtype"))?;
    let mut paths: Vec<PathBuf> = args.manifest.iter().map(PathBuf::from).collect();
    for model in &args.model {
        paths.push(crate::load::resolve_model(
            model,
            BackendChoice::Explicit(backend),
            Some(dtype),
            args.revision.clone(),
            args.token.clone(),
            args.cache_dir.clone(),
            false,
        )?);
    }
    let mut golden_bindings = BTreeMap::new();
    for binding in &args.qualification_golden {
        let (name, path) = binding
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("qualification golden must be MODEL=PATH"))?;
        anyhow::ensure!(
            !name.is_empty() && !path.is_empty() && golden_bindings.insert(name, path).is_none(),
            "invalid or duplicate qualification binding"
        );
    }
    // Parse all receipts before listening, including unknown fields/version.
    let records = RecordBindings::load(&args.qualification_record)?;
    let runtime_environment = environment();
    let mut registry = ModelRegistry::new();
    registry.enable_lazy(
        usize::from(args.resident_models),
        Duration::from_secs(args.idle_evict_secs),
    )?;
    let mut names = BTreeSet::new();
    for path in paths {
        let path = path.canonicalize()?;
        let manifest = ModelManifest::load(&path)?;
        let name = manifest.name.clone();
        anyhow::ensure!(names.insert(name.clone()), "duplicate lazy model `{name}`");
        anyhow::ensure!(
            manifest
                .calibration
                .resolve(&backend.to_string(), dtype)
                .status
                != CalibrationStatus::Pending,
            "lazy serving requires fitted calibration for `{name}`"
        );
        let golden = golden_bindings.get(name.as_str()).ok_or_else(|| anyhow::anyhow!("lazy serving requires --qualification-golden {name}=PATH with complete observed labels"))?;
        let suite = load_suite(golden)?;
        anyhow::ensure!(
            !suite.cases.is_empty()
                && suite.family == manifest.family.to_string()
                && suite.cases.iter().all(|case| {
                    case.request.model == name
                        && !case.expected.is_empty()
                        && case.targets.len() == case.expected.len()
                        && case.expected.len() == case.request.questions.len()
                        && case.expected.iter().all(|(id, probabilities)| {
                            case.request.questions.contains_key(id)
                                && case
                                    .targets
                                    .get(id)
                                    .is_some_and(|target| probabilities.contains_key(target))
                        })
                }),
            "lazy serving requires complete observed-label vectors for every question of `{name}`"
        );
        let inputs =
            InputSnapshot::capture(&path, &manifest, backend, dtype, path.parent().unwrap())?;
        let mut pinned_paths = vec![PathBuf::from(golden), std::env::current_exe()?];
        for binding in &args.qualification_record {
            let (model, record) = binding.split_once('=').unwrap();
            if model == name {
                pinned_paths.push(PathBuf::from(record));
            }
        }
        for key in ["HUNCHO_CPU_BLAS_LIBRARY", "ORT_DYLIB_PATH"] {
            if let Some(path) = std::env::var_os(key) {
                pinned_paths.push(PathBuf::from(path));
            }
        }
        let pins = pinned_paths
            .into_iter()
            .map(|path| Ok((path.clone(), hash_file(&path)?)))
            .collect::<huncho_core::Result<Vec<_>>>()?;
        let mut isolated = args.clone();
        isolated.manifest = vec![path.to_string_lossy().into_owned()];
        isolated.model.clear();
        isolated.auth_token = None;
        isolated.token = None;
        isolated.preload.clear();
        isolated.lazy = false;
        isolated
            .qualification_golden
            .retain(|binding| binding.split_once('=').unwrap().0 == name);
        isolated
            .qualification_record
            .retain(|binding| binding.split_once('=').unwrap().0 == name);
        let expected_environment = runtime_environment.clone();
        registry.register_lazy(
            ModelDescription {
                name: name.clone(),
                family: manifest.family,
                backend,
                dtype: dtype.into(),
                max_context: manifest.backbone.max_context,
                replicas: usize::from(args.replicas),
                residency: "cold".into(),
            },
            move || {
                let recheck = || -> huncho_core::Result<()> {
                    if environment() != expected_environment {
                        return Err(Error::Conformance(
                            "runtime environment changed since lazy registration".into(),
                        ));
                    }
                    inputs.recheck()?;
                    for (path, digest) in &pins {
                        if hash_file(path)? != *digest {
                            return Err(Error::Conformance(format!(
                                "pinned lazy input changed: {}",
                                path.display()
                            )));
                        }
                    }
                    Ok(())
                };
                recheck()?;
                // Includes independent/optimized/cross-request/replica gates and
                // retained receipt verification. No engine escapes before success.
                let loaded = load_models(&isolated)
                    .map_err(|error| Error::Conformance(error.to_string()))?;
                recheck()?;
                let model = loaded
                    .get(&name)
                    .ok_or_else(|| Error::ModelNotFound(name.clone()))?;
                Ok(model.replica_engines().to_vec())
            },
        )?;
    }
    anyhow::ensure!(
        golden_bindings.keys().all(|name| names.contains(*name))
            && records.records.keys().all(|name| names.contains(name)),
        "qualification binding names an unregistered lazy model"
    );
    anyhow::ensure!(
        args.preload.len() <= usize::from(args.resident_models)
            && args.preload.iter().all(|name| names.contains(name))
            && args.preload.iter().collect::<BTreeSet<_>>().len() == args.preload.len(),
        "preload names must be unique registered models and fit resident slots"
    );
    anyhow::ensure!(
        !names.is_empty(),
        "lazy serving requires at least one pinned model package"
    );
    Ok(registry)
}
