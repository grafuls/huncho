//! CPU-only paired diagnostic on an explicit local Laya checkpoint.
//! No observed labels or external reference: never a qualification certificate.
use huncho_backend::CandleBackend;
use huncho_core::{
    backend::Backend,
    contract::{Answer, SystemOneRequest},
    engine::{Engine, EvalOptions},
    head::HeadParams,
    manifest::{BackendId, Family, ModelManifest},
    tokenizer::HfTokenizer,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    path::Path,
    time::Instant,
};

fn hash(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut sha = Sha256::new();
    let mut block = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        sha.update(&block[..n]);
    }
    Ok(format!("{:x}", sha.finalize()))
}

fn distribution(answer: &Answer) -> BTreeMap<String, f32> {
    match answer {
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
            probabilities.clone()
        }
        Answer::Noul { noul } => BTreeMap::from([("yes".into(), *noul), ("no".into(), 1. - noul)]),
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(
        args.len() == 2,
        "usage: laya_profile LOCAL_PACKAGE NEW_REPORT.json"
    );
    let root = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    anyhow::ensure!(!output.exists(), "diagnostic report already exists");
    let manifest = ModelManifest::load(root.join("huncho-model.json"))?;
    anyhow::ensure!(
        manifest.family == Family::F1 && manifest.prompt_contract.template == "laya-v1",
        "requires Laya F1"
    );
    let weights = root.join(
        &manifest
            .find_artifact(BackendId::Candle, "fp32")
            .ok_or_else(|| anyhow::anyhow!("requires FP32 Candle artifact"))?
            .path,
    );
    let tokenizer = root.join(
        manifest
            .backbone
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("requires real tokenizer"))?,
    );
    let paths = [
        root.join("huncho-model.json"),
        root.join("config.json"),
        weights.clone(),
        tokenizer.clone(),
    ];
    let before: BTreeMap<_, _> = paths
        .iter()
        .map(|p| Ok((p.display().to_string(), hash(p)?)))
        .collect::<anyhow::Result<_>>()?;
    let load = |selected| -> anyhow::Result<Engine> {
        let backend = CandleBackend::load(
            root.join("config.json"),
            &weights,
            manifest.backbone.max_context,
            "fp32",
        )?
        .with_selected_laya_head(selected)?;
        let _ = backend.capabilities();
        Ok(Engine::new(
            manifest.clone(),
            Box::new(backend),
            Box::new(HfTokenizer::from_file(&tokenizer)?),
            HeadParams::default(),
            BackendId::Candle,
            "fp32",
        )?)
    };
    let baseline = load(false)?;
    let selected = load(true)?;
    let states = ["The service responds correctly and meets its deadline.".to_owned(), "The service responds correctly and meets its deadline. Sometimes a retry occurs and the operator reviews the result. ".repeat(32)];
    let requests: Vec<SystemOneRequest> = states.iter().map(|state| serde_json::from_value(json!({"model":manifest.name,"state":state,"questions":{
        "choice":{"type":"choice","instructions":"Choose the next action.","criteria":{"accept":"Accept the result","review":"Ask for review","retry":"Retry the operation"}},
        "score":{"type":"score","instructions":"Rate the quality.","criteria":["poor","fair","good","excellent"]},
        "noul":{"type":"noul","instructions":"Is the service ready?"}
    }}))).collect::<Result<_, _>>()?;
    let opts = EvalOptions {
        extensions: true,
        ..Default::default()
    };
    let mut rows = Vec::<Value>::new();
    let mut max_delta = 0f32;
    let mut max_raw_delta = 0f32;
    let mut match_count = 0usize;
    let mut total_count = 0usize;
    for request in &requests {
        let a = baseline.eval(request, &opts)?;
        let b = selected.eval(request, &opts)?;
        anyhow::ensure!(
            a.usage.input_tokens == b.usage.input_tokens
                && a.usage.output_tokens == b.usage.output_tokens,
            "logical usage differs"
        );
        for (key, answer) in &a.answers {
            let actual = &b.answers[key];
            let expected_probs = distribution(answer);
            let actual_probs = distribution(actual);
            anyhow::ensure!(
                expected_probs.keys().eq(actual_probs.keys()),
                "candidate labels differ"
            );
            for (label, p) in &expected_probs {
                max_delta = max_delta.max((p - actual_probs[label]).abs());
            }
            let ar = &a.extensions.as_ref().unwrap().raw_logits.as_ref().unwrap()[key];
            let br = &b.extensions.as_ref().unwrap().raw_logits.as_ref().unwrap()[key];
            anyhow::ensure!(ar.len() == br.len(), "readout widths differ");
            for (a, b) in ar.iter().zip(br) {
                max_raw_delta = max_raw_delta.max((a - b).abs());
            }
            match_count += usize::from(
                huncho_core::calibration::argmax(ar) == huncho_core::calibration::argmax(br),
            );
            total_count += 1;
        }
        rows.push(json!({"request":request,"baseline":a,"selected":b}));
    }
    let mut timings = Vec::new();
    // Warm each engine, then alternate the order. Native calls complete before
    // the next run; no concurrent Cargo/runtime/GPU task is part of this pilot.
    for engine in [&baseline, &selected] {
        for request in &requests {
            engine.eval(request, &opts)?;
        }
    }
    for repeat in 0..4 {
        for selected_first in [false, true] {
            let selected_path = selected_first != (repeat % 2 == 0);
            let engine = if selected_path { &selected } else { &baseline };
            let start = Instant::now();
            for request in &requests {
                engine.eval(request, &opts)?;
            }
            timings.push(json!({"repeat":repeat,"selected":selected_path,"seconds":start.elapsed().as_secs_f64(),"requests":requests.len(),"questions":total_count}));
        }
    }
    for path in &paths {
        anyhow::ensure!(
            before[&path.display().to_string()] == hash(path)?,
            "diagnostic input changed"
        );
    }
    let report = json!({"qualified":false,"basis":"paired CPU diagnostic, synthetic unlabeled inputs; no independent external golden or outcome calibration", "model_files":before,"binary_sha256":hash(&std::env::current_exe()?)?, "source_sha256":{"example":format!("{:x}",Sha256::digest(include_str!("laya_profile.rs"))),"backend":format!("{:x}",Sha256::digest(include_str!("../../huncho-backend/src/candle.rs")))}, "baseline_profile":baseline.execution_metadata(),"selected_profile":selected.execution_metadata(),"threads":{"RAYON_NUM_THREADS":std::env::var("RAYON_NUM_THREADS").ok(),"CANDLE_NUM_THREADS":std::env::var("CANDLE_NUM_THREADS").ok()},"max_probability_delta":max_delta,"max_raw_delta":max_raw_delta,"argmax_agreement":match_count as f64 / total_count as f64,"paired_passed":max_delta <= 1e-4 && match_count == total_count,"cases":rows,"timings":timings});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    anyhow::ensure!(
        max_delta <= 1e-4 && match_count == total_count,
        "selected head failed unchanged paired gates; raw evidence retained"
    );
    Ok(())
}
