use std::path::PathBuf;

use huncho_backend::CandleBackend;
use huncho_core::contract::{Question, StateValue};
use huncho_core::engine::{Engine, EvalOptions};
use huncho_core::head::HeadParams;
use huncho_core::manifest::ModelManifest;
use huncho_core::prompt::formatter_for;
use huncho_core::tokenizer::HfTokenizer;
use huncho_core::BackendId;

/// Translate a Laya-style question dict (`t`/`ins`/`crit`) into huncho's
/// `Question` JSON (`type`/`instructions`/`criteria`).
fn to_question(q: &serde_json::Value) -> serde_json::Value {
    let t = q["t"].as_str().unwrap();
    let ins = q.get("ins").cloned().unwrap_or(serde_json::json!(""));
    match t {
        "choice" => serde_json::json!({
            "type": "choice",
            "instructions": ins,
            "criteria": q.get("crit").cloned().unwrap_or(serde_json::json!({})),
        }),
        "score" => serde_json::json!({
            "type": "score",
            "instructions": ins,
            "criteria": q.get("crit").cloned().unwrap_or(serde_json::json!([])),
        }),
        "noul" => {
            let mut v = serde_json::json!({
                "type": "noul",
                "instructions": ins,
            });
            if let Some(crit) = q.get("crit") {
                v["criteria"] = crit.clone();
            }
            v
        }
        other => panic!("unknown question type {other}"),
    }
}

fn main() -> anyhow::Result<()> {
    let cases_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/laya_cases.json".into());
    let dir = PathBuf::from("my-laya");

    let manifest = ModelManifest::load(dir.join("huncho-model.json"))?;
    let reporting_tok = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    let formatter = formatter_for(&manifest);

    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cases_path)?)?;

    // Report the token sequence + marker positions the formatter emits, per case.
    for c in cases.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let statev = StateValue::new(c["state"].clone());
        let q: Question = serde_json::from_value(to_question(&c["q"]))?;
        let prompt = formatter.build(&statev, &q, &reporting_tok)?;
        let markers: Vec<usize> = prompt.candidates.iter().map(|c| c.position).collect();
        println!("=== {name} ===");
        println!("ids      = {:?}", prompt.tokens);
        println!("markers  = {markers:?}");
        println!("qtype    = {}", prompt.qtype);
    }

    let backend = CandleBackend::load(
        dir.join("config.json"),
        dir.join("model.safetensors"),
        manifest.backbone.max_context,
        "fp32",
    )?;
    let engine_tok = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    let engine = Engine::new(
        manifest.clone(),
        Box::new(backend),
        Box::new(engine_tok),
        HeadParams::default(),
        BackendId::Candle,
        "fp32",
    )?;

    for c in cases.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let req = serde_json::from_value(serde_json::json!({
            "state": c["state"],
            "model": "laya",
            "questions": { name: to_question(&c["q"]) },
        }))?;
        let resp = engine.eval(
            &req,
            &EvalOptions {
                extensions: true,
                ..Default::default()
            },
        )?;
        let logits = resp
            .extensions
            .as_ref()
            .and_then(|e| e.raw_logits.as_ref())
            .and_then(|m| m.get(name))
            .cloned()
            .unwrap_or_default();
        println!("{name} logits = {logits:?}");
    }
    Ok(())
}
