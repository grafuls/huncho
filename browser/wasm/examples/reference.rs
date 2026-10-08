//! Independent native fixture arithmetic; never a released-model reference.
use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use huncho_core::conformance::{GoldenCase, GoldenSuite};
use huncho_core::contract::{Answer, SystemOneRequest};
use huncho_core::engine::{Engine, EvalOptions};
use huncho_core::error::{Error, Result};
use huncho_core::head::HeadParams;
use huncho_core::manifest::{BackendId, Family, ModelManifest};
use huncho_core::tensor::Tensor;
use huncho_core::tokenizer::HfTokenizer;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Weights {
    embedding: Vec<Vec<f32>>,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}
struct Independent(Weights);
impl Backend for Independent {
    fn id(&self) -> BackendId {
        BackendId::Onnx
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: BackendId::Onnx,
            dtype: "fp32".into(),
            families: vec![Family::F1],
            max_context: 128,
            ..Default::default()
        }
    }
    fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
        let w = &self.0;
        let mut context = [0.0f32; 4];
        for &token in &input.tokens {
            for (j, value) in context.iter_mut().enumerate() {
                *value += w.embedding[token as usize][j];
            }
        }
        for value in &mut context {
            *value /= input.tokens.len() as f32;
        }
        let mut values = Vec::new();
        for &position in &input.positions {
            let token = input.tokens[position + 1] as usize;
            let mut hidden = [0.0f32; 3];
            for (i, value) in hidden.iter_mut().enumerate() {
                for (j, context) in context.iter().enumerate() {
                    *value +=
                        (w.embedding[token][j] + context) * (input.qtype + 1) as f32 * w.w1[j][i];
                }
                *value = (*value + w.b1[i]).max(0.0);
            }
            let score = hidden
                .iter()
                .zip(&w.w2)
                .map(|(x, weight)| x * weight)
                .sum::<f32>()
                + w.b2
                + position as f32 * 0.023;
            values.push(score);
        }
        Ok(ForwardOutput::Features {
            values: Tensor::new(vec![values.len(), 1], values)?,
            positions: input.positions,
        })
    }
    fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported("fixture".into()))
    }
}
fn probabilities(answer: &Answer) -> BTreeMap<String, f32> {
    match answer {
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
            probabilities.clone()
        }
        Answer::Noul { noul } => BTreeMap::from([("no".into(), 1.0 - noul), ("yes".into(), *noul)]),
    }
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("fixture directory required")?,
    );
    let read = |name| std::fs::read(path.join(name));
    let manifest: ModelManifest = serde_json::from_slice(&read("huncho-model.json")?)?;
    let weights = serde_json::from_slice(&read("weights.json")?)?;
    let tokenizer = HfTokenizer::from_json(&read("tokenizer.json")?)?;
    let engine = Engine::new(
        manifest,
        Box::new(Independent(weights)),
        Box::new(tokenizer),
        HeadParams::scalar_linear(1, vec![1.0], 0.0)?,
        BackendId::Onnx,
        "fp32",
    )?;
    let requests: Vec<SystemOneRequest> = serde_json::from_slice(&read("requests.json")?)?;
    let mut cases = Vec::new();
    let mut responses = Vec::new();
    let mut inputs = Vec::new();
    for (index, request) in requests.into_iter().enumerate() {
        let response = engine.eval(
            &request,
            &EvalOptions {
                extensions: true,
                ..Default::default()
            },
        )?;
        let prepared = engine.prepare_external_markers(request.clone(), EvalOptions::default())?;
        inputs.push(serde_json::to_value(prepared.readouts())?);
        cases.push(GoldenCase {
            id: format!("synthetic-{index}"),
            request,
            expected: response
                .answers
                .iter()
                .map(|(id, answer)| (id.clone(), probabilities(answer)))
                .collect(),
            // Fixed synthetic labels solely exercise the serving gate. They
            // are not observations and cannot qualify a released checkpoint.
            targets: BTreeMap::from([
                ("a_noul".into(), "yes".into()),
                ("m_score".into(), "1".into()),
                ("z_choice".into(), "returns".into()),
            ]),
        });
        responses.push(response);
    }
    let suite = GoldenSuite {
        schema_version: "1.0".into(),
        family: "F1".into(),
        hash: None,
        cases,
    };
    std::fs::write(path.join("golden.json"), serde_json::to_vec_pretty(&suite)?)?;
    std::fs::write(
        path.join("reference.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"qualified":false,"responses":responses,"readouts":inputs,"reference":"independent native synthetic arithmetic"}),
        )?,
    )?;
    Ok(())
}
