//! Metadata planning must not copy token/marker payloads; no runtime speed claim.
#![cfg(feature = "external-scores")]
#[path = "support/allocations.rs"]
mod allocations;
use allocations::measure;
use huncho_core::{
    backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput},
    contract::SystemOneRequest,
    engine::{Engine, EvalOptions, MarkerBatchProfile},
    error::{Error, Result},
    manifest::{BackendId, Family, ModelManifest},
    tokenizer::SimpleTokenizer,
};

struct ShapeFixture;
impl Backend for ShapeFixture {
    fn id(&self) -> BackendId {
        BackendId::Onnx
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: BackendId::Onnx,
            dtype: "fp32".into(),
            families: vec![Family::F1],
            max_context: 8192,
            ..Default::default()
        }
    }
    fn forward(&mut self, _: ForwardInput) -> Result<ForwardOutput> {
        panic!("dimension planning must never submit a forward")
    }
    fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported("explicit shape fixture".into()))
    }
}

#[test]
fn external_planner_heap_requests_do_not_grow_with_token_payloads() {
    let manifest: ModelManifest = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/mock-model/huncho-model.json"
    )))
    .unwrap();
    let engine = Engine::new(
        manifest,
        Box::new(ShapeFixture),
        Box::new(SimpleTokenizer::new(4096)),
        Default::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();
    let profile = MarkerBatchProfile {
        max_batch_tokens: 65536,
        max_batch_padding_percent: 0,
    };
    let mut previous = None;
    for words in [1, 800] {
        let request: SystemOneRequest = serde_json::from_value(serde_json::json!({
            "model":"mock-laya", "state":"ticket ".repeat(words),
            "questions":{
                "q1":{"type":"choice","instructions":"Team?","criteria":{"billing":"Charge","returns":"Refund"}},
                "q2":{"type":"choice","instructions":"Team?","criteria":{"billing":"Charge","returns":"Refund"}},
                "q3":{"type":"choice","instructions":"Team?","criteria":{"billing":"Charge","returns":"Refund"}},
                "q4":{"type":"choice","instructions":"Team?","criteria":{"billing":"Charge","returns":"Refund"}}
            }
        })).unwrap();
        let plan = engine
            .prepare_external_markers(request, EvalOptions::default())
            .unwrap();
        let length = plan.readouts()[0].tokens.len();
        // Measure just the historical clone stage, retaining every input so
        // the compiler cannot discard unused payload copies. Grouping would
        // allocate additional metadata; this is a conservative comparison.
        let (historical, historical_calls) = measure(|| {
            plan.readouts()
                .iter()
                .enumerate()
                .map(|(i, input)| {
                    (
                        i,
                        ForwardInput::new(input.tokens.clone(), input.positions.clone()),
                    )
                })
                .collect::<Vec<_>>()
        });
        let (groups, planning_calls) = measure(|| plan.marker_batches(profile).unwrap());
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].readouts, [0, 1, 2, 3]);
        assert_eq!(groups[0].sequence, length);
        assert!(
            historical_calls.1
                >= plan
                    .readouts()
                    .iter()
                    .map(|r| r.tokens.len() * 4)
                    .sum::<usize>()
        );
        if let Some((short_length, short_calls, short_historical)) = previous {
            assert!(length > short_length);
            assert_eq!(planning_calls, short_calls);
            assert!(historical_calls.1 > short_historical);
            assert!(historical_calls.1 > planning_calls.1 * 4);
        }
        previous = Some((length, planning_calls, historical_calls.1));
        assert_eq!(
            historical[0].1.tokens.as_slice(),
            plan.readouts()[0].tokens.as_slice()
        );
        println!("tokens_per_question={length}, questions=4: historical clone-stage calls={}, requested_bytes={}; actual borrowed planner calls={}, requested_bytes={}",
            historical_calls.0, historical_calls.1, planning_calls.0, planning_calls.1);
    }
}
