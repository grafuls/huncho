//! Actual CPU mask-sensitive ONNX execution. Synthetic vectors never release a model.
#![cfg(feature = "onnx")]
use huncho_backend::{onnx::OnnxOptions, OnnxBackend};
use huncho_core::{
    backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput},
    conformance::{
        run_suite_with_cross_request_batches, run_suite_with_options, GoldenCase, GoldenSuite,
    },
    contract::{Answer, SystemOneRequest},
    engine::{Engine, EvalOptions, EvalStats},
    manifest::{BackendId, ModelManifest},
    tensor::Tensor,
    tokenizer::SimpleTokenizer,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
fn load(budget: usize) -> OnnxBackend {
    OnnxBackend::load_with_options(
        root().join("tiny_encoder_masked.onnx"),
        8,
        128,
        "fp32",
        OnnxOptions {
            native_batch: true,
            intra_threads: 1,
            output_buffer_bytes: budget,
            ..Default::default()
        },
    )
    .unwrap()
}
fn embedding(t: u32, j: usize) -> f32 {
    (((t as usize * 7 + j * 3) % 19) as f32 - 9.) / 8.
}
fn independent(input: &ForwardInput) -> Tensor {
    let context: Vec<f32> = (0..8)
        .map(|j| {
            input.tokens.iter().map(|&t| embedding(t, j)).sum::<f32>() / input.tokens.len() as f32
        })
        .collect();
    Tensor::new(
        vec![input.positions.len(), 8],
        input
            .positions
            .iter()
            .flat_map(|&p| {
                let context = &context;
                (0..8).map(move |j| embedding(input.tokens[p], j) + context[j] + p as f32 * 0.03125)
            })
            .collect(),
    )
    .unwrap()
}
fn parity(actual: &ForwardOutput, expected: &Tensor) {
    assert_eq!(actual.values().shape(), expected.shape());
    assert!(actual
        .values()
        .data()
        .iter()
        .zip(expected.data())
        .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-5));
}

#[test]
fn actual_masked_rectangles_match_independent_scores_and_preserve_owned_repeated_readouts() {
    let rows: Vec<_> = [1, 3, 17, 47, 128]
        .into_iter()
        .enumerate()
        .map(|(row, len)| {
            let mut tokens: Vec<_> = (0..len).map(|p| ((p * 7 + row * 3) % 16) as u32).collect();
            tokens[0] = 0; // Zero is a real token, never a mask sentinel.
            ForwardInput::new(tokens, vec![len - 1, 0, len / 2, 0]).with_qtype((row % 3) as u32)
        })
        .collect();
    for budget in [0, 5 * 128 * 8 * 4] {
        let mut backend = load(budget);
        assert!(backend.supports_padded_batch());
        assert_eq!(
            backend.capabilities().extra["padded_batch_execution"],
            "onnx-cpu-right-mask-v1"
        );
        let expected: Vec<_> = rows.iter().map(independent).collect();
        let first = backend.forward_padded_batch(rows.clone()).unwrap();
        for _ in 0..2 {
            let outputs = backend.forward_padded_batch(rows.clone()).unwrap();
            for (index, output) in outputs.iter().enumerate() {
                assert_eq!(output.positions(), rows[index].positions);
                parity(output, &expected[index]);
                parity(&first[index], &expected[index]);
            }
        }
        assert_eq!(backend.retained_output_bytes(), budget);
        assert_eq!(
            backend.output_buffer_reuses(),
            if budget > 0 { 2 } else { 0 }
        );
        let mut wrong = rows[1].clone();
        wrong.tokens.resize(128, 0);
        assert!(
            backend
                .forward(wrong)
                .unwrap()
                .values()
                .data()
                .iter()
                .zip(expected[1].data())
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "fixture must expose an all-ones padding mask"
        );
        let equal = vec![rows[2].clone(); 2];
        let padded = backend.forward_padded_batch(equal.clone()).unwrap();
        let native = backend.forward_batch(equal).unwrap();
        assert!(padded
            .iter()
            .zip(native)
            .all(|(a, b)| a.values().data() == b.values().data()));
        let mut empty_readout = rows[1].clone();
        empty_readout.positions.clear();
        assert_eq!(
            backend
                .forward_padded_batch(vec![rows[0].clone(), empty_readout])
                .unwrap()[1]
                .values()
                .shape(),
            &[0, 8]
        );
        for invalid in [
            vec![],
            vec![rows[0].clone(); 65],
            vec![ForwardInput::new(vec![], vec![])],
            vec![ForwardInput::new(vec![1; 129], vec![0])],
            vec![ForwardInput::new(vec![1], vec![1])],
        ] {
            assert!(backend.forward_padded_batch(invalid).is_err());
        }
        for kind in 0..3 {
            let mut bad = rows[0].clone();
            match kind {
                0 => bad.retain_cache = true,
                1 => bad.fork_from = Some(CacheHandle { id: 1 }),
                _ => bad.logit_codes = Some(vec![0]),
            }
            assert!(backend.forward_padded_batch(vec![bad]).is_err());
        }
        let repeated = backend.forward_padded_batch(rows.clone()).unwrap();
        repeated
            .iter()
            .zip(&expected)
            .for_each(|(a, b)| parity(a, b));
    }
    for file in ["tiny_encoder.onnx", "tiny_encoder_batch_nomask.onnx"] {
        let mut backend = OnnxBackend::load_with_options(
            root().join(file),
            8,
            128,
            "fp32",
            OnnxOptions {
                native_batch: file.contains("batch"),
                intra_threads: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!backend.supports_padded_batch());
        assert!(backend.forward_padded_batch(rows.clone()).is_err());
    }
}

struct Independent(Capabilities);
impl Backend for Independent {
    fn id(&self) -> BackendId {
        BackendId::Onnx
    }
    fn capabilities(&self) -> Capabilities {
        self.0.clone()
    }
    fn forward(&mut self, input: ForwardInput) -> huncho_core::Result<ForwardOutput> {
        let values = independent(&input);
        Ok(ForwardOutput::Features {
            positions: input.positions,
            values,
        })
    }
    fn fork(&mut self, _: CacheHandle) -> huncho_core::Result<CacheHandle> {
        unreachable!()
    }
}
fn probabilities(a: &Answer) -> BTreeMap<String, f32> {
    match a {
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
            probabilities.clone()
        }
        Answer::Noul { noul } => BTreeMap::from([("no".into(), 1. - noul), ("yes".into(), *noul)]),
    }
}
#[test]
fn whole_engine_native_padding_counts_actual_work_and_passes_fixed_synthetic_labeled_gates() {
    let mut manifest = ModelManifest::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/mock-model/huncho-model.json"),
    )
    .unwrap();
    manifest.backbone.hidden_size = 8;
    manifest.backbone.max_context = 128;
    manifest.prompt_contract.state_budget = 64;
    manifest.prompt_contract.head_budget = 64;
    manifest.prompt_contract.max_len = 128;
    manifest.prompt_contract.head_max_len = 64;
    let backend = load(1024 * 1024);
    let reference = Engine::new(
        manifest.clone(),
        Box::new(Independent(backend.capabilities())),
        Box::new(SimpleTokenizer::new(16)),
        Default::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();
    let engine = Engine::new(
        manifest,
        Box::new(backend),
        Box::new(SimpleTokenizer::new(16)),
        Default::default(),
        BackendId::Onnx,
        "fp32",
    )
    .unwrap();
    let mut cases = Vec::new();
    for (index, state) in ["refund", "a customer needs a refund"]
        .into_iter()
        .enumerate()
    {
        let request:SystemOneRequest=serde_json::from_value(serde_json::json!({"model":"mock-laya","state":state,"questions":{
            "team":{"type":"choice","instructions":"Team?","criteria":{"shipping":null,"billing":"Charges","returns":"Refunds"}},
            "urgent":{"type":"noul","instructions":"Urgent?"},
            "priority":{"type":"score","instructions":"Priority?","criteria":["low","medium","high","very high"]}
        }})).unwrap();
        let response = reference.eval(&request, &Default::default()).unwrap();
        cases.push(GoldenCase {
            id: index.to_string(),
            request,
            expected: response
                .answers
                .iter()
                .map(|(id, a)| (id.clone(), probabilities(a)))
                .collect(),
            targets: BTreeMap::from([
                ("team".into(), "shipping".into()),
                ("urgent".into(), "yes".into()),
                ("priority".into(), "2".into()),
            ]),
        });
    }
    let suite = GoldenSuite {
        schema_version: "1.0".into(),
        family: "F1".into(),
        hash: None,
        cases,
    };
    let options = EvalOptions {
        max_batch_tokens: Some(768),
        max_batch_padding_percent: 100,
        prepare_all: true,
        ..Default::default()
    };
    let reports = [
        run_suite_with_options(&engine, &suite, &Default::default(), &options).unwrap(),
        run_suite_with_cross_request_batches(&engine, &suite, &Default::default(), &options, 2)
            .unwrap(),
    ];
    let logical_tokens: u64 = suite
        .cases
        .iter()
        .map(|case| {
            reference
                .eval(&case.request, &Default::default())
                .unwrap()
                .usage
                .input_tokens
        })
        .sum();
    for report in &reports {
        assert!(report.passed, "{report:?}");
        assert!(report.max_prob_delta <= 1e-5);
        assert_eq!(report.outcome_calibration.as_ref().unwrap().questions, 6);
        assert!(report.optimization_parity.as_ref().unwrap().max_prob_delta <= 1e-4);
        assert!(report.work.padded_batch_calls > 0 && report.work.padded_tokens > 0);
        assert_eq!(
            report.work.processed_tokens,
            logical_tokens + report.work.padded_tokens
        );
        assert_eq!(report.work.result_cache_hits, 0);
    }
    assert!(reports[1].work.cross_request_batches > 0);
    assert!(reports[1].work.forward_calls < reports[0].work.forward_calls);
    let mut stats = EvalStats::default();
    let result = engine
        .eval_with_stats(&suite.cases[0].request, &options, &mut stats)
        .unwrap();
    assert_eq!(
        stats.processed_tokens,
        result.usage.input_tokens + stats.padded_tokens
    );
    assert_eq!(result.usage.output_tokens, 0);
    let mut incomplete = suite.clone();
    incomplete.cases[0].targets.remove("urgent");
    assert!(run_suite_with_options(&engine, &incomplete, &Default::default(), &options).is_err());
    let mut drifted = suite;
    drifted.cases[0].expected.insert(
        "urgent".into(),
        BTreeMap::from([("yes".into(), 0.999), ("no".into(), 0.001)]),
    );
    assert!(
        !run_suite_with_options(&engine, &drifted, &Default::default(), &options)
            .unwrap()
            .passed
    );
}

#[cfg(feature = "onnx-shared")]
#[test]
fn masked_native_shared_contexts_are_concurrently_independent() {
    let backend = OnnxBackend::load_with_options(
        root().join("tiny_encoder_masked.onnx"),
        8,
        128,
        "fp32",
        OnnxOptions {
            native_batch: true,
            shared_initializers: true,
            intra_threads: 1,
            output_buffer_bytes: 4096,
            ..Default::default()
        },
    )
    .unwrap();
    let replicas = (0..3)
        .map(|_| backend.replica().unwrap())
        .collect::<Vec<_>>();
    drop(backend);
    let threads = replicas
        .into_iter()
        .map(|mut backend| {
            std::thread::spawn(move || {
                for i in 0..5 {
                    let rows = vec![
                        ForwardInput::new(vec![0, 1, 2], vec![2, 0, 2]),
                        ForwardInput::new(vec![i, 0], vec![1, 0]),
                    ];
                    let outputs = backend.forward_padded_batch(rows.clone()).unwrap();
                    outputs
                        .iter()
                        .zip(&rows)
                        .for_each(|(a, b)| parity(a, &independent(b)));
                }
            })
        })
        .collect::<Vec<_>>();
    threads.into_iter().for_each(|t| t.join().unwrap());
}
