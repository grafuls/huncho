//! Browser transport for the actual shared Rust prompt/calibration pipeline.
use huncho_core::backend::{Backend, CacheHandle, Capabilities, ForwardInput, ForwardOutput};
use huncho_core::conformance::{
    run_external_marker_batched_suite, run_external_marker_suite, GoldenSuite,
};
use huncho_core::contract::{SystemOneRequest, SystemOneResponse};
use huncho_core::engine::{Engine, EvalOptions, EvalStats, ExternalEvaluation, MarkerBatchProfile};
use huncho_core::error::{Error, Result};
use huncho_core::head::HeadParams;
use huncho_core::manifest::{BackendId, CalibrationStatus, Family, HeadKind, ModelManifest};
use huncho_core::tokenizer::HfTokenizer;
use serde::Serialize;
use std::collections::BTreeMap;
use wasm_bindgen::prelude::*;

fn js(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
fn serialize(value: &impl Serialize) -> std::result::Result<String, JsValue> {
    serde_json::to_string(value).map_err(js)
}

// No placeholder forward: JS must actually execute the integrated head graph
// and supply its raw marker scores through the explicit external-score ABI.
struct ExternalGraph(Capabilities);
impl Backend for ExternalGraph {
    fn id(&self) -> BackendId {
        BackendId::Onnx
    }
    fn capabilities(&self) -> Capabilities {
        self.0.clone()
    }
    fn forward(&mut self, _: ForwardInput) -> Result<ForwardOutput> {
        Err(Error::Unsupported(
            "use the external marker-score ABI with actual browser graph execution".into(),
        ))
    }
    fn fork(&mut self, _: CacheHandle) -> Result<CacheHandle> {
        Err(Error::Unsupported(
            "browser prefixes are not implemented".into(),
        ))
    }
}

#[wasm_bindgen]
pub struct MarkerEngine {
    engine: Engine,
    plans: BTreeMap<u32, (ExternalEvaluation, bool)>,
    next_plan: u32,
    batch: Option<MarkerBatchProfile>,
}

#[wasm_bindgen]
impl MarkerEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(
        manifest_json: &str,
        tokenizer_json: &str,
        identity_json: &str,
    ) -> std::result::Result<MarkerEngine, JsValue> {
        if manifest_json.len() > 1024 * 1024
            || tokenizer_json.len() > 16 * 1024 * 1024
            || identity_json.len() > 64 * 1024
        {
            return Err(js("browser metadata/tokenizer exceeds memory limits"));
        }
        let manifest: ModelManifest = serde_json::from_str(manifest_json).map_err(js)?;
        manifest.validate().map_err(js)?;
        if manifest.family != Family::F1
            || manifest.head.kind != HeadKind::OptionMarker
            || manifest.head.width != 1
            || manifest.backbone.max_context == 0
            || manifest.backbone.max_context > 4096
            || manifest.prompt_contract.max_options > 255
            || manifest.backbone.tokenizer.is_none()
            || manifest.find_artifact(BackendId::Onnx, "fp32").is_none()
            || !manifest
                .calibration
                .entries
                .get("onnx:fp32")
                .is_some_and(|entry| entry.status != CalibrationStatus::Pending)
        {
            return Err(js("browser runtime requires a small fitted F1 FP32 integrated option-head package and tokenizer"));
        }
        let mut metadata: BTreeMap<String, String> =
            serde_json::from_str(identity_json).map_err(js)?;
        let provider = metadata.get("onnx_execution_provider").map(String::as_str);
        if provider != Some("wasm")
            || metadata.get("onnx_web_version").map(String::as_str) != Some("1.30.0")
        {
            return Err(js("unsupported browser execution identity"));
        }
        metadata.insert("device".into(), "WASM".into());
        let batch = if metadata.get("external_marker_scores").map(String::as_str)
            == Some("graph-integrated-f1-batch-v1")
        {
            let number = |key: &str| -> std::result::Result<usize, JsValue> {
                metadata
                    .get(key)
                    .ok_or_else(|| js("missing browser batch profile"))?
                    .parse()
                    .map_err(js)
            };
            let profile = MarkerBatchProfile {
                max_batch_tokens: number("browser_batch_tokens")?,
                max_batch_padding_percent: number("browser_batch_padding_percent")?,
            };
            profile.validate().map_err(js)?;
            Some(profile)
        } else {
            metadata.insert(
                "external_marker_scores".into(),
                "graph-integrated-f1-v1".into(),
            );
            None
        };
        let caps = Capabilities {
            id: BackendId::Onnx,
            dtype: "fp32".into(),
            families: vec![Family::F1],
            max_context: manifest.backbone.max_context,
            extra: metadata,
            ..Default::default()
        };
        let tokenizer = HfTokenizer::from_json(tokenizer_json.as_bytes()).map_err(js)?;
        let engine = Engine::new(
            manifest,
            Box::new(ExternalGraph(caps)),
            Box::new(tokenizer),
            HeadParams::default(),
            BackendId::Onnx,
            "fp32",
        )
        .map_err(js)?;
        Ok(Self {
            engine,
            plans: BTreeMap::new(),
            next_plan: 1,
            batch,
        })
    }

    pub fn prepare(
        &mut self,
        request_json: &str,
        extensions: bool,
    ) -> std::result::Result<String, JsValue> {
        self.prepare_plan(request_json, extensions, self.batch.is_some())
    }

    pub fn prepare_independent(
        &mut self,
        request_json: &str,
        extensions: bool,
    ) -> std::result::Result<String, JsValue> {
        self.prepare_plan(request_json, extensions, false)
    }

    fn prepare_plan(
        &mut self,
        request_json: &str,
        extensions: bool,
        batched: bool,
    ) -> std::result::Result<String, JsValue> {
        if request_json.len() > 1024 * 1024 || self.plans.len() >= 8 {
            return Err(js("browser request/pending-plan budget exceeded"));
        }
        let value: serde_json::Value = serde_json::from_str(request_json).map_err(js)?;
        if value.get("images").is_some() || value.get("videos").is_some() {
            return Err(js("browser runtime accepts text/JSON state only"));
        }
        let request: SystemOneRequest = serde_json::from_value(value).map_err(js)?;
        if request.questions.len() > 64 {
            return Err(js("browser request supports at most 64 questions"));
        }
        let plan = self
            .engine
            .prepare_external_markers(
                request,
                EvalOptions {
                    extensions,
                    ..Default::default()
                },
            )
            .map_err(js)?;
        let handle = self.next_plan;
        self.next_plan = self
            .next_plan
            .checked_add(1)
            .ok_or_else(|| js("browser plan ids exhausted"))?;
        let groups = if batched {
            Some(plan.marker_batches(self.batch.unwrap()).map_err(js)?)
        } else {
            None
        };
        let json = serialize(
            &serde_json::json!({"handle":handle,"readouts":plan.readouts(),"groups":groups}),
        )?;
        self.plans.insert(handle, (plan, batched));
        Ok(json)
    }

    pub fn finish(
        &mut self,
        handle: u32,
        scores_json: &str,
    ) -> std::result::Result<String, JsValue> {
        let (plan, batched) = self
            .plans
            .remove(&handle)
            .ok_or_else(|| js("unknown or consumed browser plan"))?;
        if scores_json.len() > 1024 * 1024 {
            return Err(js("browser scores exceed memory limits"));
        }
        let scores: Vec<Vec<f32>> = serde_json::from_str(scores_json).map_err(js)?;
        let (response, work) = if batched {
            self.engine
                .finish_external_marker_batches(plan, self.batch.unwrap(), scores)
        } else {
            self.engine.finish_external_markers(plan, scores)
        }
        .map_err(js)?;
        serialize(&serde_json::json!({"response":response, "work":work}))
    }

    pub fn cancel(&mut self, handle: u32) -> bool {
        self.plans.remove(&handle).is_some()
    }

    pub fn requests_for_qualification(
        &self,
        golden_json: &str,
    ) -> std::result::Result<String, JsValue> {
        if golden_json.len() > 32 * 1024 * 1024 {
            return Err(js("browser golden suite exceeds memory budget"));
        }
        let suite: GoldenSuite = serde_json::from_str(golden_json).map_err(js)?;
        if suite.schema_version != "1.0"
            || suite.family != "F1"
            || suite.cases.is_empty()
            || suite.cases.iter().any(|case| {
                case.request.model != self.engine.manifest().name
                    || case.targets.is_empty()
                    || case.expected.len() != case.request.questions.len()
                    || case.targets.len() != case.expected.len()
                    || case.expected.iter().any(|(id, probabilities)| {
                        !case.request.questions.contains_key(id)
                            || !case
                                .targets
                                .get(id)
                                .is_some_and(|target| probabilities.contains_key(target))
                    })
            })
        {
            return Err(js("browser serving requires complete observed-label vectors for every requested question"));
        }
        serialize(
            &suite
                .cases
                .into_iter()
                .map(|case| case.request)
                .collect::<Vec<_>>(),
        )
    }

    pub fn qualify(
        &self,
        golden_json: &str,
        responses_json: &str,
        work_json: &str,
    ) -> std::result::Result<String, JsValue> {
        if self.batch.is_some() {
            return Err(js(
                "browser batches require fresh independent and tensor qualification",
            ));
        }
        if golden_json.len() > 32 * 1024 * 1024
            || responses_json.len() > 32 * 1024 * 1024
            || work_json.len() > 64 * 1024
        {
            return Err(js("browser qualification exceeds memory budgets"));
        }
        let suite: GoldenSuite = serde_json::from_str(golden_json).map_err(js)?;
        let responses: Vec<SystemOneResponse> = serde_json::from_str(responses_json).map_err(js)?;
        let work: EvalStats = serde_json::from_str(work_json).map_err(js)?;
        serialize(&run_external_marker_suite(&self.engine, &suite, &responses, work).map_err(js)?)
    }

    pub fn qualify_batched(
        &self,
        golden_json: &str,
        responses_json: &str,
        work_json: &str,
        independent_json: &str,
        independent_work_json: &str,
    ) -> std::result::Result<String, JsValue> {
        if golden_json.len() > 32 * 1024 * 1024
            || responses_json.len() > 32 * 1024 * 1024
            || independent_json.len() > 32 * 1024 * 1024
            || work_json.len() > 64 * 1024
            || independent_work_json.len() > 64 * 1024
        {
            return Err(js("browser batch qualification exceeds memory budgets"));
        }
        let profile = self
            .batch
            .ok_or_else(|| js("no immutable browser batch profile"))?;
        let suite: GoldenSuite = serde_json::from_str(golden_json).map_err(js)?;
        let responses: Vec<SystemOneResponse> = serde_json::from_str(responses_json).map_err(js)?;
        let work: EvalStats = serde_json::from_str(work_json).map_err(js)?;
        let independent: Vec<SystemOneResponse> =
            serde_json::from_str(independent_json).map_err(js)?;
        let independent_work: EvalStats =
            serde_json::from_str(independent_work_json).map_err(js)?;
        serialize(
            &run_external_marker_batched_suite(
                &self.engine,
                &suite,
                &responses,
                work,
                &independent,
                independent_work,
                profile,
            )
            .map_err(js)?,
        )
    }
}
