//! The model engine: ties the manifest, backend, tokenizer, head, prompt
//! builder, and calibration layer into the evaluation pipeline.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::backend::{Backend, ForwardInput};
use crate::calibration::{self, bucket_size, confidence};
use crate::contract::{Answer, Question, SystemOneRequest, SystemOneResponse, Usage};
use crate::error::{Error, Result};
use crate::head::{self, HeadParams};
use crate::manifest::{BackendId, CalibrationEntry, Family, ModelManifest};
use crate::prompt::{Candidate, CandidateKind, PromptFormatter, formatter_for};
#[cfg(test)]
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

/// Options controlling a single evaluation.
#[derive(Debug, Clone, Default)]
pub struct EvalOptions {
    /// Include engine-specific extras in the response (API-05). Off by default.
    pub extensions: bool,
    /// Override the model's max context (for testing). `None` uses manifest.
    pub max_context: Option<usize>,
}

/// A loaded, serving-ready model.
pub struct Engine {
    manifest: ModelManifest,
    /// The backend. Wrapped in a [`Mutex`] so a shared [`Engine`] can drive it
    /// through `&self`; this is the synchronization point for the (v1) in-process
    /// scheduler, which serializes forwards per model.
    backend: Mutex<Box<dyn Backend>>,
    tokenizer: Box<dyn Tokenizer>,
    formatter: Box<dyn PromptFormatter>,
    head: HeadParams,
    backend_id: BackendId,
    dtype: String,
    /// Resolved calibration entry for this backend+dtype.
    calibration: CalibrationEntry,
}

impl Engine {
    /// Build an engine from its parts.
    pub fn new(
        manifest: ModelManifest,
        backend: Box<dyn Backend>,
        tokenizer: Box<dyn Tokenizer>,
        head: HeadParams,
        backend_id: BackendId,
        dtype: impl Into<String>,
    ) -> Result<Engine> {
        let dtype = dtype.into();
        let calibration = manifest.calibration.resolve(&backend_id.to_string(), &dtype);
        Ok(Engine {
            formatter: formatter_for(&manifest),
            manifest,
            backend: Mutex::new(backend),
            tokenizer,
            head,
            backend_id,
            dtype,
            calibration,
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn family(&self) -> Family {
        self.manifest.family
    }

    pub fn backend_id(&self) -> BackendId {
        self.backend_id
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn calibration(&self) -> &CalibrationEntry {
        &self.calibration
    }

    /// Evaluate a request and produce a response.
    pub fn eval(&self, req: &SystemOneRequest, opts: &EvalOptions) -> Result<SystemOneResponse> {
        req.validate()?;

        if self.family() == Family::F5 {
            return self.eval_joint(req, opts);
        }

        let max_context = opts.max_context.unwrap_or(self.manifest.backbone.max_context);
        let max_options = self.manifest.prompt_contract.max_options;

        let mut answers: BTreeMap<String, Answer> = BTreeMap::new();
        let mut raw_logits: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut total_tokens = 0u64;

        // Iterate questions deterministically.
        for (id, question) in &req.questions {
            let prompt = self.formatter.build(&req.state, question, self.tokenizer.as_ref())?;

            // CORE-08: enforce model input budgets; never silently truncate.
            let n_tokens = prompt.token_len();
            if n_tokens > max_context {
                return Err(Error::Request(format!(
                    "question `{id}` requires {n_tokens} tokens but the model context is {max_context}"
                )));
            }
            let n_options = prompt.candidates.len();
            if n_options > max_options {
                return Err(Error::Request(format!(
                    "question `{id}` has {n_options} options; the model supports at most {max_options}"
                )));
            }
            total_tokens += n_tokens as u64;

            // Run the backend (v1 serializes forwards per model).
            let positions: Vec<usize> = prompt.candidates.iter().map(|c| c.position).collect();
            let input = ForwardInput::new(prompt.tokens.clone(), positions).with_qtype(prompt.qtype);
            let mut backend = self
                .backend
                .lock()
                .map_err(|_| Error::Backend("backend lock poisoned".into()))?;
            let output = backend.forward(input)?;
            drop(backend);

            // Head -> candidate logits.
            let logits = head::candidate_logits(
                self.manifest.family,
                self.manifest.head.kind,
                &output,
                &prompt.candidates,
                &self.head,
            )?;

            // Calibration: temperature + softmax.
            let temperature = self.temperature_for(question, n_options);
            let probabilities = calibration::calibrate(&logits, temperature)?;

            if opts.extensions {
                raw_logits.insert(id.clone(), logits.clone());
            }

            let answer = self.build_answer(question, &prompt.candidates, &probabilities)?;
            answers.insert(id.clone(), answer);
        }

        let mut response = SystemOneResponse::new(req.model.clone(), answers, Usage::new(total_tokens));
        if opts.extensions {
            response.extensions = Some(self.extensions(raw_logits));
        }
        Ok(response)
    }

    /// Joint-schema models own tokenization and score every question together.
    fn eval_joint(&self, req: &SystemOneRequest, opts: &EvalOptions) -> Result<SystemOneResponse> {
        let max_context = opts
            .max_context
            .unwrap_or(self.manifest.backbone.max_context)
            .min(self.manifest.backbone.max_context);
        let candidates: BTreeMap<_, _> = req
            .questions
            .iter()
            .map(|(id, question)| {
                let values = joint_candidates(question);
                if values.is_empty() || values.len() > self.manifest.prompt_contract.max_options {
                    return Err(Error::Request(format!(
                        "question `{id}` has an unsupported number of options"
                    )));
                }
                Ok((id.clone(), values))
            })
            .collect::<Result<_>>()?;
        let output = self
            .backend
            .lock()
            .map_err(|_| Error::Backend("backend lock poisoned".into()))?
            .forward_request(req, max_context)?;
        if output.input_tokens == 0 || output.input_tokens > max_context as u64 {
            return Err(Error::Backend(
                "joint backend returned invalid token usage".into(),
            ));
        }
        if output.logits.len() != req.questions.len() {
            return Err(Error::Backend(
                "joint backend returned the wrong number of questions".into(),
            ));
        }
        let mut answers = BTreeMap::new();
        let mut raw_logits = BTreeMap::new();
        for (id, question) in &req.questions {
            let options = &candidates[id];
            let scores = output
                .logits
                .get(id)
                .ok_or_else(|| Error::Backend(format!("joint backend omitted question `{id}`")))?;
            if scores.len() != options.len() {
                return Err(Error::Backend(format!(
                    "joint backend returned the wrong options for `{id}`"
                )));
            }
            let logits = options
                .iter()
                .map(|c| {
                    let label = match (question, c.label.as_str()) {
                        (Question::Noul { .. }, "yes") => "true",
                        (Question::Noul { .. }, "no") => "false",
                        _ => &c.label,
                    };
                    let value = scores.get(label).copied().ok_or_else(|| {
                        Error::Backend(format!("joint backend omitted option `{label}` for `{id}`"))
                    })?;
                    if !value.is_finite() {
                        return Err(Error::Backend(format!(
                            "joint backend returned non-finite logits for `{id}`"
                        )));
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>>>()?;
            let probabilities =
                calibration::calibrate(&logits, self.temperature_for(question, options.len()))?;
            answers.insert(
                id.clone(),
                self.build_answer(question, options, &probabilities)?,
            );
            if opts.extensions {
                raw_logits.insert(id.clone(), logits);
            }
        }
        let mut response =
            SystemOneResponse::new(req.model.clone(), answers, Usage::new(output.input_tokens));
        if opts.extensions {
            response.extensions = Some(self.extensions(raw_logits));
        }
        Ok(response)
    }

    /// Resolve the temperature for a question (Laya uses per-type base temps plus
    /// per `{type}:{bucket}` overrides; F4 uses per-type temps; others use the default).
    fn temperature_for(&self, question: &Question, n_options: usize) -> f32 {
        let type_name = question.type_name();
        // Laya-style per-option-count override, keyed `{type}:{bucket}`.
        if let Some(tbo) = &self.calibration.temperature_by_options {
            let bucket = format!("{}:{}", type_name, bucket_size(n_options));
            if let Some(t) = tbo.get(&bucket) {
                return *t;
            }
        }
        if let Some(per_type) = &self.calibration.per_type_temperatures {
            if let Some(t) = per_type.get(type_name) {
                return *t;
            }
        }
        self.calibration.temperature
    }

    fn build_answer(
        &self,
        question: &Question,
        candidates: &[Candidate],
        probabilities: &[f32],
    ) -> Result<Answer> {
        if candidates.len() != probabilities.len() {
            return Err(Error::Calibration(format!(
                "candidate/probability count mismatch: {} vs {}",
                candidates.len(),
                probabilities.len()
            )));
        }
        let conf = if self.manifest.prompt_contract.template == "kev-v1"
            && matches!(question, Question::Score { .. })
        {
            calibration::confidence_score(probabilities)
        } else {
            confidence(probabilities, &self.calibration.confidence)
        };
        match question {
            Question::Choice { .. } => {
                let mut best = 0usize;
                for (i, &p) in probabilities.iter().enumerate() {
                    if p > probabilities[best] {
                        best = i;
                    }
                }
                let mut probs_map = BTreeMap::new();
                for c in candidates {
                    probs_map.insert(c.label.clone(), probabilities[c.index]);
                }
                Ok(Answer::Choice {
                    choice: candidates[best].label.clone(),
                    probabilities: probs_map,
                    confidence: conf,
                })
            }
            Question::Score { .. } => {
                let mut probs_map = BTreeMap::new();
                let mut legend = BTreeMap::new();
                let mut score = 0.0f32;
                for c in candidates {
                    let key = c.label.clone(); // level index as string
                    probs_map.insert(key.clone(), probabilities[c.index]);
                    legend.insert(
                        key,
                        c.description.clone().unwrap_or_else(|| "".into()),
                    );
                    score += c.index as f32 * probabilities[c.index];
                }
                Ok(Answer::Score {
                    probabilities: probs_map,
                    score,
                    legend,
                    confidence: conf,
                })
            }
            Question::Noul { .. } => {
                // Candidates are [yes, no]. The answer is the probability of yes.
                let p_yes = candidates
                    .iter()
                    .position(|c| c.label == "yes")
                    .map(|i| probabilities[i])
                    .unwrap_or(0.0);
                Ok(Answer::Noul { noul: p_yes })
            }
        }
    }

    fn extensions(&self, raw_logits: BTreeMap<String, Vec<f32>>) -> crate::contract::Extensions {
        crate::contract::Extensions {
            backend: Some(self.backend_id.to_string()),
            dtype: Some(self.dtype.clone()),
            calibration_status: Some(format!("{:?}", self.calibration.status).to_lowercase()),
            confidence_definition: Some(self.calibration.confidence.to_string()),
            prompt_contract_hash: Some(self.manifest.prompt_contract.contract_hash.clone()),
            raw_logits: if raw_logits.is_empty() {
                None
            } else {
                Some(raw_logits)
            },
        }
    }
}

fn joint_candidates(question: &Question) -> Vec<Candidate> {
    let labels: Vec<(String, Option<String>, CandidateKind)> = match question {
        Question::Choice { criteria, .. } => criteria
            .keys()
            .map(|label| (label.clone(), None, CandidateKind::Option))
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(index, level)| {
                (
                    index.to_string(),
                    Some(
                        level
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| level.to_string()),
                    ),
                    CandidateKind::Level,
                )
            })
            .collect(),
        Question::Noul { .. } => vec![
            ("yes".into(), None, CandidateKind::YesNo),
            ("no".into(), None, CandidateKind::YesNo),
        ],
    };
    labels
        .into_iter()
        .enumerate()
        .map(|(index, (label, description, kind))| Candidate {
            kind,
            position: 0,
            code_id: 0,
            label,
            description,
            index,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, CacheHandle, Capabilities, ForwardOutput};
    use crate::manifest::{
        Backbone, BackboneSource, CalibrationConfig, CalibrationEntry, CalibrationStatus,
        ConfidenceDef, HeadConfig, HeadKind, PromptContract, Reference,
    };
    use crate::tokenizer::SimpleTokenizer;

    /// A deterministic backend that returns hidden states (Features) so the
    /// feature-projection head (F1) can be exercised without real weights.
    struct TestBackend;

    impl Backend for TestBackend {
        fn id(&self) -> BackendId { BackendId::Onnx }
        fn capabilities(&self) -> Capabilities {
            let mut c = Capabilities::default();
            c.id = BackendId::Onnx;
            c.dtype = "fp32".into();
            c.max_context = 4096;
            c.families = vec![Family::F1];
            c
        }
        fn forward(&mut self, input: ForwardInput) -> Result<ForwardOutput> {
            // Produce a sparse deterministic hidden vector per position derived
            // from the token id at that position.
            let n = input.positions.len();
            let hidden = 64usize;
            let mut data = vec![0.0f32; n * hidden];
            for (row, &pos) in input.positions.iter().enumerate() {
                let tok = input.tokens.get(pos).copied().unwrap_or(0);
                let col = (tok as usize) % hidden;
                data[row * hidden + col] = tok as f32 + 0.5;
            }
            let values = Tensor::new(vec![n, hidden], data).unwrap();
            Ok(ForwardOutput::Features { positions: input.positions, values })
        }
        fn fork(&mut self, handle: CacheHandle) -> Result<CacheHandle> { Ok(handle) }
    }

    fn manifest() -> ModelManifest {
        let m = ModelManifest {
            schema_version: crate::manifest::MANIFEST_SCHEMA_VERSION.into(),
            name: "test".into(),
            family: Family::F1,
            backbone: Backbone {
                source: BackboneSource::Hf { repo: "x".into(), revision: "y".into() },
                artifacts: Default::default(),
                hidden_size: 1024,
                max_context: 4096,
                tokenizer: None,
            },
            adapter: None,
            f3: None,
            head: HeadConfig { kind: HeadKind::OptionMarker, weights: "head.safetensors".into(), width: 1, pointer_offset: None },
            prompt_contract: PromptContract {
                template: "laya-v1".into(),
                option_marker_tokens: vec!["<option:0>".into()],
                state_budget: 3072,
                head_budget: 512,
                max_options: 255,
                contract_hash: "abc".into(),
                max_len: 512,
                head_max_len: 192,
            },
            calibration: CalibrationConfig {
                default: CalibrationEntry { temperature: 1.0, per_type_temperatures: None, temperature_by_options: None, confidence: ConfidenceDef::Peak, status: CalibrationStatus::Fit },
                entries: Default::default(),
                eval_set_hash: None,
            },
            reference: Some(Reference { family_impl: "x".into(), revision: "y".into(), golden: "g.json".into() }),
            capabilities: Default::default(),
        };
        m
    }

    fn engine() -> Engine {
        let tk: Box<dyn Tokenizer> = Box::new(SimpleTokenizer::new(32768));
        Engine::new(manifest(), Box::new(TestBackend), tk, HeadParams::default(), BackendId::Onnx, "fp32").unwrap()
    }

    fn req() -> SystemOneRequest {
        let json = serde_json::json!({
            "state": "The customer wants a refund because the shoes are too small.",
            "model": "test",
            "questions": {
                "department": { "type": "choice", "instructions": "Which team?", "criteria": { "returns": "money back", "billing": "charge issue" } },
                "is_refund": { "type": "noul", "instructions": "The customer is requesting a refund" },
                "severity": { "type": "score", "instructions": "severity?", "criteria": ["low","mid","high"] }
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn eval_returns_typed_answers() {
        let e = engine();
        let resp = e.eval(&req(), &EvalOptions::default()).unwrap();
        assert_eq!(resp.answers.len(), 3);
        assert!(matches!(resp.answers["department"], Answer::Choice { .. }));
        assert!(matches!(resp.answers["is_refund"], Answer::Noul { .. }));
        assert!(matches!(resp.answers["severity"], Answer::Score { .. }));
        assert!(resp.extensions.is_none());
    }

    #[test]
    fn extensions_off_by_default() {
        let e = engine();
        let resp = e.eval(&req(), &EvalOptions::default()).unwrap();
        assert!(resp.extensions.is_none());
        let resp2 = e.eval(&req(), &EvalOptions { extensions: true, ..Default::default() }).unwrap();
        assert!(resp2.extensions.is_some());
        let ext = resp2.extensions.unwrap();
        assert_eq!(ext.backend.as_deref(), Some("onnx"));
        assert!(ext.raw_logits.is_some());
    }

    #[test]
    fn score_distribution_sums_to_one() {
        let e = engine();
        let resp = e.eval(&req(), &EvalOptions::default()).unwrap();
        if let Answer::Score { probabilities, .. } = &resp.answers["severity"] {
            let s: f32 = probabilities.values().sum();
            assert!((s - 1.0).abs() < 1e-4, "sum={s}");
        } else {
            panic!("expected score answer");
        }
    }

    #[test]
    fn rejects_oversized_input() {
        let e = engine();
        let r = req();
        // Force a context that is too small.
        let resp = e.eval(&r, &EvalOptions { max_context: Some(4), ..Default::default() });
        assert!(resp.is_err());
    }
}
