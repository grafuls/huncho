//! The Jev `/v1/systemone` wire contract.
//!
//! These types are intentionally byte-compatible with the public TypeSafe / Jev
//! HTTP contract so that the unmodified TypeSafe Python SDK works against `huncho`
//! with only a base-URL change.
//!
//! Reference: https://docs.typesafe.ai/api

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Maximum number of options for a Choice question (Jev contract).
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Maximum number of levels for a Score question (Jev contract).
pub const MAX_SCORE_LEVELS: usize = 10;
/// Minimum number of levels for a Score question (Jev contract).
pub const MIN_SCORE_LEVELS: usize = 2;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// The top-level request body for `POST /v1/systemone`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneRequest {
    /// The content to evaluate. A string, object, or array.
    pub state: StateValue,
    /// The model that handles the request.
    pub model: String,
    /// A map of question-id -> question. Answers come back under the same ids.
    pub questions: BTreeMap<String, Question>,
}

/// The `state` field: plain string, or structured JSON (object/array).
///
/// We accept any JSON value here and validate that it is a string, object, or
/// array in [`SystemOneRequest::validate`], matching the Jev contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StateValue(serde_json::Value);

impl StateValue {
    pub fn new(v: serde_json::Value) -> Self {
        StateValue(v)
    }

    pub fn as_value(&self) -> &serde_json::Value {
        &self.0
    }

    pub fn into_value(self) -> serde_json::Value {
        self.0
    }

    pub fn as_str(&self) -> Option<&str> {
        self.0.as_str()
    }
}

impl From<&str> for StateValue {
    fn from(s: &str) -> Self {
        StateValue(serde_json::Value::String(s.to_string()))
    }
}

impl From<String> for StateValue {
    fn from(s: String) -> Self {
        StateValue(serde_json::Value::String(s))
    }
}

impl From<serde_json::Value> for StateValue {
    fn from(v: serde_json::Value) -> Self {
        StateValue(v)
    }
}

/// The `instructions` field: string, object, or array.
///
/// The Jev contract forbids `null`, `true`/`false`, and numbers here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Instructions(serde_json::Value);

impl Instructions {
    pub fn as_value(&self) -> &serde_json::Value {
        &self.0
    }
}

impl From<serde_json::Value> for Instructions {
    fn from(v: serde_json::Value) -> Self {
        Instructions(v)
    }
}

/// A typed question object. The `type` field selects the variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Pick one option from a set.
    Choice {
        instructions: Instructions,
        /// Option label -> optional description. Insertion order is preserved so
        /// the typed option-marker head (e.g. Laya) sees options in the order the
        /// caller supplied, matching the trained model's expectation.
        #[serde(default)]
        criteria: indexmap::IndexMap<String, Option<serde_json::Value>>,
    },
    /// Rate the state along an ordered rubric.
    Score {
        instructions: Instructions,
        criteria: Vec<serde_json::Value>,
    },
    /// Yes/no. Returns the probability the answer is yes.
    Noul {
        instructions: Instructions,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },
}

/// Optional descriptions for a Noul question's yes and no meanings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(default)]
    #[serde(alias = "True")]
    pub yes: Option<serde_json::Value>,
    #[serde(default)]
    #[serde(alias = "False")]
    pub no: Option<serde_json::Value>,
}

impl Question {
    /// The question type name as it appears on the wire.
    pub fn type_name(&self) -> &'static str {
        match self {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        }
    }

    /// The typed-question embedding index used by the Laya decision head
    /// (`type_emb`: choice=0, score=1, noul=2).
    pub fn qtype_index(&self) -> u32 {
        match self {
            Question::Choice { .. } => 0,
            Question::Score { .. } => 1,
            Question::Noul { .. } => 2,
        }
    }

    pub fn instructions(&self) -> &Instructions {
        match self {
            Question::Choice { instructions, .. } => instructions,
            Question::Score { instructions, .. } => instructions,
            Question::Noul { instructions, .. } => instructions,
        }
    }
}

impl SystemOneRequest {
    /// Validate the request against the Jev contract.
    ///
    /// Enforces: state is string/object/array; instructions are never null,
    /// boolean, or numeric; Choice has <= 255 options; Score has between
    /// `MIN_SCORE_LEVELS` and `MAX_SCORE_LEVELS` levels; questions map is
    /// non-empty and ids are non-empty.
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(Error::Request("`model` must be a non-empty string".into()));
        }
        if self.questions.is_empty() {
            return Err(Error::Request("`questions` must contain at least one question".into()));
        }
        for (id, q) in &self.questions {
            if id.trim().is_empty() {
                return Err(Error::Request("question ids must be non-empty".into()));
            }
            validate_state(&self.state.0)?;
            validate_instruction(q.instructions().as_value())?;
            match q {
                Question::Choice { criteria, .. } => {
                    if criteria.len() > MAX_CHOICE_OPTIONS {
                        return Err(Error::Request(format!(
                            "Choice `{id}` has {} options; the Jev contract allows at most {MAX_CHOICE_OPTIONS}",
                            criteria.len()
                        )));
                    }
                }
                Question::Score { criteria, .. } => {
                    if criteria.len() < MIN_SCORE_LEVELS || criteria.len() > MAX_SCORE_LEVELS {
                        return Err(Error::Request(format!(
                            "Score `{id}` has {} levels; the Jev contract requires between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS}",
                            criteria.len()
                        )));
                    }
                }
                Question::Noul { .. } => {}
            }
        }
        Ok(())
    }
}

fn validate_state(v: &serde_json::Value) -> Result<()> {
    if v.is_string() || v.is_object() || v.is_array() {
        Ok(())
    } else {
        Err(Error::Request(
            "`state` must be a string, object, or array".into(),
        ))
    }
}

fn validate_instruction(v: &serde_json::Value) -> Result<()> {
    if v.is_string() || v.is_object() || v.is_array() {
        Ok(())
    } else {
        Err(Error::Request(
            "`instructions` must be a string, object, or array".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// The top-level response body returned by `POST /v1/systemone`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneResponse {
    /// The model that performed the evaluation.
    pub model: String,
    /// One answer per question, keyed by the same ids used in the request.
    pub answers: BTreeMap<String, Answer>,
    /// Token usage for the request.
    pub usage: Usage,
    /// Engine-specific extras. Off by default; only present when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Extensions>,
}

impl SystemOneResponse {
    pub fn new(model: String, answers: BTreeMap<String, Answer>, usage: Usage) -> Self {
        SystemOneResponse {
            model,
            answers,
            usage,
            extensions: None,
        }
    }
}

/// A typed answer. The `type` field matches the question type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// A choice answer: the selected option plus the full distribution.
    Choice {
        /// The option with the highest probability after calibration.
        choice: String,
        /// Every option mapped to its calibrated probability (sums to 1).
        probabilities: BTreeMap<String, f32>,
        /// Confidence in `[0, 1]`, derived from the probability shape.
        confidence: f32,
    },
    /// A score answer: the probability-weighted position along the levels.
    Score {
        /// Each level number (string key) mapped to its probability.
        probabilities: BTreeMap<String, f32>,
        /// The probability-weighted position; can fall between levels.
        score: f32,
        /// Each level number mapped back to its description.
        legend: BTreeMap<String, String>,
        /// Confidence in `[0, 1]`.
        confidence: f32,
    },
    /// A Noul answer: the single probability that the answer is yes.
    Noul {
        /// Probability the answer is yes, in `[0, 1]`.
        noul: f32,
    },
}

/// Token usage for a request. `huncho` is prefill-only, so `output_tokens` is
/// always 0 for the contract-visible count.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    pub fn new(input_tokens: u64) -> Self {
        Usage {
            input_tokens,
            output_tokens: 0,
        }
    }
}

/// Engine-specific extras (API-05). Off by default so default responses stay
/// strictly Jev-shaped.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Extensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_definition: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_contract_hash: Option<String>,
    /// Raw per-option logits (pre-softmax) for each question id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_logits: Option<BTreeMap<String, Vec<f32>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> SystemOneRequest {
        let json = serde_json::json!({
            "state": "A support ticket about a refund",
            "model": "jest-mock",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team handles this?",
                    "criteria": { "returns": "The customer wants money back", "billing": "Charge problem" }
                },
                "is_refund": {
                    "type": "noul",
                    "instructions": "The customer is requesting a refund",
                    "criteria": { "true": "Asks for money back", "false": "Does not ask" }
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate severity",
                    "criteria": ["Minor", "Moderate", "Severe"]
                }
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn parses_all_question_types() {
        let req = sample_request();
        assert_eq!(req.questions.len(), 3);
        assert!(req.validate().is_ok());
        assert_eq!(req.questions["department"].type_name(), "choice");
        assert_eq!(req.questions["is_refund"].type_name(), "noul");
        assert_eq!(req.questions["severity"].type_name(), "score");
    }

    #[test]
    fn rejects_too_many_options() {
        let mut req = sample_request();
        // 300 options
        let mut criteria = indexmap::IndexMap::new();
        for i in 0..300 {
            criteria.insert(format!("opt{i}"), None);
        }
        req.questions.insert(
            "big".into(),
            Question::Choice {
                instructions: Instructions(serde_json::json!("x")),
                criteria,
            },
        );
        assert!(req.validate().is_err());
    }

    #[test]
    fn rejects_bad_state() {
        let json = serde_json::json!({
            "state": 42,
            "model": "m",
            "questions": { "q": { "type": "noul", "instructions": "x" } }
        });
        let req: SystemOneRequest = serde_json::from_value(json).unwrap();
        assert!(req.validate().is_err());
    }

    #[test]
    fn response_round_trips() {
        let mut answers = BTreeMap::new();
        answers.insert(
            "department".into(),
            Answer::Choice {
                choice: "returns".into(),
                probabilities: [("returns".into(), 0.9f32)].into_iter().collect(),
                confidence: 1.0,
            },
        );
        answers.insert("is_refund".into(), Answer::Noul { noul: 0.95 });
        let resp = SystemOneResponse::new(
            "m".into(),
            answers,
            Usage::new(120),
        );
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("\"model\""));
        assert!(s.contains("\"department\""));
        let _: SystemOneResponse = serde_json::from_str(&s).unwrap();
    }
}
