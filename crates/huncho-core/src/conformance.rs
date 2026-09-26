//! Conformance tooling (PRD §6.3, CONF-01/02/03).
//!
//! A reference implementation per family produces golden outputs (fixed inputs
//! -> probability vectors). [`ConformanceSuite::run`] evaluates any backend
//! against those golden vectors and reports max probability delta, argmax
//! agreement, and ECE drift against configurable thresholds.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::contract::{Answer, SystemOneRequest};
use crate::engine::{Engine, EvalOptions};
use crate::error::{Error, Result};

/// Pass/fail thresholds for a conformance run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceThresholds {
    /// Max allowed absolute delta per probability value.
    #[serde(default = "default_max_delta")]
    pub max_prob_delta: f32,
    /// Minimum fraction of cases whose argmax agrees with the reference.
    #[serde(default = "default_argmax")]
    pub min_argmax_agreement: f32,
    /// Max allowed ECE drift.
    #[serde(default = "default_ece")]
    pub max_ece: f32,
}

fn default_max_delta() -> f32 {
    1e-3
}
fn default_argmax() -> f32 {
    1.0
}
fn default_ece() -> f32 {
    0.02
}

impl Default for ConformanceThresholds {
    fn default() -> Self {
        ConformanceThresholds {
            max_prob_delta: default_max_delta(),
            min_argmax_agreement: default_argmax(),
            max_ece: default_ece(),
        }
    }
}

/// A single golden case: a request plus the reference probability vectors,
/// keyed by question id and then by candidate label (in candidate order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenCase {
    pub id: String,
    pub request: SystemOneRequest,
    pub expected: BTreeMap<String, BTreeMap<String, f32>>,
}

/// A golden-vector file (CONF-01 output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenSuite {
    pub schema_version: String,
    pub family: String,
    #[serde(default)]
    pub hash: Option<String>,
    pub cases: Vec<GoldenCase>,
}

/// Per-case conformance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseReport {
    pub id: String,
    pub max_prob_delta: f32,
    pub argmax_match: bool,
    pub ece: f32,
}

/// The full conformance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceReport {
    pub model: String,
    pub backend: String,
    pub dtype: String,
    pub cases: Vec<CaseReport>,
    pub max_prob_delta: f32,
    pub argmax_agreement: f32,
    pub ece: f32,
    pub passed: bool,
}

/// Run a golden suite against an engine and produce a report.
pub fn run_suite(engine: &Engine, suite: &GoldenSuite, thresholds: &ConformanceThresholds) -> Result<ConformanceReport> {
    let mut cases = Vec::new();
    let mut global_max = 0.0f32;
    let mut argmax_matches = 0usize;
    // Backend-side ECE inputs (backend confidence vs. whether it matched the
    // reference argmax) and reference-side ECE inputs (reference confidence vs.
    // its own argmax, which is definitionally correct). The reported ECE is the
    // *drift* between the two, per the PRD's "ECE drift bound".
    let mut back_preds = Vec::new();
    let mut back_correct = Vec::new();
    let mut ref_preds = Vec::new();
    let mut ref_correct = Vec::new();

    for case in &suite.cases {
        // Validate the case's family matches the engine's model.
        let case_family = crate::manifest::Family::parse(&suite.family)?;
        if case_family != engine.family() {
            return Err(Error::Conformance(format!(
                "golden suite is for {case_family} but the engine model is {}",
                engine.family()
            )));
        }

        let resp = engine.eval(&case.request, &EvalOptions::default())?;
        let mut case_max = 0.0f32;
        let mut case_argmax_match = true;
        let mut case_preds = Vec::new();
        let mut case_correct = Vec::new();

        for (qid, expected_map) in &case.expected {
            let answer = resp
                .answers
                .get(qid)
                .ok_or_else(|| Error::Conformance(format!("case `{}` missing answer for `{qid}`", case.id)))?;

            let probs = answer_probabilities(answer)?;

            // Union of labels.
            for (label, &e) in expected_map {
                let p = probs.get(label).copied().unwrap_or(0.0);
                let delta = (p - e).abs();
                if delta > case_max {
                    case_max = delta;
                }
            }
            // Also ensure every predicted label was covered by expected (no extra mass).
            for (label, &p) in &probs {
                if !expected_map.contains_key(label) && p > 1e-6 {
                    case_max = case_max.max(p);
                }
            }

            // Argmax agreement (compare by label, since map ordering may differ).
            let exp_label = argmax_label(expected_map);
            let pred_label = argmax_label(&probs);
            if exp_label != pred_label {
                case_argmax_match = false;
            }
            // Backend confidence in its own top prediction, and whether that
            // prediction agrees with the reference.
            let (pred, correct) = (probs[&pred_label], exp_label == pred_label);
            case_preds.push(pred);
            case_correct.push(correct);
            back_preds.push(pred);
            back_correct.push(correct);
            // Reference confidence in its top prediction, which is by definition
            // correct against itself.
            ref_preds.push(expected_map[&exp_label]);
            ref_correct.push(true);
        }

        if case_max > global_max {
            global_max = case_max;
        }
        if case_argmax_match {
            argmax_matches += 1;
        }
        cases.push(CaseReport {
            id: case.id.clone(),
            max_prob_delta: case_max,
            argmax_match: case_argmax_match,
            ece: crate::calibration::ece(&case_preds, &case_correct, 10),
        });
    }

    let total = cases.len().max(1);
    let argmax_agreement = argmax_matches as f32 / total as f32;
    let back_ece = crate::calibration::ece(&back_preds, &back_correct, 10);
    let ref_ece = crate::calibration::ece(&ref_preds, &ref_correct, 10);
    let ece = (back_ece - ref_ece).abs();

    let passed = global_max <= thresholds.max_prob_delta
        && argmax_agreement >= thresholds.min_argmax_agreement
        && ece <= thresholds.max_ece;

    Ok(ConformanceReport {
        model: engine.manifest().name.clone(),
        backend: engine.backend_id().to_string(),
        dtype: engine.dtype().to_string(),
        cases,
        max_prob_delta: global_max,
        argmax_agreement,
        ece,
        passed,
    })
}

/// Load a golden suite from a JSON file.
pub fn load_suite(path: impl AsRef<std::path::Path>) -> Result<GoldenSuite> {
    let bytes = std::fs::read(path.as_ref())?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Save a golden suite to a JSON file.
pub fn save_suite(suite: &GoldenSuite, path: impl AsRef<std::path::Path>) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(suite)?;
    Ok(std::fs::write(path, bytes)?)
}

/// Extract an ordered label->probability map from an answer.
fn answer_probabilities(answer: &Answer) -> Result<BTreeMap<String, f32>> {
    Ok(match answer {
        Answer::Choice { probabilities, .. } => probabilities.clone(),
        Answer::Score { probabilities, .. } => probabilities.clone(),
        Answer::Noul { noul: p } => {
            let mut m = BTreeMap::new();
            m.insert("yes".to_string(), *p);
            m.insert("no".to_string(), 1.0 - p);
            m
        }
    })
}

fn argmax_label(map: &BTreeMap<String, f32>) -> String {
    map.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k.clone())
        .unwrap_or_default()
}
