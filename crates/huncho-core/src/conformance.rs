//! Conformance tooling (PRD §6.3, CONF-01/02/03).
//!
//! A reference implementation per family produces golden outputs (fixed inputs
//! -> probability vectors). [`run_suite`] evaluates any backend
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
/// keyed by question id and then by candidate label. Candidate order comes
/// from the request and the loaded prompt contract, not this sorted map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenCase {
    pub id: String,
    pub request: SystemOneRequest,
    pub expected: BTreeMap<String, BTreeMap<String, f32>>,
    /// Optional observed target labels. If any case has targets, every question
    /// in the suite must have one; unlabeled suites measure reference fidelity.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub targets: BTreeMap<String, String>,
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
    #[serde(default)]
    pub device: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub execution_metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub reference_readout: bool,
    #[serde(default)]
    pub prefix_cache: bool,
    #[serde(default)]
    pub max_batch_tokens: Option<usize>,
    #[serde(default)]
    pub prepare_all: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_request_max_requests: Option<usize>,
    #[serde(default)]
    pub work: crate::engine::EvalStats,
    pub cases: Vec<CaseReport>,
    pub max_prob_delta: f32,
    pub argmax_agreement: f32,
    pub ece: f32,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_calibration: Option<OutcomeCalibration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimization_parity: Option<OptimizationParity>,
}

/// Paired qualification against independent forwards on this loaded engine,
/// in addition to the unchanged external golden vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParity {
    pub max_prob_delta: f32,
    pub argmax_agreement: f32,
}

/// Metrics against observed outcomes, distinct from reference agreement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeCalibration {
    pub questions: usize,
    pub backend_ece: f32,
    pub reference_ece: f32,
    /// Multiclass Brier score: mean sum of squared errors per question.
    pub backend_brier: f64,
    pub reference_brier: f64,
}

/// Run a golden suite against an engine and produce a report.
pub fn run_suite(
    engine: &Engine,
    suite: &GoldenSuite,
    thresholds: &ConformanceThresholds,
) -> Result<ConformanceReport> {
    run_suite_with_options(engine, suite, thresholds, &EvalOptions::default())
}

/// Run the same unchanged golden vectors with an explicit execution option,
/// including the legacy F3 readout used for optimization parity checks.
pub fn run_suite_with_options(
    engine: &Engine,
    suite: &GoldenSuite,
    thresholds: &ConformanceThresholds,
    options: &EvalOptions,
) -> Result<ConformanceReport> {
    run_suite_impl(engine, suite, thresholds, options, None)
}

/// Qualify actual cross-request tensor collation, with fresh preparations and
/// independent forwards. A suite that never mixes two requests cannot pass.
pub fn run_suite_with_cross_request_batches(
    engine: &Engine,
    suite: &GoldenSuite,
    thresholds: &ConformanceThresholds,
    options: &EvalOptions,
    max_requests: usize,
) -> Result<ConformanceReport> {
    if !(2..=64).contains(&max_requests) || !engine.supports_batch() {
        return Err(Error::Conformance(
            "cross-request qualification requires a native batch backend and 2–64 requests".into(),
        ));
    }
    let budget = options.max_batch_tokens.ok_or_else(|| {
        Error::Conformance("cross-request qualification requires a token budget".into())
    })?;
    let mut responses = Vec::with_capacity(suite.cases.len());
    let mut work = crate::engine::EvalStats::default();
    for group in suite.cases.chunks(max_requests) {
        let packets = group
            .iter()
            .map(|case| {
                engine.prepare_eval_uncached_with_stats(
                    case.request.clone(),
                    options.clone(),
                    &mut Default::default(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut group_work = Default::default();
        let result = engine.eval_prepared_batch_with_stats(packets, budget, &mut group_work);
        work.accumulate(&group_work);
        responses.extend(result?);
    }
    if work.cross_request_batches == 0 {
        return Err(Error::Conformance(
            "cross-request qualification requires a native batch containing questions from distinct requests".into(),
        ));
    }
    let mut report = run_suite_impl(engine, suite, thresholds, options, Some((&responses, work)))?;
    report.cross_request_max_requests = Some(max_requests);
    Ok(report)
}

fn run_suite_impl(
    engine: &Engine,
    suite: &GoldenSuite,
    thresholds: &ConformanceThresholds,
    options: &EvalOptions,
    collated: Option<(
        &[crate::contract::SystemOneResponse],
        crate::engine::EvalStats,
    )>,
) -> Result<ConformanceReport> {
    let case_family = crate::manifest::Family::parse(&suite.family)?;
    if case_family != engine.family() {
        return Err(Error::Conformance(format!(
            "golden suite is for {case_family} but the engine model is {}",
            engine.family()
        )));
    }
    if suite.cases.is_empty() {
        return Err(Error::Conformance("golden suite must contain cases".into()));
    }
    for threshold in [
        thresholds.max_prob_delta,
        thresholds.min_argmax_agreement,
        thresholds.max_ece,
    ] {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(Error::Conformance(
                "thresholds must be finite and between zero and one".into(),
            ));
        }
    }
    let mut cases = Vec::new();
    let (responses, mut work) = match collated {
        Some((responses, work)) => (Some(responses), work),
        None => (None, Default::default()),
    };
    let labeled = suite.cases.iter().any(|case| !case.targets.is_empty());
    let mut backend_brier = 0.0f64;
    let mut reference_brier = 0.0f64;
    let mut global_max = 0.0f32;
    let candidate_readout =
        engine.family() == crate::manifest::Family::F3 && !options.reference_readout;
    let paired = options.prefix_cache
        || options.max_batch_tokens.is_some()
        || candidate_readout
        || options.prepare_all;
    let mut independent_options = options.clone();
    independent_options.prefix_cache = false;
    independent_options.max_batch_tokens = None;
    independent_options.prepare_all = false;
    if candidate_readout {
        independent_options.reference_readout = true;
    }
    let mut parity_max = 0.0f32;
    let mut parity_matches = 0usize;
    let mut parity_questions = 0usize;
    let mut argmax_matches = 0usize;
    // With observed labels, ECE measures outcome calibration. Legacy unlabeled
    // golden suites retain their reference-agreement fidelity metric.
    let mut back_preds = Vec::new();
    let mut back_correct = Vec::new();
    let mut ref_preds = Vec::new();
    let mut ref_correct = Vec::new();

    for (index, case) in suite.cases.iter().enumerate() {
        if case.request.questions.len() != case.expected.len()
            || case
                .request
                .questions
                .keys()
                .any(|qid| !case.expected.contains_key(qid))
        {
            return Err(Error::Conformance(format!(
                "case `{}` must cover every requested question exactly",
                case.id
            )));
        }
        for (qid, probabilities) in &case.expected {
            validate_probabilities(probabilities, &case.id, qid)?;
            if labeled
                && !case
                    .targets
                    .get(qid)
                    .is_some_and(|target| probabilities.contains_key(target))
            {
                return Err(Error::Conformance(format!(
                    "labeled case `{}` requires an observed target for question `{qid}`",
                    case.id
                )));
            }
        }
        if labeled && case.targets.len() != case.expected.len() {
            return Err(Error::Conformance(format!(
                "case `{}` has extra target labels",
                case.id
            )));
        }

        let mut stats = crate::engine::EvalStats::default();
        let resp = match responses {
            Some(responses) => responses
                .get(index)
                .ok_or_else(|| Error::Conformance("collation omitted a case".into()))?
                .clone(),
            None => engine.eval_uncached_with_stats(&case.request, options, &mut stats)?,
        };
        work.accumulate(&stats);
        let independent = paired
            .then(|| {
                engine.eval_uncached_with_stats(
                    &case.request,
                    &independent_options,
                    &mut Default::default(),
                )
            })
            .transpose()?;
        let mut case_max = 0.0f32;
        let mut case_argmax_match = true;
        let mut case_preds = Vec::new();
        let mut case_correct = Vec::new();

        for (qid, expected_map) in &case.expected {
            let labels =
                engine.candidate_labels(&case.request.state, &case.request.questions[qid])?;
            let answer = resp.answers.get(qid).ok_or_else(|| {
                Error::Conformance(format!("case `{}` missing answer for `{qid}`", case.id))
            })?;

            let probs = answer_probabilities(answer)?;
            validate_probabilities(&probs, &case.id, qid)?;
            if let Some(independent) = &independent {
                let independent =
                    answer_probabilities(independent.answers.get(qid).ok_or_else(|| {
                        Error::Conformance("independent forward omitted a question".into())
                    })?)?;
                validate_probabilities(&independent, &case.id, qid)?;
                if probs.keys().ne(independent.keys()) {
                    return Err(Error::Conformance(
                        "optimized/independent label sets differ".into(),
                    ));
                }
                parity_questions += 1;
                parity_matches += usize::from(
                    argmax_label(&probs, &labels)? == argmax_label(&independent, &labels)?,
                );
                for (label, value) in &probs {
                    parity_max = parity_max.max((value - independent[label]).abs());
                }
            }
            if probs.keys().ne(expected_map.keys()) {
                return Err(Error::Conformance(format!(
                    "case `{}` question `{qid}` has mismatched reference/output labels",
                    case.id
                )));
            }

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
            let exp_label = argmax_label(expected_map, &labels)?;
            let pred_label = argmax_label(&probs, &labels)?;
            if exp_label != pred_label {
                case_argmax_match = false;
            }
            let target = case.targets.get(qid);
            let correct = target.map_or(exp_label == pred_label, |label| label == &pred_label);
            let pred = probs[&pred_label];
            case_preds.push(pred);
            case_correct.push(correct);
            back_preds.push(pred);
            back_correct.push(correct);
            ref_preds.push(expected_map[&exp_label]);
            ref_correct.push(match target {
                Some(label) => label == &exp_label,
                None => true,
            });
            if let Some(target) = target {
                backend_brier += brier(&probs, target);
                reference_brier += brier(expected_map, target);
            }
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

    if options.prefix_cache && work.cache_forks == 0 {
        return Err(Error::Conformance("prefix-cache qualification requires a supported multi-question Kev case that actually forks".into()));
    }
    if options.max_batch_tokens.is_some() && work.batch_calls == 0 {
        return Err(Error::Conformance("batch qualification requires a supported case that actually batches equal-length questions".into()));
    }
    if options.prepare_all && work.prepared_questions == 0 {
        return Err(Error::Conformance(
            "prepared qualification requires actual F1–F4 prompt preparation".into(),
        ));
    }
    let total = cases.len().max(1);
    let argmax_agreement = argmax_matches as f32 / total as f32;
    let back_ece = crate::calibration::ece(&back_preds, &back_correct, 10);
    let ref_ece = crate::calibration::ece(&ref_preds, &ref_correct, 10);
    let ece = (back_ece - ref_ece).abs();
    let outcome_calibration = labeled.then(|| OutcomeCalibration {
        questions: back_preds.len(),
        backend_ece: back_ece,
        reference_ece: ref_ece,
        backend_brier: backend_brier / back_preds.len() as f64,
        reference_brier: reference_brier / back_preds.len() as f64,
    });

    let optimization_parity = paired.then(|| OptimizationParity {
        max_prob_delta: parity_max,
        argmax_agreement: parity_matches as f32 / parity_questions.max(1) as f32,
    });
    let passed = global_max <= thresholds.max_prob_delta
        && argmax_agreement >= thresholds.min_argmax_agreement
        && ece <= thresholds.max_ece
        && !optimization_parity
            .as_ref()
            .is_some_and(|parity| parity.max_prob_delta > 1e-4 || parity.argmax_agreement != 1.0);

    Ok(ConformanceReport {
        model: engine.manifest().name.clone(),
        backend: engine.backend_id().to_string(),
        dtype: engine.dtype().to_string(),
        device: engine.device().to_string(),
        execution_metadata: engine.execution_metadata().clone(),
        reference_readout: options.reference_readout,
        prefix_cache: options.prefix_cache,
        max_batch_tokens: options.max_batch_tokens,
        prepare_all: options.prepare_all,
        cross_request_max_requests: None,
        work,
        cases,
        max_prob_delta: global_max,
        argmax_agreement,
        ece,
        passed,
        outcome_calibration,
        optimization_parity,
    })
}

fn validate_probabilities(map: &BTreeMap<String, f32>, case: &str, question: &str) -> Result<()> {
    let sum: f64 = map.values().map(|&value| value as f64).sum();
    if map.is_empty()
        || map
            .values()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (sum - 1.0).abs() > 1e-4
    {
        return Err(Error::Conformance(format!(
            "case `{case}` question `{question}` requires finite normalized probabilities"
        )));
    }
    Ok(())
}

fn brier(probabilities: &BTreeMap<String, f32>, target: &str) -> f64 {
    probabilities
        .iter()
        .map(|(label, &probability)| {
            let expected = if label == target { 1.0 } else { 0.0 };
            (probability as f64 - expected).powi(2)
        })
        .sum()
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
pub(crate) fn answer_probabilities(answer: &Answer) -> Result<BTreeMap<String, f32>> {
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

fn argmax_label(map: &BTreeMap<String, f32>, labels: &[String]) -> Result<String> {
    let first = labels
        .first()
        .ok_or_else(|| Error::Conformance("question must have at least one candidate".into()))?;
    let mut best = first;
    let mut peak = f32::NEG_INFINITY;
    if labels.len() != map.len() {
        return Err(Error::Conformance(
            "probability/candidate counts differ".into(),
        ));
    }
    for label in labels {
        let probability = map
            .get(label)
            .ok_or_else(|| Error::Conformance(format!("probabilities omit candidate `{label}`")))?;
        // Strict comparison retains the first candidate on exact ties, just
        // like Engine::build_answer. BTreeMap iteration loses that order.
        if *probability > peak {
            best = label;
            peak = *probability;
        }
    }
    Ok(best.clone())
}
