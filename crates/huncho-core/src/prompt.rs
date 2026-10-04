//! Per-family prompt / contract building (PRD §6.2 `prompt_contract`).
//!
//! The engine builds a [`BuiltPrompt`] for each question. The exact template is
//! pinned by the manifest's `prompt_contract.template`; `huncho convert` extracts the
//! reference template into the package so prompts stay byte-identical to the
//! reference implementation. This module ships canonical default templates for
//! each family and a [`PromptFormatter`] trait for custom ones.

use crate::contract::{Question, StateValue};
use crate::error::{Error, Result};
use crate::manifest::{F3Config, Family, ModelManifest};
use crate::tokenizer::Tokenizer;

mod kev;
pub use kev::KevFormatter;
pub mod clef;

/// The mask/special-token string used by ModernBERT-based decision models (Laya).
const LAYA_MASK_STR: &str = "[MASK]";
/// Laya caps each option description at this many tokens.
const LAYA_OPTION_MAX_TOKENS: usize = 48;

/// What a candidate represents, which drives how its logits become an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    Option,
    Level,
    YesNo,
}

/// A single option/level/yes-no candidate that the head scores.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub kind: CandidateKind,
    /// Token position to read features/logits from.
    pub position: usize,
    /// Token id used for candidate-logit style reading (`logits[pos][code_id]`).
    pub code_id: u32,
    /// The label: an option name, `"yes"`/`"no"`, or a level index string.
    pub label: String,
    /// Optional human description (used as the Score `legend`).
    pub description: Option<String>,
    /// Order among candidates (0-based).
    pub index: usize,
}

/// The result of building a prompt for one question.
#[derive(Debug, Clone)]
pub struct BuiltPrompt {
    pub tokens: Vec<u32>,
    pub candidates: Vec<Candidate>,
    /// Length of the shared state/instructions prefix (the F2 fork boundary).
    pub prefix_len: usize,
    /// Typed-question index for the Laya decision head type embedding
    /// (choice=0, score=1, noul=2). Ignored by non-Laya formatters.
    pub qtype: u32,
}

impl BuiltPrompt {
    /// Number of tokens in the prompt (excluding the trailing answer marker,
    /// if the family appends one). Used for usage / budget reporting.
    pub fn token_len(&self) -> usize {
        self.tokens.len()
    }
}

/// Builds a prompt for one question against a state.
pub trait PromptFormatter: Send + Sync {
    /// Which family this formatter targets.
    fn family(&self) -> Family;
    /// Build the prompt.
    fn build(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt>;
}

/// Return the default [`PromptFormatter`] for a family.
pub fn default_formatter(family: Family) -> Box<dyn PromptFormatter> {
    Box::new(DefaultFormatter { family })
}

/// Select the [`PromptFormatter`] pinned by a manifest's prompt contract.
///
/// * `laya-v1` selects the Laya option-marker formatter (typed `[MASK]` markers).
/// * `nimble-v1` selects the F3 candidate-logit formatter (single-field schema
///   + one-token answer codes via the LM head), using the manifest's `f3`
///   codebook.
/// * All other templates use the family default.
pub fn formatter_for(manifest: &ModelManifest) -> Box<dyn PromptFormatter> {
    let pc = &manifest.prompt_contract;
    match pc.template.as_str() {
        "kev-v1" => Box::new(KevFormatter {
            max_state: pc.state_budget,
            max_row: manifest.backbone.max_context,
        }),
        "laya-v1" => Box::new(LayaFormatter {
            max_len: pc.max_len,
            head_max_len: pc.head_max_len,
        }),
        "nimble-v1" => match &manifest.f3 {
            Some(config) => Box::new(NimbleFormatter {
                config: config.clone(),
            }),
            // A `nimble-v1` contract without an `f3` codebook cannot build the
            // candidate codes; fall back to the family default (which errors
            // informatively on a malformed package via manifest validation).
            None => Box::new(DefaultFormatter {
                family: manifest.family,
            }),
        },
        _ => Box::new(DefaultFormatter {
            family: manifest.family,
        }),
    }
}

/// Appends the textual representation of the question's instructions.
fn instructions_text(q: &Question) -> String {
    match q.instructions().as_value() {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Render the state to a single string. Objects/arrays are serialized to JSON
/// (compact), strings are used verbatim.
fn state_text(state: &StateValue) -> String {
    match state.as_value() {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Render a criteria value (option description / level description) to text.
fn criteria_text(v: &Option<serde_json::Value>) -> String {
    match v {
        None => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Render a Score level (a bare serde_json::Value) to plain text. String
/// levels are used verbatim; structured levels are serialized to JSON.
fn level_text(level: &serde_json::Value) -> String {
    match level {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The generic default formatter.
struct DefaultFormatter {
    family: Family,
}

impl DefaultFormatter {
    fn option_marker(tokenizer: &dyn Tokenizer, i: usize) -> u32 {
        tokenizer
            .id_for(&format!("<option:{i}>"))
            .unwrap_or(0)
    }
}

impl PromptFormatter for DefaultFormatter {
    fn family(&self) -> Family {
        self.family
    }

    fn build(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt> {
        match self.family {
            Family::F1 => self.build_f1(state, question, tokenizer),
            Family::F2 => self.build_f2(state, question, tokenizer),
            Family::F3 => self.build_f3(state, question, tokenizer),
            Family::F4 => self.build_f4(state, question, tokenizer),
            Family::F5 => Err(Error::Unsupported(
                "joint-schema models require a whole-request forward".into(),
            )),
        }
    }
}

impl DefaultFormatter {
    fn build_f1(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt> {
        let mut text = format!(
            "[state]\n{}\n\nQuestion: {}\n\nOptions:\n",
            state_text(state),
            instructions_text(question)
        );
        match question {
            Question::Choice { criteria, .. } => {
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    text.push_str(&format!(
                        "<option:{i}> {label}: {}\n",
                        criteria_text(desc)
                    ));
                }
            }
            Question::Score { criteria, .. } => {
                for (i, level) in criteria.iter().enumerate() {
                    text.push_str(&format!("<option:{i}> {level}\n"));
                }
            }
            Question::Noul { .. } => {
                text.push_str("<option:0> yes\n<option:1> no\n");
            }
        }
        let tokens = tokenizer.encode(&text, true)?;

        // Record the position of each option marker.
        let mut candidates = Vec::new();
        match question {
            Question::Choice { criteria, .. } => {
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    let marker = Self::option_marker(tokenizer, i);
                    let position = tokens
                        .iter()
                        .position(|&t| t == marker)
                        .ok_or_else(|| Error::Request(format!("option marker {i} not found")))?;
                    candidates.push(Candidate {
                        kind: CandidateKind::Option,
                        position,
                        code_id: marker,
                        label: label.clone(),
                        description: {
                            let d = criteria_text(desc);
                            if d.is_empty() {
                                None
                            } else {
                                Some(d)
                            }
                        },
                        index: i,
                    });
                }
            }
            Question::Score { criteria, .. } => {
                for (i, level) in criteria.iter().enumerate() {
                    let marker = Self::option_marker(tokenizer, i);
                    let position = tokens
                        .iter()
                        .position(|&t| t == marker)
                        .ok_or_else(|| Error::Request(format!("level marker {i} not found")))?;
                    candidates.push(Candidate {
                        kind: CandidateKind::Level,
                        position,
                        code_id: marker,
                        label: i.to_string(),
                        description: Some(level_text(level)),
                        index: i,
                    });
                }
            }
            Question::Noul { .. } => {
                let yes_pos = tokens
                    .iter()
                    .position(|&t| t == Self::option_marker(tokenizer, 0))
                    .ok_or_else(|| Error::Request("yes marker not found".into()))?;
                let no_pos = tokens
                    .iter()
                    .position(|&t| t == Self::option_marker(tokenizer, 1))
                    .ok_or_else(|| Error::Request("no marker not found".into()))?;
                candidates = vec![
                    Candidate {
                        kind: CandidateKind::YesNo,
                        position: yes_pos,
                        code_id: Self::option_marker(tokenizer, 0),
                        label: "yes".into(),
                        description: None,
                        index: 0,
                    },
                    Candidate {
                        kind: CandidateKind::YesNo,
                        position: no_pos,
                        code_id: Self::option_marker(tokenizer, 1),
                        label: "no".into(),
                        description: None,
                        index: 1,
                    },
                ];
            }
        }

        let prefix_len = tokens
            .iter()
            .position(|&t| t == Self::option_marker(tokenizer, 0))
            .unwrap_or(tokens.len());

        Ok(BuiltPrompt {
            tokens,
            candidates,
            prefix_len,
            qtype: question.qtype_index(),
        })
    }

    fn build_f2(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt> {
        // Kev: shared state prefix, then one isolated branch per question.
        // Pointer head reads hidden states at option-boundary markers.
        let mut text = format!(
            "{}\n\nQuestion: {}\n\nOptions:\n",
            state_text(state),
            instructions_text(question)
        );
        match question {
            Question::Choice { criteria, .. } => {
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    text.push_str(&format!(
                        "<option:{i}> {label}: {}\n",
                        criteria_text(desc)
                    ));
                }
            }
            Question::Score { criteria, .. } => {
                for (i, level) in criteria.iter().enumerate() {
                    text.push_str(&format!("<option:{i}> {level}\n"));
                }
            }
            Question::Noul { .. } => {
                text.push_str("<option:0> yes\n<option:1> no\n");
            }
        }
        let tokens = tokenizer.encode(&text, true)?;
        // For F2 the block-causal fork boundary is the end of the shared prefix.
        let prefix_len = tokens
            .iter()
            .position(|&t| t == Self::option_marker(tokenizer, 0))
            .unwrap_or(tokens.len());

        let mut candidates = Vec::new();
        match question {
            Question::Choice { criteria, .. } => {
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    let marker = Self::option_marker(tokenizer, i);
                    let position = tokens
                        .iter()
                        .position(|&t| t == marker)
                        .unwrap_or(tokens.len().saturating_sub(1));
                    candidates.push(Candidate {
                        kind: CandidateKind::Option,
                        position,
                        code_id: marker,
                        label: label.clone(),
                        description: {
                            let d = criteria_text(desc);
                            if d.is_empty() {
                                None
                            } else {
                                Some(d)
                            }
                        },
                        index: i,
                    });
                }
            }
            Question::Score { criteria, .. } => {
                for (i, level) in criteria.iter().enumerate() {
                    let marker = Self::option_marker(tokenizer, i);
                    let position = tokens
                        .iter()
                        .position(|&t| t == marker)
                        .unwrap_or(tokens.len().saturating_sub(1));
                    candidates.push(Candidate {
                        kind: CandidateKind::Level,
                        position,
                        code_id: marker,
                        label: i.to_string(),
                        description: Some(level_text(level)),
                        index: i,
                    });
                }
            }
            Question::Noul { .. } => {
                let yes_m = Self::option_marker(tokenizer, 0);
                let no_m = Self::option_marker(tokenizer, 1);
                let yes_pos = tokens.iter().position(|&t| t == yes_m).unwrap_or(0);
                let no_pos = tokens.iter().position(|&t| t == no_m).unwrap_or(0);
                candidates = vec![
                    Candidate {
                        kind: CandidateKind::YesNo,
                        position: yes_pos,
                        code_id: yes_m,
                        label: "yes".into(),
                        description: None,
                        index: 0,
                    },
                    Candidate {
                        kind: CandidateKind::YesNo,
                        position: no_pos,
                        code_id: no_m,
                        label: "no".into(),
                        description: None,
                        index: 1,
                    },
                ];
            }
        }
        Ok(BuiltPrompt {
            tokens,
            candidates,
            prefix_len,
            qtype: question.qtype_index(),
        })
    }

    fn build_f3(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt> {
        // Nimble candidate-logit: one-token answer codes via the LM head.
        let mut text = format!(
            "{}\n\nQuestion: {}\n\nOptions:\n",
            state_text(state),
            instructions_text(question)
        );
        let letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
        match question {
            Question::Choice { criteria, .. } => {
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    let letter = letters.get(i).copied().unwrap_or("?");
                    text.push_str(&format!(
                        "{letter}. {label}: {}\n",
                        criteria_text(desc)
                    ));
                }
            }
            Question::Score { criteria, .. } => {
                for (i, level) in criteria.iter().enumerate() {
                    let letter = letters.get(i).copied().unwrap_or("?");
                    text.push_str(&format!("{letter}. {level}\n"));
                }
            }
            Question::Noul { .. } => {
                text.push_str("A. yes\nB. no\n");
            }
        }
        // Append the answer marker whose logits are read.
        text.push_str("\nAnswer: <answer>");
        let tokens = tokenizer.encode(&text, true)?;
        let answer_id = tokenizer.id_for("<answer>").unwrap_or(0);
        let position = tokens
            .iter()
            .rposition(|&t| t == answer_id)
            .ok_or_else(|| Error::Request("answer marker not found".into()))?;

        let mut candidates = Vec::new();
        match question {
            Question::Choice { criteria, .. } => {
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    let letter = letters.get(i).copied().unwrap_or("?");
                    candidates.push(Candidate {
                        kind: CandidateKind::Option,
                        position,
                        code_id: tokenizer.id_for(letter).unwrap_or(0),
                        label: label.clone(),
                        description: {
                            let d = criteria_text(desc);
                            if d.is_empty() {
                                None
                            } else {
                                Some(d)
                            }
                        },
                        index: i,
                    });
                }
            }
            Question::Score { criteria, .. } => {
                for (i, level) in criteria.iter().enumerate() {
                    let letter = letters.get(i).copied().unwrap_or("?");
                    candidates.push(Candidate {
                        kind: CandidateKind::Level,
                        position,
                        code_id: tokenizer.id_for(letter).unwrap_or(0),
                        label: i.to_string(),
                        description: Some(level_text(level)),
                        index: i,
                    });
                }
            }
            Question::Noul { .. } => {
                candidates = vec![
                    Candidate {
                        kind: CandidateKind::YesNo,
                        position,
                        code_id: tokenizer.id_for("A").unwrap_or(0),
                        label: "yes".into(),
                        description: None,
                        index: 0,
                    },
                    Candidate {
                        kind: CandidateKind::YesNo,
                        position,
                        code_id: tokenizer.id_for("B").unwrap_or(0),
                        label: "no".into(),
                        description: None,
                        index: 1,
                    },
                ];
            }
        }
        let prefix_len = tokens.len().saturating_sub(1);
        Ok(BuiltPrompt {
            tokens,
            candidates,
            prefix_len,
            qtype: question.qtype_index(),
        })
    }

    fn build_f4(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt> {
        // OpenThai slot head: single forward pass, fixed-width decision head.
        let mut text = format!(
            "[state]\n{}\n\nQuestion: {}\n\nOptions:\n",
            state_text(state),
            instructions_text(question)
        );
        match question {
            Question::Choice { criteria, .. } => {
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    text.push_str(&format!(
                        "<slot:{i}> {label}: {}\n",
                        criteria_text(desc)
                    ));
                }
            }
            Question::Score { criteria, .. } => {
                for (i, level) in criteria.iter().enumerate() {
                    text.push_str(&format!("<slot:{i}> {level}\n"));
                }
            }
            Question::Noul { .. } => {
                text.push_str("<slot:0> yes\n<slot:1> no\n");
            }
        }
        let tokens = tokenizer.encode(&text, true)?;
        let mut candidates = Vec::new();
        let slot_marker = |i: usize| tokenizer.id_for(&format!("<slot:{i}>")).unwrap_or(0);
        match question {
            Question::Choice { criteria, .. } => {
                for (i, (label, desc)) in criteria.iter().enumerate() {
                    let m = slot_marker(i);
                    let position = tokens.iter().position(|&t| t == m).unwrap_or(0);
                    candidates.push(Candidate {
                        kind: CandidateKind::Option,
                        position,
                        code_id: m,
                        label: label.clone(),
                        description: {
                            let d = criteria_text(desc);
                            if d.is_empty() {
                                None
                            } else {
                                Some(d)
                            }
                        },
                        index: i,
                    });
                }
            }
            Question::Score { criteria, .. } => {
                for (i, level) in criteria.iter().enumerate() {
                    let m = slot_marker(i);
                    let position = tokens.iter().position(|&t| t == m).unwrap_or(0);
                    candidates.push(Candidate {
                        kind: CandidateKind::Level,
                        position,
                        code_id: m,
                        label: i.to_string(),
                        description: Some(level_text(level)),
                        index: i,
                    });
                }
            }
            Question::Noul { .. } => {
                let yes = slot_marker(0);
                let no = slot_marker(1);
                let yes_pos = tokens.iter().position(|&t| t == yes).unwrap_or(0);
                let no_pos = tokens.iter().position(|&t| t == no).unwrap_or(0);
                candidates = vec![
                    Candidate {
                        kind: CandidateKind::YesNo,
                        position: yes_pos,
                        code_id: yes,
                        label: "yes".into(),
                        description: None,
                        index: 0,
                    },
                    Candidate {
                        kind: CandidateKind::YesNo,
                        position: no_pos,
                        code_id: no,
                        label: "no".into(),
                        description: None,
                        index: 1,
                    },
                ];
            }
        }
        let prefix_len = tokens.len();
        Ok(BuiltPrompt {
            tokens,
            candidates,
            prefix_len,
            qtype: question.qtype_index(),
        })
    }
}

/// Render one criterion value to text the way Laya's `render_criterion` does:
/// strings pass through, null becomes `null`, and structured values are JSON.
fn laya_render_criterion(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".into(),
        other => other.to_string(),
    }
}

/// True when a criterion value counts as "no description" in Laya (None or "").
fn laya_is_empty_criterion(v: &Option<serde_json::Value>) -> bool {
    match v {
        None => true,
        Some(serde_json::Value::String(s)) => s.is_empty(),
        Some(serde_json::Value::Null) => true,
        Some(_) => false,
    }
}

/// Render the candidate option texts in Laya order (dict insertion order).
/// Note: huncho's `Choice.criteria` is a `BTreeMap`, so choice options are
/// rendered alphabetically rather than in the exact insertion order Laya preserves;
/// this is a documented limitation (the model's top choice is normally unaffected).
fn laya_option_texts(question: &Question) -> Vec<String> {
    match question {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(label, desc)| {
                if laya_is_empty_criterion(desc) {
                    label.clone()
                } else {
                    format!("{label}: {}", laya_render_criterion(desc.as_ref().unwrap()))
                }
            })
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {i}: {}", laya_render_criterion(c)))
            .collect(),
        Question::Noul { criteria, .. } => {
            let no_desc = criteria.as_ref().and_then(|c| c.no.as_ref());
            let yes_desc = criteria.as_ref().and_then(|c| c.yes.as_ref());
            let no_text = match no_desc {
                Some(v) if !(v.is_null() || v.as_str().is_some_and(|s| s.is_empty())) => {
                    laya_render_criterion(v)
                }
                _ => "no, the statement does not hold".into(),
            };
            let yes_text = match yes_desc {
                Some(v) if !(v.is_null() || v.as_str().is_some_and(|s| s.is_empty())) => {
                    laya_render_criterion(v)
                }
                _ => "yes, the statement holds".into(),
            };
            vec![
                format!("false: {no_text}"),
                format!("true: {yes_text}"),
            ]
        }
    }
}

/// The Laya `[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`
/// formatter.
struct LayaFormatter {
    /// Total per-question sequence cap (`max_len`, default 512).
    max_len: usize,
    /// Head region budget (`head_max_len`, default 192).
    head_max_len: usize,
}

impl PromptFormatter for LayaFormatter {
    fn family(&self) -> Family {
        Family::F1
    }

    fn build(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt> {
        let mask_id = tokenizer.mask_token_id().ok_or_else(|| {
            Error::Package("Laya formatter requires a tokenizer with a [MASK] token".into())
        })?;
        let cls_id = tokenizer.cls_token_id().ok_or_else(|| {
            Error::Package("Laya formatter requires a tokenizer with a [CLS] token".into())
        })?;
        let sep_id = tokenizer.sep_token_id().ok_or_else(|| {
            Error::Package("Laya formatter requires a tokenizer with a [SEP] token".into())
        })?;

        let ins = instructions_text(question).replace(LAYA_MASK_STR, " ");
        let head_text = format!("{} question: {}", question.type_name(), ins);
        let mut head_ids = tokenizer.encode(&head_text, false)?;

        let opts = laya_option_texts(question);
        let n = opts.len();
        let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut total_opt = 0usize;
        for opt in &opts {
            let mut ids = tokenizer.encode(&format!(" {opt}"), false)?;
            ids.truncate(LAYA_OPTION_MAX_TOKENS);
            total_opt += ids.len();
            opt_ids.push(ids);
        }

        let mut opt_budget = self.head_max_len.saturating_sub(total_opt);
        if opt_budget < 16 {
            let per = ((self.head_max_len.saturating_sub(16)) / n.max(1)).max(4);
            for ids in &mut opt_ids {
                ids.truncate(per);
            }
            let total: usize = opt_ids.iter().map(|v| v.len()).sum();
            opt_budget = self.head_max_len.saturating_sub(total);
        }
        head_ids.truncate(8.max(opt_budget));

        let mut ids = Vec::with_capacity(self.max_len);
        ids.push(cls_id);
        ids.extend(head_ids);
        ids.push(sep_id);

        let mut markers = Vec::with_capacity(n);
        for o in &opt_ids {
            markers.push(ids.len());
            ids.push(mask_id);
            ids.extend(o);
        }
        ids.push(sep_id);

        let room = self.max_len.saturating_sub(ids.len()).saturating_sub(1);
        let state_ids = tokenizer.encode(
            &state_text(state).replace(LAYA_MASK_STR, " "),
            false,
        )?;
        ids.extend(state_ids.iter().take(room));
        ids.push(sep_id);
        ids.truncate(self.max_len);
        markers.retain(|&m| m < self.max_len);

        let candidates = laya_candidates(question, &markers);
        let prefix_len = markers.first().copied().unwrap_or(ids.len());
        Ok(BuiltPrompt {
            tokens: ids,
            candidates,
            prefix_len,
            qtype: question.qtype_index(),
        })
    }
}

/// Build candidates for each Laya marker position, mapping to the answer
/// semantics huncho's `build_answer` expects.
fn laya_candidates(question: &Question, markers: &[usize]) -> Vec<Candidate> {
    match question {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, (label, desc))| Candidate {
                kind: CandidateKind::Option,
                position: markers.get(i).copied().unwrap_or(0),
                code_id: 0,
                label: label.clone(),
                description: {
                    let d = laya_criterion_opt(desc);
                    if d.is_empty() {
                        None
                    } else {
                        Some(d)
                    }
                },
                index: i,
            })
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, level)| Candidate {
                kind: CandidateKind::Level,
                position: markers.get(i).copied().unwrap_or(0),
                code_id: 0,
                label: i.to_string(),
                description: Some(laya_render_criterion(level)),
                index: i,
            })
            .collect(),
        Question::Noul { .. } => vec![
            // stanza[0] = false (label "no"), stanza[1] = true (label "yes").
            Candidate {
                kind: CandidateKind::YesNo,
                position: markers.first().copied().unwrap_or(0),
                code_id: 0,
                label: "no".into(),
                description: None,
                index: 0,
            },
            Candidate {
                kind: CandidateKind::YesNo,
                position: markers.get(1).copied().unwrap_or(0),
                code_id: 0,
                label: "yes".into(),
                description: None,
                index: 1,
            },
        ],
    }
}

fn laya_criterion_opt(v: &Option<serde_json::Value>) -> String {
    match v {
        Some(val) => laya_render_criterion(val),
        None => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Nimble (F3 candidate-logit) formatter
// ---------------------------------------------------------------------------

/// Serialize a JSON value the way the Nimble reference `safe_json` does:
/// compact JSON with `<` and `>` escaped as `\u003c`/`\u003e` so context can
/// never inject markup ambiguity into the schema.
fn nimble_safe_json(value: &serde_json::Value) -> String {
    let s = serde_json::to_string(value).unwrap_or_default();
    s.replace('<', "\\u003c").replace('>', "\\u003e")
}

/// The single field name + description a huncho [`Question`] maps to in the
/// Nimble schema. Since huncho asks one decision at a time, the schema has a
/// single field and we request it.
fn nimble_field(q: &Question) -> (String, String) {
    (q.type_name().to_string(), instructions_text(q))
}

/// The schema choices for a question as `(value, label, description)` triples.
///
/// * `value` is the JSON value written into the schema (`bool` for Noul).
/// * `label` is the huncho answer label (`"yes"`/`"no"` for Noul).
/// * `description` is the per-choice description, when present.
fn nimble_choice_specs(q: &Question) -> Vec<(serde_json::Value, String, Option<String>)> {
    match q {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(label, desc)| {
                let d = criteria_text(desc);
                (
                    serde_json::Value::String(label.clone()),
                    label.clone(),
                    if d.is_empty() { None } else { Some(d) },
                )
            })
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, level)| {
                (
                    serde_json::Value::String(i.to_string()),
                    i.to_string(),
                    Some(level_text(level)),
                )
            })
            .collect(),
        Question::Noul { criteria, .. } => {
            let no_desc = criteria
                .as_ref()
                .and_then(|c| c.no.as_ref())
                .map(|v| criteria_text(&Some(v.clone())))
                .filter(|s| !s.is_empty());
            let yes_desc = criteria
                .as_ref()
                .and_then(|c| c.yes.as_ref())
                .map(|v| criteria_text(&Some(v.clone())))
                .filter(|s| !s.is_empty());
            vec![
                (serde_json::Value::Bool(false), "no".to_string(), no_desc),
                (serde_json::Value::Bool(true), "yes".to_string(), yes_desc),
            ]
        }
    }
}

/// The `nimble-v1` F3 candidate-logit formatter.
///
/// Builds the classified-fields prompt used by `Bespoke-Nimble` adapters: a
/// single-field schema rendered as `{"context": ..., "schema": [...]}` plus a
/// `Requested field:` marker. Candidates are one-token answer codes scored
/// through the LM head; logits are read at the final position.
///
/// Note: the upstream prompt is produced by the Qwen chat template
/// (`apply_chat_template(..., enable_thinking=False)`). This formatter emits the
/// same schema/field structure but plain text; byte-identical prompts require a
/// chat-template-aware tokenizer (a later stage).
struct NimbleFormatter {
    config: F3Config,
}

impl PromptFormatter for NimbleFormatter {
    fn family(&self) -> Family {
        Family::F3
    }

    fn build(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt> {
        let context = state_text(state);
        let (field_name, description) = nimble_field(question);
        let specs = nimble_choice_specs(question);
        if specs.is_empty() {
            return Err(Error::Request("question has no choices".into()));
        }
        let codes = &self.config.candidate_codes;
        let ids = &self.config.candidate_token_ids;
        if specs.len() > codes.len() || specs.len() > ids.len() {
            return Err(Error::Request(format!(
                "question has {} choices but the candidate codebook declares only {} codes",
                specs.len(),
                codes.len()
            )));
        }

        let schema_choices: Vec<serde_json::Value> = specs
            .iter()
            .enumerate()
            .map(|(i, (value, _label, desc))| {
                let mut obj = serde_json::Map::new();
                obj.insert("code".into(), serde_json::Value::String(codes[i].clone()));
                obj.insert("value".into(), value.clone());
                if let Some(d) = desc {
                    obj.insert("description".into(), serde_json::Value::String(d.clone()));
                }
                serde_json::Value::Object(obj)
            })
            .collect();
        let field = serde_json::json!({
            "name": field_name,
            "description": description,
            "choices": schema_choices,
        });
        let content = format!(
            "{}\n\nRequested field: {}",
            nimble_safe_json(&serde_json::json!({"context": context, "schema": [field]})),
            field_name
        );
        let system = &self.config.system_prompt;
        let messages_text = if system.is_empty() {
            content.clone()
        } else {
            format!("{system}\n\n{content}")
        };

        // Encode without extra special tokens so the final token is the answer
        // generation point whose logits are scored for the one-token codes.
        let tokens = tokenizer.encode(&messages_text, false)?;
        let position = tokens.len().saturating_sub(1);

        let candidates = specs
            .iter()
            .enumerate()
            .map(|(i, (_value, label, desc))| Candidate {
                kind: match question {
                    Question::Choice { .. } => CandidateKind::Option,
                    Question::Score { .. } => CandidateKind::Level,
                    Question::Noul { .. } => CandidateKind::YesNo,
                },
                position,
                code_id: ids[i],
                label: label.clone(),
                description: desc.clone(),
                index: i,
            })
            .collect();

        Ok(BuiltPrompt {
            tokens,
            candidates,
            prefix_len: position + 1,
            qtype: question.qtype_index(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Question, StateValue};

    fn test_state() -> StateValue {
        StateValue::from(serde_json::json!({ "ticket": "customer wants a refund" }))
    }
    use crate::tokenizer::SimpleTokenizer;
    use crate::manifest::Family;

    fn q_choice() -> Question {
        let json = serde_json::json!({
            "type": "choice",
            "instructions": "Which team?",
            "criteria": { "returns": "money back", "billing": "charge issue" }
        });
        serde_json::from_value(json).unwrap()
    }

    fn q_noul() -> Question {
        let json = serde_json::json!({ "type": "noul", "instructions": "refund?" });
        serde_json::from_value(json).unwrap()
    }

    fn q_score() -> Question {
        let json = serde_json::json!({
            "type": "score", "instructions": "severity?", "criteria": ["low","mid","high"]
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn f1_builds_choice() {
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F1);
        let built = f.build(&test_state(), &q_choice(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 2);
        // `criteria` preserves insertion order, so labels appear as supplied.
        assert_eq!(built.candidates[0].label, "returns");
        assert_eq!(built.candidates[1].label, "billing");
        // positions differ
        assert_ne!(built.candidates[0].position, built.candidates[1].position);
    }

    #[test]
    fn f3_builds_noul_same_position() {
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F3);
        let built = f.build(&test_state(), &q_noul(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 2);
        assert_eq!(built.candidates[0].label, "yes");
        // Both candidates share one answer position (candidate-logit).
        assert_eq!(built.candidates[0].position, built.candidates[1].position);
    }

    #[test]
    fn f1_builds_score_legend() {
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F1);
        let built = f.build(&test_state(), &q_score(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 3);
        assert_eq!(built.candidates[0].label, "0");
        assert!(built.candidates[0].description.is_some());
    }

    #[test]
    fn f1_prefix_len_is_option_marker_boundary() {
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F1);
        let built = f.build(&test_state(), &q_choice(), &tk).unwrap();
        let first_marker = built
            .tokens
            .iter()
            .position(|&t| t == built.candidates[0].code_id)
            .unwrap();
        assert_eq!(built.prefix_len, first_marker);
        assert!(built.prefix_len <= built.tokens.len());
        // All candidates live at/after the fork boundary.
        for c in &built.candidates {
            assert!(c.position >= built.prefix_len);
        }
    }

    #[test]
    fn f2_builds_choice_at_distinct_positions() {
        // F2 (pointer family, Qwen3 block-causal) uses option-boundary markers;
        // the pointer head reads the hidden state at each marker position.
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F2);
        let built = f.build(&test_state(), &q_choice(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 2);
        assert_eq!(built.candidates[0].label, "returns");
        assert_eq!(built.candidates[1].label, "billing");
        // Each option has its own marker position (pointer head), unlike F3.
        assert_ne!(built.candidates[0].position, built.candidates[1].position);
        // `prefix_len` is the block-causal fork boundary: the first option marker.
        let first_marker = built
            .tokens
            .iter()
            .position(|&t| t == built.candidates[0].code_id)
            .unwrap();
        assert_eq!(built.prefix_len, first_marker);
        assert!(built.prefix_len < built.tokens.len());
        assert!(built.prefix_len > 0); // state is non-empty
    }

    #[test]
    fn f2_builds_noul_yes_no() {
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F2);
        let built = f.build(&test_state(), &q_noul(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 2);
        assert_eq!(built.candidates[0].label, "yes");
        assert_eq!(built.candidates[1].label, "no");
        assert_ne!(built.candidates[0].position, built.candidates[1].position);
    }

    #[test]
    fn f2_builds_score_levels() {
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F2);
        let built = f.build(&test_state(), &q_score(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 3);
        assert_eq!(built.candidates[0].label, "0");
        assert_eq!(built.candidates[2].label, "2");
        for c in &built.candidates {
            assert!(c.description.is_some());
        }
    }

    #[test]
    fn f4_builds_slot_choice() {
        // F4 (OpenThai) uses `<slot:n>` markers; candidates are emitted in
        // the supplied (insertion) order and each has its own slot.
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F4);
        let built = f.build(&test_state(), &q_choice(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 2);
        assert_eq!(built.candidates[0].label, "returns");
        assert_eq!(built.candidates[1].label, "billing");
        assert_ne!(built.candidates[0].position, built.candidates[1].position);
        // F4 is a single forward pass, so the whole prompt is the "prefix".
        assert_eq!(built.prefix_len, built.tokens.len());
    }

    #[test]
    fn f4_builds_slot_noul() {
        let tk = SimpleTokenizer::new(32768);
        let f = default_formatter(Family::F4);
        let built = f.build(&test_state(), &q_noul(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 2);
        assert_eq!(built.candidates[0].label, "yes");
        assert_eq!(built.candidates[1].label, "no");
    }

    // --- Nimble (F3 candidate-logit) formatter ---

    fn nimble_config() -> F3Config {
        F3Config {
            candidate_codes: vec!["A".into(), "B".into(), "C".into()],
            candidate_token_ids: vec![32, 33, 34],
            system_prompt: "Classify the context using the supplied schema.".into(),
            prompt_code_sha256: "deadbeef".into(),
            max_input_tokens: 8192,
        }
    }

    fn nimble_formatter() -> NimbleFormatter {
        NimbleFormatter {
            config: nimble_config(),
        }
    }

    #[test]
    fn nimble_builds_choice_all_at_last_position() {
        let tk = SimpleTokenizer::new(32768);
        let f = nimble_formatter();
        let built = f.build(&test_state(), &q_choice(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 2);
        assert_eq!(built.candidates[0].label, "returns");
        assert_eq!(built.candidates[1].label, "billing");
        // Candidate-logit: all codes share the final generation position.
        assert_eq!(built.candidates[0].position, built.candidates[1].position);
        assert_eq!(built.candidates[0].position, built.tokens.len() - 1);
        // code ids come from the codebook.
        assert_eq!(built.candidates[0].code_id, 32);
        assert_eq!(built.candidates[1].code_id, 33);
    }

    #[test]
    fn nimble_builds_noul_yes_no() {
        let tk = SimpleTokenizer::new(32768);
        let f = nimble_formatter();
        let built = f.build(&test_state(), &q_noul(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 2);
        // A -> no (false), B -> yes (true).
        assert_eq!(built.candidates[0].label, "no");
        assert_eq!(built.candidates[1].label, "yes");
        assert_eq!(built.candidates[0].code_id, 32);
        assert_eq!(built.candidates[1].code_id, 33);
        assert_eq!(built.candidates[0].position, built.candidates[1].position);
    }

    #[test]
    fn nimble_builds_score_levels_with_legend() {
        let tk = SimpleTokenizer::new(32768);
        let f = nimble_formatter();
        let built = f.build(&test_state(), &q_score(), &tk).unwrap();
        assert_eq!(built.candidates.len(), 3);
        assert_eq!(built.candidates[0].label, "0");
        assert_eq!(built.candidates[2].label, "2");
        assert!(built.candidates[0].description.is_some());
        for c in &built.candidates {
            assert_eq!(c.position, built.tokens.len() - 1);
        }
    }
}
