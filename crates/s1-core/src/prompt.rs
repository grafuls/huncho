//! Per-family prompt / contract building (PRD §6.2 `prompt_contract`).
//!
//! The engine builds a [`BuiltPrompt`] for each question. The exact template is
//! pinned by the manifest's `prompt_contract.template`; `s1 convert` extracts the
//! reference template into the package so prompts stay byte-identical to the
//! reference implementation. This module ships canonical default templates for
//! each family and a [`PromptFormatter`] trait for custom ones.

use crate::contract::{Question, StateValue};
use crate::error::{Error, Result};
use crate::manifest::Family;
use crate::tokenizer::Tokenizer;

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
        // `criteria` is a BTreeMap, so labels are emitted in sorted key order.
        assert_eq!(built.candidates[0].label, "billing");
        assert_eq!(built.candidates[1].label, "returns");
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
}
