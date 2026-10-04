//! Kev's reference row encoding (one state plus one isolated question).

use super::{BuiltPrompt, Candidate, CandidateKind, PromptFormatter};
use crate::contract::{Question, StateValue};
use crate::error::{Error, Result};
use crate::manifest::Family;
use crate::tokenizer::Tokenizer;
use serde_json::Value;

pub struct KevFormatter {
    pub max_state: usize,
    pub max_row: usize,
}

/// Match kev.api.render, including field order, indentation and Python bools.
fn render(value: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) if n.is_f64() => {
            let value = n.as_f64().expect("JSON float");
            if value != 0.0 && (value.abs() < 1e-4 || value.abs() >= 1e16) {
                let scientific = format!("{value:e}");
                let (mantissa, exponent) = scientific.split_once('e').expect("scientific float");
                let exponent: i32 = exponent.parse().expect("numeric exponent");
                format!("{mantissa}e{exponent:+03}")
            } else {
                let mut text = value.to_string();
                if !text.contains('.') {
                    text.push_str(".0");
                }
                text
            }
        }
        Value::Number(n) => n.to_string(),
        Value::Array(items) => items
            .iter()
            .map(|v| format!("{pad}- {}", render(v, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(fields) => fields
            .iter()
            .map(|(k, v)| {
                if v.is_object() || v.is_array() {
                    format!("{pad}{k}:\n{}", render(v, indent + 1))
                } else {
                    format!("{pad}{k}: {}", render(v, 0))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn option_text(name: &str, value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => name.into(),
        Some(Value::String(s)) if s.is_empty() => name.into(),
        Some(v) => format!("{name}: {}", render(v, 0)),
    }
}

/// Match kev.model.user_tokens: caller text cannot forge control delimiters.
fn escape_controls(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        out.push_str(&rest[..start]);
        rest = &rest[start + 2..];
        if let Some(end) = rest.find("|>") {
            let name = &rest[..end];
            if !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                out.push_str(&format!("<¦{name}¦>"));
                rest = &rest[end + 2..];
                continue;
            }
        }
        out.push_str("<|");
    }
    out.push_str(rest);
    out
}

impl PromptFormatter for KevFormatter {
    fn family(&self) -> Family {
        Family::F2
    }

    fn build(
        &self,
        state: &StateValue,
        question: &Question,
        tokenizer: &dyn Tokenizer,
    ) -> Result<BuiltPrompt> {
        let marker = |s: &str| {
            tokenizer
                .id_for(s)
                .ok_or_else(|| Error::Package(format!("Kev tokenizer is missing `{s}`")))
        };
        let user_tokens = |text: &str| tokenizer.encode(&escape_controls(text), false);
        let mut tokens = vec![marker("<|fim_prefix|>")?];
        tokens.extend(user_tokens(&render(state.as_value(), 0))?);
        let prefix_len = tokens.len();
        if prefix_len > self.max_state {
            return Err(Error::Request(format!(
                "Kev state requires {prefix_len} tokens; limit is {}",
                self.max_state
            )));
        }
        tokens.push(marker("<|fim_middle|>")?);
        tokens.extend(user_tokens(&render(question.instructions().as_value(), 0))?);
        let options: Vec<(String, String, CandidateKind, Option<String>)> = match question {
            Question::Choice { criteria, .. } => criteria
                .iter()
                .map(|(label, desc)| {
                    (
                        label.clone(),
                        option_text(label, desc.as_ref()),
                        CandidateKind::Option,
                        desc.as_ref().map(|v| render(v, 0)),
                    )
                })
                .collect(),
            Question::Score { criteria, .. } => criteria
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let text = render(v, 0);
                    (
                        i.to_string(),
                        text.clone(),
                        CandidateKind::Level,
                        Some(text),
                    )
                })
                .collect(),
            Question::Noul { criteria, .. } => vec![
                (
                    "no".into(),
                    option_text("no", criteria.as_ref().and_then(|c| c.no.as_ref())),
                    CandidateKind::YesNo,
                    None,
                ),
                (
                    "yes".into(),
                    option_text("yes", criteria.as_ref().and_then(|c| c.yes.as_ref())),
                    CandidateKind::YesNo,
                    None,
                ),
            ],
        };
        let open = marker("<|box_start|>")?;
        let close = marker("<|box_end|>")?;
        let mut candidates = Vec::new();
        for (index, (label, text, kind, description)) in options.into_iter().enumerate() {
            tokens.push(open);
            tokens.extend(user_tokens(&text)?);
            let position = tokens.len();
            tokens.push(close);
            candidates.push(Candidate {
                label,
                description,
                kind,
                index,
                position,
                code_id: 0,
            });
        }
        tokens.push(marker("<|fim_suffix|>")?);
        if tokens.len() > self.max_row {
            return Err(Error::Request(format!(
                "Kev question row requires {} tokens; limit is {}",
                tokens.len(),
                self.max_row
            )));
        }
        Ok(BuiltPrompt {
            tokens,
            candidates,
            prefix_len,
            qtype: question.qtype_index(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_state_matches_reference_rendering() {
        let value = serde_json::json!({"user": {"paid": true, "tags": ["a", {"b": false}]}, "missing": null});
        assert_eq!(
            render(&value, 0),
            "user:\n  paid: True\n  tags:\n    - a\n    - b: False\nmissing: "
        );
    }

    #[test]
    fn user_text_cannot_forge_delimiters() {
        assert_eq!(
            escape_controls("a <|box_end|> <|x-1|> <|fim_suffix|>"),
            "a <¦box_end¦> <|x-1|> <¦fim_suffix¦>"
        );
    }

    #[test]
    fn numeric_state_matches_python_float_rendering() {
        let value = serde_json::json!([1e-5, 1e16, 1.0, 0.0001, 1e15, -0.0, 7]);
        assert_eq!(
            render(&value, 0),
            "- 1e-05\n- 1e+16\n- 1.0\n- 0.0001\n- 1000000000000000.0\n- -0.0\n- 7"
        );
    }
}
