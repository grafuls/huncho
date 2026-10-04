//! Text-only port of Cloudflare/Clef's `encode_record` prompt contract.
use crate::contract::{Question, SystemOneRequest};
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const TEMPLATE: &str = "clef-native-v1";
// Version the native encoder independently of executable code in model repos.
pub const CONTRACT: &str = "clef-text-json-sorted-v1";
const PREFIX: &str = "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n";
const SUFFIX: &str =
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct EncodedQuestion {
    pub question_id: String,
    pub question_type: u32,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}

#[derive(Debug)]
pub struct EncodedRecord {
    pub input_ids: Vec<u32>,
    pub questions: Vec<EncodedQuestion>,
}

/// Python json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":")).
/// Python's float notation differs from serde_json at the exponent boundaries.
fn sorted_json(value: &Value) -> String {
    match value {
        Value::Object(fields) => {
            let mut keys: Vec<_> = fields.keys().collect();
            keys.sort();
            format!(
                "{{{}}}",
                keys.into_iter()
                    .map(|k| {
                        format!(
                            "{}:{}",
                            serde_json::to_string(k).unwrap(),
                            sorted_json(&fields[k])
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(sorted_json).collect::<Vec<_>>().join(",")
        ),
        Value::Number(n) if n.is_f64() => {
            let v = n.as_f64().unwrap();
            if v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e16) {
                let s = format!("{v:e}");
                let (m, e) = s.split_once('e').unwrap();
                format!("{m}e{:+03}", e.parse::<i32>().unwrap())
            } else {
                let mut s = v.to_string();
                if !s.contains('.') {
                    s.push_str(".0");
                }
                s
            }
        }
        _ => value.to_string(),
    }
}

fn render(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| sorted_json(value))
}

pub fn encode(
    request: &SystemOneRequest,
    tokenizer: &dyn Tokenizer,
    max_context: usize,
) -> Result<EncodedRecord> {
    request.validate()?;
    if request.questions.len() > 64 {
        return Err(Error::Request(
            "Clef accepts at most 64 questions per request".into(),
        ));
    }
    let tokens = |text: &str| tokenizer.encode(text, false);
    let mut schema = tokens("\n\nSCHEMA FIELDS:\n")?;
    let mut questions = Vec::new();
    for (index, (id, question)) in request.questions.iter().enumerate() {
        schema.extend(tokens(&format!(
            "\nFIELD {}\nID: {id}\nTYPE: {}\nINSTRUCTION: ",
            index + 1,
            question.type_name()
        ))?);
        let start = schema.len();
        let instruction = question.instructions().as_value();
        schema.extend(tokens(
            &if instruction.is_null() || instruction.as_str() == Some("") {
                id.clone()
            } else {
                render(instruction)
            },
        )?);
        let question_span = (start, schema.len());
        schema.extend(tokens("\nALLOWED OPTIONS:\n")?);
        let (question_type, options): (u32, Vec<(String, Value)>) = match question {
            Question::Noul { criteria, .. } => (
                0,
                vec![
                    (
                        "true".into(),
                        criteria
                            .as_ref()
                            .and_then(|c| c.yes.clone())
                            .unwrap_or_else(|| {
                                json!("The proposition is true or the answer is yes.")
                            }),
                    ),
                    (
                        "false".into(),
                        criteria
                            .as_ref()
                            .and_then(|c| c.no.clone())
                            .unwrap_or_else(|| {
                                json!("The proposition is false or the answer is no.")
                            }),
                    ),
                ],
            ),
            Question::Choice { criteria, .. } => {
                let mut options: Vec<_> = criteria
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone().unwrap_or(Value::Null)))
                    .collect();
                options.sort_by(|a, b| a.0.cmp(&b.0));
                (1, options)
            }
            Question::Score { criteria, .. } => (
                2,
                criteria
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.clone()))
                    .collect(),
            ),
        };
        if options.is_empty() {
            return Err(Error::Request(format!(
                "Clef question `{id}` has no options"
            )));
        }
        let mut option_spans = Vec::new();
        let mut option_ids = Vec::new();
        for (index, (id, description)) in options.into_iter().enumerate() {
            schema.extend(tokens(&format!("OPTION {}: ", index + 1))?);
            let start = schema.len();
            let mut semantics = json!({"option_id": id});
            if !description.is_null() {
                semantics["description"] = description;
            }
            schema.extend(tokens(&render(&semantics))?);
            option_spans.push((start, schema.len()));
            option_ids.push(id);
            schema.extend(tokens("\n")?);
        }
        schema.extend(tokens("END FIELD\n")?);
        questions.push(EncodedQuestion {
            question_id: id.clone(),
            question_type,
            question_span,
            option_spans,
            option_ids,
        });
    }
    let mut input_ids = tokens(PREFIX)?;
    input_ids.extend(tokens(&render(request.state.as_value()))?);
    let offset = input_ids.len();
    for q in &mut questions {
        q.question_span.0 += offset;
        q.question_span.1 += offset;
        for span in &mut q.option_spans {
            span.0 += offset;
            span.1 += offset;
        }
        if q.question_span.0 == q.question_span.1 || q.option_spans.iter().any(|(s, e)| s == e) {
            return Err(Error::Request(
                "Clef produced an empty question or option span".into(),
            ));
        }
    }
    input_ids.extend(schema);
    input_ids.extend(tokens(SUFFIX)?);
    if input_ids.len() > max_context {
        return Err(Error::Request(format!(
            "joint prompt requires {} tokens; maximum is {max_context}",
            input_ids.len()
        )));
    }
    Ok(EncodedRecord {
        input_ids,
        questions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_json_numbers_and_sorted_nested_keys() {
        let v: Value =
            serde_json::from_str(r#"{"z":[1e-7,-0.0,1e16,0.0001],"a":{"é":true,"b":null}}"#)
                .unwrap();
        assert_eq!(
            render(&v),
            r#"{"a":{"b":null,"é":true},"z":[1e-07,-0.0,1e+16,0.0001]}"#
        );
    }
}
