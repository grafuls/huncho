//! Exact keys for engine-local caches of immutable model inputs.

use crate::contract::{Question, StateValue, SystemOneRequest};
use crate::engine::EvalOptions;
use crate::error::Result;

fn description_presence(question: &Question) -> Vec<bool> {
    // serde maps both None and Some(Value::Null) to null. Library callers can
    // construct either, and formatters can distinguish them. Retain the Option
    // discriminants without changing wire serialization or prompt semantics.
    match question {
        Question::Choice { criteria, .. } => criteria.values().map(Option::is_some).collect(),
        Question::Noul { criteria, .. } => {
            let mut presence = vec![criteria.is_some()];
            if let Some(criteria) = criteria {
                presence.extend([criteria.yes.is_some(), criteria.no.is_some()]);
            }
            presence
        }
        Question::Score { .. } => Vec::new(),
    }
}

pub(crate) fn request(req: &SystemOneRequest, opts: &EvalOptions) -> Result<Vec<u8>> {
    let presence: Vec<_> = req.questions.values().map(description_presence).collect();
    Ok(serde_json::to_vec(&(req, opts, presence))?)
}

pub(crate) fn prompt(state: &StateValue, question: &Question) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(
        state,
        question,
        description_presence(question),
    ))?)
}
