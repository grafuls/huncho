//! Core error types for the `huncho` engine.

use std::fmt;

/// The root error type for the `huncho` core.
///
/// Errors are intentionally flat and carry a human-readable message plus an
/// optional machine-readable `code`. The API layer maps these onto HTTP status
/// codes (see `huncho-api`).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid model package: {0}")]
    Package(String),

    #[error("invalid request: {0}")]
    Request(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("calibration error: {0}")]
    Calibration(String),

    #[error("conformance error: {0}")]
    Conformance(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    /// A short machine-readable code for this error category.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Package(_) => "invalid_package",
            Error::Request(_) => "invalid_request",
            Error::ModelNotFound(_) => "model_not_found",
            Error::Unsupported(_) => "unsupported",
            Error::Backend(_) => "backend_error",
            Error::Calibration(_) => "calibration_error",
            Error::Conformance(_) => "conformance_error",
            Error::Io(_) => "io_error",
            Error::Json(_) => "json_error",
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// A structured error payload suitable for the JSON error body.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

/// The detail portion of an error body.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl From<&Error> for ErrorBody {
    fn from(e: &Error) -> Self {
        ErrorBody {
            error: ErrorDetail {
                code: e.code().to_string(),
                message: e.to_string(),
                details: None,
            },
        }
    }
}

/// A convenience helper that creates a [`fmt`]-style request error.
pub fn request_err(msg: impl fmt::Display) -> Error {
    Error::Request(msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_and_body() {
        let e = Error::Request("bad".into());
        assert_eq!(e.code(), "invalid_request");
        let body: ErrorBody = (&e).into();
        assert_eq!(body.error.code, "invalid_request");
    }
}
