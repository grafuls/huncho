//! Optional bearer-token authentication (API-03).

use axum::http::{HeaderMap, StatusCode};
use subtle::ConstantTimeEq;

use crate::config::ServerConfig;

/// Check the `Authorization` header against the configured token.
///
/// Returns `Ok(())` when authentication passes (or when no token is
/// configured). Returns `Err(StatusCode::UNAUTHORIZED)` when the header is
/// malformed or the token is wrong.
pub fn check_auth(
    headers: &HeaderMap,
    config: &ServerConfig,
) -> Result<(), StatusCode> {
    let expected = match &config.auth_token {
        Some(t) => t,
        None => return Ok(()),
    };

    let value = headers
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;

    let token = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Constant-time comparison to avoid timing side channels.
    let ok = token.as_bytes().ct_eq(expected.as_bytes()).into();
    if ok {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

/// Whether a request asked for engine extensions (API-05). Off by default.
pub fn wants_extensions(headers: &HeaderMap) -> bool {
    headers
        .get("x-s1-extensions")
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            let v = v.trim().to_lowercase();
            !matches!(v.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(http::header::AUTHORIZATION, HeaderValue::from_str(v).unwrap());
        h
    }

    fn config(token: Option<&str>) -> ServerConfig {
        ServerConfig {
            auth_token: token.map(|t| t.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn no_token_always_passes() {
        let c = config(None);
        assert!(check_auth(&HeaderMap::new(), &c).is_ok());
    }

    #[test]
    fn correct_token_passes() {
        let c = config(Some("secret"));
        assert!(check_auth(&headers("Bearer secret"), &c).is_ok());
    }

    #[test]
    fn wrong_token_fails() {
        let c = config(Some("secret"));
        assert_eq!(check_auth(&headers("Bearer nope"), &c), Err(StatusCode::UNAUTHORIZED));
        assert_eq!(check_auth(&HeaderMap::new(), &c), Err(StatusCode::UNAUTHORIZED));
        assert_eq!(check_auth(&headers("Basic abc"), &c), Err(StatusCode::UNAUTHORIZED));
    }

    #[test]
    fn extensions_header() {
        let mut h = HeaderMap::new();
        h.insert("x-s1-extensions", HeaderValue::from_static("1"));
        assert!(wants_extensions(&h));
        let mut h2 = HeaderMap::new();
        h2.insert("x-s1-extensions", HeaderValue::from_static("0"));
        assert!(!wants_extensions(&h2));
        assert!(!wants_extensions(&HeaderMap::new()));
    }
}
