//! Ser and startup for the HTTP API.

use std::sync::Arc;

use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use huncho_core::engine::QualificationStatus;

use crate::routes::router;
use crate::state::AppState;

struct EvictionTask(Option<tokio::task::JoinHandle<()>>);
impl Drop for EvictionTask {
    fn drop(&mut self) {
        if let Some(task) = &self.0 {
            task.abort();
        }
    }
}

/// Run the HTTP server until it is asked to stop.
pub async fn serve(state: Arc<AppState>) -> std::io::Result<()> {
    // Startup admission. Programmatic API users get the same qualification
    // boundary as the CLI, with the same strict/trial split: strict mode
    // (`require_qualified`) refuses to open a listener for unqualified or
    // numerical-only models; trial mode (default) logs their honest status
    // and serves. Pending/unfitted calibration refuses in every mode.
    // Lazy factories are checked again before their first request.
    let strict = state.config.require_qualified;
    if strict {
        tracing::info!("serving mode: strict (every model must hold fresh labeled qualification)");
    } else {
        tracing::info!("serving mode: trial (unqualified models serve with an honest status; pass --require-qualified for production strictness)");
    }
    for (name, model) in state.registry.read().await.models() {
        let status = model.qualification_status(&state.config);
        match status {
            QualificationStatus::Qualified => {
                tracing::info!(model = name.as_str(), "qualification: qualified");
            }
            QualificationStatus::Numerical => {
                if strict {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "strict serving requires observed target labels for `{name}`: \
                             it holds numerical qualification only (probability/argmax gates passed, \
                             ECE/Brier outcome gates have not run)"
                        ),
                    ));
                }
                tracing::warn!(
                    model = name.as_str(),
                    "qualification: numerical (probability/argmax gates passed; outcome labels not yet collected)"
                );
            }
            QualificationStatus::Unfitted => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "model `{name}` has pending or unfitted calibration and cannot serve \
                         in any mode; fit calibration or restore a fitted variant"
                    ),
                ));
            }
            QualificationStatus::Unqualified => {
                if strict {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "strict serving of a native, refitted or optimized variant of `{name}` \
                             requires fresh labeled qualification: pass \
                             --qualification-golden {name}=/path/to/pinned-golden.json"
                        ),
                    ));
                }
                tracing::warn!(
                    model = name.as_str(),
                    "qualification: UNQUALIFIED — `{name}` is serving in trial mode without fresh \
                     labeled qualification; probabilities are unverified. Qualify it with \
                     `huncho qualify --model {name}` before trusting its numbers"
                );
            }
        }
    }
    let app = router(state.clone())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let bind = state.config.bind.clone();
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    let eviction = EvictionTask({
        let idle = state
            .registry
            .read()
            .await
            .residency
            .as_ref()
            .map(|residency| residency.idle);
        idle.map(|idle| {
            let owner = Arc::downgrade(&state);
            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(idle.min(std::time::Duration::from_secs(60)));
                loop {
                    interval.tick().await;
                    let Some(state) = owner.upgrade() else { break };
                    state.evict_idle_models().await;
                }
            })
        })
    });
    tracing::info!("huncho listening on {bind}");
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    drop(eviction);
    result?;
    tracing::info!("huncho shutdown complete");
    Ok(())
}

/// Wait for SIGINT/SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
