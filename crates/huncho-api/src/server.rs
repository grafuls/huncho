//! Ser and startup for the HTTP API.

use std::sync::Arc;

use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

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
    let app = router()
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

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
