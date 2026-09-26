//! Ser and startup for the HTTP API.

use std::sync::Arc;

use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::routes::router;
use crate::state::AppState;

/// Run the HTTP server until it is asked to stop.
pub async fn serve(state: Arc<AppState>) -> std::io::Result<()> {
    let app = router()
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind = state.config.bind.clone();
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!("huncho listening on {bind}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
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
