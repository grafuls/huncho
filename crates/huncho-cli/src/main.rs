//! `huncho` — the System One decision-model serving engine CLI.

mod bench;
mod calibrate;
mod conform;
mod convert;
mod load;
mod serve;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "huncho",
    version,
    about = "Portable serving engine for System One decision models",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Serve(serve::ServeArgs),
    Convert(convert::ConvertArgs),
    Calibrate(calibrate::CalibrateArgs),
    Conform(conform::ConformArgs),
    Bench(bench::BenchArgs),
    /// Packaging probe: initialize CUDA and run kernels without loading a model.
    #[command(name = "__check-cuda", hide = true)]
    CheckCuda,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Default to `info` when `RUST_LOG` is unset so a `serve` shows its
    // lifecycle (resolving / loading / listening) instead of silently blocking
    // on a first-time model load or download. `RUST_LOG` still overrides.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Serve(a) => serve::run(a).await,
        Command::Convert(a) => convert::run(a),
        Command::Calibrate(a) => calibrate::run(a),
        Command::Conform(a) => conform::run(a),
        Command::Bench(a) => bench::run(a),
        Command::CheckCuda => {
            #[cfg(feature = "candle")]
            if huncho_backend::device::device_from_env()?.is_cuda() {
                return Ok(());
            }
            anyhow::bail!("no usable CUDA device")
        }
    }
}
