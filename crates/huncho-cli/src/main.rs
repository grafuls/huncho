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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Serve(a) => serve::run(a).await,
        Command::Convert(a) => convert::run(a),
        Command::Calibrate(a) => calibrate::run(a),
        Command::Conform(a) => conform::run(a),
        Command::Bench(a) => bench::run(a),
    }
}
