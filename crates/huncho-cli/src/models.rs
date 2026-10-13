//! `huncho models` — curated model listing for onboarding.
//!
//! A static, offline answer to "what can I actually run?": the models this
//! project has checked against its loaders, whether their packages ship
//! reference goldens, and what serving mode to expect. The full
//! format-compatible inventory (383 repositories) lives in
//! `docs/huggingface-models.md`; this command lists the curated shortlist.

use clap::Args;
use serde::Serialize;

#[derive(Args)]
pub struct ModelsArgs {
    /// Emit the listing as JSON (for scripts).
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Serialize)]
struct CuratedModel {
    /// Reference to pass to `huncho serve --model` / `huncho qualify --model`.
    reference: &'static str,
    family: &'static str,
    runtime: &'static str,
    /// Qualification the package supports out of the box:
    /// - `built-in`: no weights; needs no qualification run.
    /// - `golden`: package ships a `golden.json` reference suite (numerical
    ///   tier; add observed labels to reach full qualification).
    /// - `trial`: no package golden; serves in the default trial mode only
    ///   until you supply your own golden.
    package: &'static str,
    notes: &'static str,
}

const CURATED: &[CuratedModel] = &[
    CuratedModel {
        reference: "mock",
        family: "F1",
        runtime: "built-in",
        package: "built-in",
        notes: "No download. `huncho serve --mock` is qualified by construction.",
    },
    CuratedModel {
        reference: "jaredpalmer/kev-0.8b",
        family: "F2",
        runtime: "candle",
        package: "trial",
        notes: "Curated quick start: smallest real model, ~2 GB download, runs \
                on CPU FP32. No package golden; serves unqualified in trial \
                mode until you pin one.",
    },
    CuratedModel {
        reference: "jaredpalmer/kev-4b",
        family: "F2",
        runtime: "candle",
        package: "trial",
        notes: "CPU FP32 needs ~16 GB RAM. No package golden; the retained \
                CPU FP32 held-out profile currently fails the probability gate.",
    },
    CuratedModel {
        reference: "jaredpalmer/kev-9b",
        family: "F2",
        runtime: "candle",
        package: "trial",
        notes: "CPU FP32 needs ~36 GB RAM. No package golden; serves in trial \
                mode until you pin one.",
    },
    CuratedModel {
        reference: "Cloudflare/clef",
        family: "F5",
        runtime: "clef",
        package: "trial",
        notes: "27B: needs ~54 GB+ of memory (GPU-class). Starts with pending \
                calibration: fit it (see docs/clef.md) before serving.",
    },
    CuratedModel {
        reference: "Cloudflare/clef-flash",
        family: "F5",
        runtime: "clef",
        package: "trial",
        notes: "9B Clef variant. Starts with pending calibration: fit it (see \
                docs/clef.md) before serving.",
    },
    CuratedModel {
        reference: "convaiinnovations/laya",
        family: "F1",
        runtime: "candle",
        package: "trial",
        notes: "421M; small and fast on CPU. No package golden; known \
                divergence on structured JSON states (see \
                docs/huggingface-models.md).",
    },
    CuratedModel {
        reference: "bespokelabs/Bespoke-Nimble-9B",
        family: "F3",
        runtime: "candle",
        package: "trial",
        notes: "Format-compatible; no pinned golden in the package. Serves in \
                trial mode until you supply a golden.",
    },
];

pub fn run(args: ModelsArgs) -> anyhow::Result<()> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(&CURATED)?);
        return Ok(());
    }
    println!("Curated models (checked against huncho loaders):");
    println!();
    println!(
        "{:<34} {:<7} {:<9} {:<9} NOTES",
        "MODEL", "FAMILY", "RUNTIME", "PACKAGE"
    );
    for model in CURATED {
        println!(
            "{:<34} {:<7} {:<9} {:<9} {}",
            model.reference, model.family, model.runtime, model.package, model.notes
        );
    }
    println!();
    println!("Package column:");
    println!("  built-in  no weights; qualified by construction");
    println!("  golden    the repo ships a golden.json reference suite (numerical");
    println!("            tier; add observed labels with `huncho qualify` for full qualification)");
    println!("  trial     no package golden; serves in the default trial mode");
    println!();
    println!("Quick start:");
    println!("  huncho serve --model jaredpalmer/kev-0.8b \\");
    println!("    --revision bf75a6a8848ea6960ff2ed108d9ed44c2941174f");
    println!();
    println!("Serving modes: trial (default, honest status per model) or strict");
    println!("(`--require-qualified`). See README.md and docs/operations.md.");
    println!();
    println!("383 more format-compatible repositories: docs/huggingface-models.md");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_table_lists_quick_start_and_documented_repos() {
        assert!(CURATED.iter().any(|model| model.reference == "mock"));
        assert!(CURATED
            .iter()
            .any(|model| model.reference == "jaredpalmer/kev-0.8b" && model.package == "trial"));
        assert!(CURATED
            .iter()
            .any(|model| model.reference == "Cloudflare/clef" && model.family == "F5"));
        assert!(CURATED
            .iter()
            .any(|model| model.reference == "bespokelabs/Bespoke-Nimble-9B"
                && model.package == "trial"));
        // The rendered listing and the JSON form both succeed.
        run(ModelsArgs { json: false }).unwrap();
        run(ModelsArgs { json: true }).unwrap();
    }
}
