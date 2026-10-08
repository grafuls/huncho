//! Offline fitting-logit collection breaks the circular dependency between a
//! pending variant and HTTP's mandatory fitted/labeled qualification gates.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;

use clap::Args;
use huncho_core::contract::{Question, SystemOneRequest};
use huncho_core::engine::{EvalOptions, EvalStats};
use huncho_core::manifest::Family;
use serde::Deserialize;

#[derive(Args)]
pub struct CaptureArgs {
    /// Local Kev package directory or manifest, never changed by collection.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "candle")]
    backend: String,
    #[arg(long)]
    dtype: Option<String>,
    /// Fitting JSONL: id, request and observed targets keyed by question ID.
    #[arg(long)]
    data: PathBuf,
    /// New output directory. Writes fitting rows and an execution audit.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct Record {
    id: String,
    request: SystemOneRequest,
    targets: BTreeMap<String, String>,
    #[serde(default)]
    source: serde_json::Value,
}

fn labels(question: &Question) -> Vec<String> {
    match question {
        Question::Choice { criteria, .. } => criteria.keys().cloned().collect(),
        Question::Score { criteria, .. } => (0..criteria.len()).map(|i| i.to_string()).collect(),
        // This command explicitly requires the trained kev-v1 contract.
        Question::Noul { .. } => vec!["no".into(), "yes".into()],
    }
}

fn write_new(path: &std::path::Path, value: &impl serde::Serialize) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

pub fn run(args: CaptureArgs) -> anyhow::Result<()> {
    let hash = crate::qualification::hash_file(&args.data)?;
    let records: Vec<Record> = std::fs::read_to_string(&args.data)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(!records.is_empty(), "fitting records must be nonempty");
    let mut ids = BTreeSet::new();
    for record in &records {
        anyhow::ensure!(
            !record.id.is_empty() && ids.insert(&record.id),
            "fitting IDs must be nonempty and unique"
        );
        anyhow::ensure!(
            !record.request.questions.is_empty()
                && record.request.questions.len() == record.targets.len()
                && record
                    .request
                    .questions
                    .keys()
                    .all(|id| record.targets.contains_key(id)),
            "every fitting question needs an observed target"
        );
        for (id, question) in &record.request.questions {
            anyhow::ensure!(
                labels(question).contains(&record.targets[id]),
                "observed target must identify a candidate"
            );
        }
    }
    let path = if args.model.is_dir() {
        args.model.join("huncho-model.json")
    } else {
        args.model.clone()
    };
    let manifest = huncho_core::manifest::ModelManifest::load(&path)?;
    anyhow::ensure!(
        manifest.family == Family::F2 && manifest.prompt_contract.template == "kev-v1",
        "offline capture currently requires Kev F2"
    );
    anyhow::ensure!(
        records.iter().all(|r| r.request.model == manifest.name),
        "fitting request model must match the pinned package name"
    );
    std::fs::create_dir(&args.output)?;
    let mut inputs = None;
    let engine = crate::load::engine_from_resolved_manifest_observed(
        &path,
        crate::load::BackendChoice::parse(&args.backend)?,
        args.dtype.as_deref(),
        |manifest, backend, dtype, dir| {
            inputs = Some(crate::qualification::InputSnapshot::capture(
                &path, manifest, backend, dtype, dir,
            )?);
            Ok(())
        },
    )?
    .with_prompt_cache(0)
    .with_result_cache(0);
    let inputs = inputs.unwrap();
    inputs.recheck()?;
    let options = EvalOptions {
        extensions: true,
        reference_readout: true,
        ..Default::default()
    };
    let identity =
        crate::qualification::ExecutionIdentity::capture(&engine, &inputs, &options, None)?;
    let mut checkpoint = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output.join("fit-logits.jsonl"))?;
    let (mut rows, mut targets, mut qtypes) = (Vec::new(), Vec::new(), Vec::new());
    let mut work = EvalStats::default();
    for record in records {
        let response = engine.eval_with_stats(&record.request, &options, &mut work)?;
        let logits = response
            .extensions
            .and_then(|e| e.raw_logits)
            .ok_or_else(|| anyhow::anyhow!("backend supplied no raw fitting logits"))?;
        for (id, question) in &record.request.questions {
            let labels = labels(question);
            let row = logits
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("missing raw logits for fitting question {id}"))?;
            anyhow::ensure!(
                row.len() == labels.len() && row.iter().all(|x| x.is_finite()),
                "finite candidate-aligned fitting logits are required"
            );
            let target = labels
                .iter()
                .position(|label| label == &record.targets[id])
                .unwrap();
            serde_json::to_writer(
                &mut checkpoint,
                &serde_json::json!({"id":record.id, "question":id, "source":record.source, "labels":labels, "logits":row, "target":target, "qtype":question.type_name()}),
            )?;
            checkpoint.write_all(b"\n")?;
            checkpoint.flush()?;
            rows.push(row.clone());
            targets.push(target);
            qtypes.push(question.type_name().to_owned());
        }
    }
    inputs.recheck()?;
    anyhow::ensure!(
        crate::qualification::ExecutionIdentity::capture(&engine, &inputs, &options, None)?
            == identity,
        "execution identity changed during fitting collection"
    );
    anyhow::ensure!(
        crate::qualification::hash_file(&args.data)? == hash,
        "fitting input bytes changed during collection"
    );
    let fitting = serde_json::json!({"rows": rows, "targets": targets, "qtypes": qtypes});
    write_new(&args.output.join("fit.json"), &fitting)?;
    let audit = serde_json::json!({"schema_version":1, "identity":identity, "fitting_inputs":hash, "work":work, "questions":rows.len(), "qualified":false,
        "limits":["Offline independent fitting logits only; no probabilities or goldens are exported.", "Pending variants are allowed for analysis, without bypassing HTTP serving gates.", "This command neither fits a temperature nor evaluates held-out outcomes or proves dataset independence."]});
    write_new(&args.output.join("identity.json"), &audit)?;
    println!("{}", serde_json::to_string_pretty(&audit)?);
    Ok(())
}
