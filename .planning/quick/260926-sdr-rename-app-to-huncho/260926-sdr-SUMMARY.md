# Quick task complete: Rename app to Huncho

Date: 2026-09-26

Renamed the executable to `huncho`, all four workspace crates and their
directories to `huncho-*`, and the example/generated manifest to
`huncho-model.json`. Updated imports, logs, converter `HUNCHO_*` variables,
the `x-huncho-extensions` header, and Prometheus `huncho_` metric names.
Updated the README, documentation, PRD filename, and project planning records.
The Jev `/v1/systemone` contract and System One model terminology are preserved.

Verification:

- `cargo test --offline --workspace --target-dir /tmp/huncho-rename-target`:
  all 35 existing tests passed.
- CLI version and help for all five subcommands passed.
- CLI conformance passed all five golden cases with both the built-in model
  and the renamed example manifest.
- Conversion smoke check verified runner environment variables, generated
  manifest, artifact path, and README instructions.
- All 38 source/example files match their prior contents apart from naming.

Build artifacts were written outside the tracked `target/` directory.
Changes are left in the working tree; no commit was created.
