# Quick task: Rename app to Huncho

Date: 2026-09-26

The user selected Huncho as the app name. Use `Huncho` in prose, `huncho`
for the executable, `huncho-*` for crates and model packages, and `HUNCHO_*`
for converter environment variables. Preserve System One terminology for
the model class and Jev API contract.

1. Rename workspace crates, Rust imports, CLI identity, generated package
   names, extension header, metrics, and examples consistently.
2. Update documentation and project planning references to the new name.
3. Run the existing workspace tests and smoke-check CLI help, conversion,
   and conformance using a build directory outside the tracked `target/`.

Execute inline in this runtime. Leave the changes in the working tree for
review; Git metadata is read-only in this workspace.
