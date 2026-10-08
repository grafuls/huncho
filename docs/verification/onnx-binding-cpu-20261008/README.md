# Actual CPU ONNX output allocation reuse — 2026-10-08

The previous bounded-output implementation retained one tensor but bound a
clone on every forward. In pinned `ort` 2.0.0-rc.13, tensor cloning allocates and
copies through an identity session. Its reuse counter did not establish reuse
of the original output allocation.

The native address regression failed before the fix: ORT's output address
differed from the simultaneously live original allocation. After retaining
the owning I/O binding, the exact original address is reused with changed
token inputs and returned core tensors retain their earlier owned data. A
second native check confirms that both successful inference and a nonfinite
output failure clear bound request inputs: ORT refuses a run without newly
bound input data, then accepts the next ordinary call. Output shape/budget
replacement and empty/oversized behavior retain their original tests.

Sequential CPU-only checks with ONNX Runtime 1.28.0:

| Check | Result | Evidence |
|---|---|---|
| Reproduce original allocation bug | Expected failure | [red.log](red.log) |
| Actual address/owned-output and success/error input cleanup checks | 2 passed | [unit.log](unit.log) |
| Full backend suite, `--features onnx-shared` | Passed | [backend.log](backend.log) |
| Actual integrated, padded/batched and shared-session CPU CLI qualification groups | 3 passed | [cli.log](cli.log) |
| Default workspace | Passed | [default-workspace.log](default-workspace.log) |

The backend command was `cargo test -p huncho-backend --features onnx-shared`.
The two new unit checks additionally ran with
`--lib output_binding_tests -- --test-threads=1`.
CLI checks used
`cargo test -p huncho-cli --features onnx-shared,tokenizers,qualification --test onnx_integrated --test onnx_padded --test onnx_shared -- --test-threads=1`.
The explicit CPU ORT library was selected through dynamic linking; no actual
GPU probe or execution ran, and Apple Silicon remains deferred.

Generic, compact, integrated raw-head, masked native batch and independent
shared-session fixtures retain their frozen values and unchanged numerical,
argmax, paired and synthetic labeled gates. No temperature, reference vector
or threshold changes. The binding mutex satisfies ownership/thread traits;
exclusively owned forwards access it without another dynamic lock. Input
clearing happens through a scope guard on all native execution/readout returns.
Default-disabled options and dependency profile are unchanged.

This corrects allocation ownership. It does not qualify a released model or
provide a latency/throughput/RSS result. Synthetic labels remain gate-plumbing
evidence; no new device arithmetic or GPU qualification is claimed.
