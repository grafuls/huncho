# Cooperative CPU Kev native question groups, 2026-10-08

No Apple or actual GPU checks run. Default Cargo dependencies/features,
original backbone/head weights, trained fixture probabilities, temperatures
and external/paired thresholds are unchanged. Synthetic targets and copied
equal-row vectors test qualification plumbing only, not released calibration.

After chunked shared-state prefill, each resumable step submits one bounded
equal/padded CPU Kev suffix group. Core jobs/inputs remain bound to original
typed questions through sorting and relative marker positions; complete
prefix KV/GDN/conv state stays immutable. The backend's live child-row limit
subtracts every active/partial parent. Repartition occurs under the same lock
as native submission and retains suffix padding/complete-context token budgets.
The 64-handle cap is unchanged; pressure may cause scalar-sized groups.

CLI/core/HTTP guards support this opt-in composition with one context and at
most 62 queued HTTP requests. Cross-request prefix collation remains rejected.
Fresh complete labeled qualification requires actual chunks/interleaving,
native cached groups and configured mixed padding, plus unchanged external
delta/full argmax/ECE drift and paired <=1e-4/full argmax. Batch qualification
uses cohorts of at most 32 active parents so native work is not made vacuous
by the scalar 63-parent admission pattern; scalar qualification stays at 63.

- [Actual CPU native](native.log): original typed frozen cases in FP32/FP16,
  CPU kernels/query blocks/grouped GQA/pages, direct FP32 pages and standard
  runtime LoRA pass fresh complete labeled/paired gates. Equal rows copy
  original questions/vectors/labels; mixed profiles keep original requests.
  Tiny budgets reject actual native qualification and revoke old proofs.
- Live 0/61/62 competing partial parents force unchanged, smaller and scalar
  groups without probability drift. Sixty-seven repeated partial cursor drops
  per pressure level reclaim this request's handles and leave other jobs alive.
  Sixty-five unchanged copied cases exercise multiple qualification cohorts
  with actual native groups. Existing chunk/retention/replica guards pass.
- [HTTP](api.log): bounded admission, prefix interleaving, native physical
  counters, coalesced leader cancellation and group failure/recovery release
  all parents. These are explicit scheduler fixtures, not native arithmetic.
- [CLI](cli.log): previous eight native CPU profile groups plus mixed
  cooperative native conformance; missing chunks still refuse startup.
- [Actual native HTTP](http-native.log): the real CPU Kev CLI process with
  runtime LoRA/query blocks/grouped GQA/direct pages starts only after fresh
  qualification, serves original typed probabilities within unchanged
  thresholds, returns zero output tokens and reports real cached/padded group
  counters. The test tears down its localhost server on success or failure.
- [Workspace](workspace.log) and [browser](browser.log): shared core/proof/
  scheduling regressions and separate CPU WASM compatibility.

Commands run sequentially to process exit. Release FP32 remains rejected and
Q8/Q4 outcome qualification pending; fixture success does not promote these
variants or establish speed/peak RSS. Every changed model/profile still needs
its own fresh actual labeled/paired gates. Direct multi-row page kernels,
cross-request/mixed-prefix collation and device scheduling remain open.
