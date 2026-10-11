# Hugging Face models for Huncho

Checked on 2026-10-09 against Huncho commit `269d8eb8ced54f4ad407f4c117eb8712f7ac8745`.

The search covered **8,633 public repository metadata records**. Of **597 repositories with recognizable decision-model files**, **383 match the current native loader formats**, **205 have identified incompatibilities**, and **9 could not be verified** without access to their release files. The matches include 10 upstream checkpoints and 373 community repositories. Mirrors and fine-tunes are separate repositories; these counts do not imply that all weights are distinct.

**These are format-compatible candidates, not tested serving approvals.** Compatibility is inferred from pinned release files, configs, tensor headers and head metadata. No full-model inference or new qualification suites were run for this inventory. Packages shipping a reference golden earn the `numerical` qualification tier automatically in the default trial mode; strict serving (`--require-qualified`) additionally demands observed outcomes. Its retained Kev-4B CPU FP32 held-out profile fails the probability-difference gate; do not interpret the Kev entries as an accepted serving profile. See [calibration](calibration.md), [native backend limits](backends.md), and [the retained Kev result](verification/kev-cpu-heldout-20261008/summary.json).

Subsequent [actual CPU tests on R6525](verification/user-qa-r6525-20261009/README.md) found that Laya Multilingual fails native tokenizer loading, and Laya/typed Laya differ from upstream on structured JSON states. Those runtime results override the initial metadata-based candidate designation. The inventory counts below describe the original static scan.

## Upstream checkpoints

| Repository | Approximate model size | Huncho runtime | Checked revision |
|---|---:|---|---|
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | 421M | Candle / F1 | [7b928d828b7b](https://huggingface.co/convaiinnovations/laya/tree/7b928d828b7b0e022f929d9bd2e44165aa270148) |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | Candle / F1 | [1720e3e3357c](https://huggingface.co/convaiinnovations/laya-multilingual/tree/1720e3e3357cfe1e281542e223f8273b0890ca34) |
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | Candle / F1 | [e929ae5cf69b](https://huggingface.co/convaiinnovations/laya-typed-decisions/tree/e929ae5cf69bc34259cd2f95c9e91145b818b1f0) |
| [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | 0.8B | Candle / F2 | [bf75a6a8848e](https://huggingface.co/jaredpalmer/kev-0.8b/tree/bf75a6a8848ea6960ff2ed108d9ed44c2941174f) |
| [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | 4B | Candle / F2 | [6cfce5c2fa4b](https://huggingface.co/jaredpalmer/kev-4b/tree/6cfce5c2fa4b4bd64026336ab649c5ca78857d52) |
| [jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b) | 9B | Candle / F2 | [db029f08b290](https://huggingface.co/jaredpalmer/kev-9b/tree/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0) |
| [bespokelabs/Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) | 9B | Candle / F3 | [bd792f44ec8e](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B/tree/bd792f44ec8e265be861bfcdf4e05967ffe0e858) |
| [bespokelabs/Bespoke-Nimble-9B-v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) | 9B | Candle / F3 | [4b8c04d1ac2c](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2/tree/4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe) |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 9B | Clef / F5 | [fde727a28700](https://huggingface.co/Cloudflare/clef-flash/tree/fde727a287004204b7518dcc983fe64379776712) |
| [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) | 27B | Clef / F5 | [ed3eed331870](https://huggingface.co/Cloudflare/clef/tree/ed3eed331870db2eff4b0db01237128ede8a00ce) |

The three Laya releases contain compatible ModernBERT configs and complete trained typed heads. The multilingual checkpoint uses an mmBERT encoder represented by a compatible `modernbert` config. Current release prompt defaults are 512/192 tokens for Laya and 1,024/256 for multilingual and typed-decisions (`max_len`/`head_max_len`); the larger encoder context alone does not increase these prompt defaults. Kev native synthesis caps context at 8,192 tokens. Nimble releases have 8,192-token schema limits. Clef native synthesis caps context at 16,384 tokens, and Huncho accepts text/JSON state only.

## What was checked

- **F1 / Laya:** `encoder/config.json` or root config reports ModernBERT; root dense `model.safetensors`, `rl_agent_config.json` and a native tokenizer exist; tensor headers contain ModernBERT backbone keys and the complete Laya `type_emb`, transformer-head and scorer tensors with matching dimensions. The runtime is Candle.
- **F2 / Kev:** `FEATURE_EXTRACTION` LoRA adapter, native tokenizer and `head.pt` exist; adapter flags match Huncho support; head metadata agrees on base/rank, uses LoRA weights, and does not enable option isolation or special embeddings; the base config, pinned where declared, reports Qwen3.5 text architecture. The runtime is Candle. Metadata was inspected as pickle opcodes, without executing remote pickle callables.
- **F3 / Nimble:** standard LoRA weights, tokenizer, `schema_config.json` and `serving_config.json` exist; schema points to a Qwen3.5 base; candidate code/token-ID lists are nonempty and have matching lengths. The runtime is Candle.
- **F5 / Clef:** root Qwen3.5 text config, compatible joint-head dimensions/tensor names, tokenizer, dense backbone artifacts and all indexed shards exist. The first backbone shard was sampled for dense tensor types. The runtime is Clef.

These checks do not prove numerical fidelity, calibration, adequate host memory, tokenizer/prompt equivalence for every fine-tune, or validity of every tensor in every shard. They identify releases that are plausible inputs to the implemented loaders. Community training changes may need different prompts or further adaptation even when file formats match.

Loader rules come from [Hub synthesis](../crates/huncho-hub/src/resolver.rs), [Clef resolution](../crates/huncho-hub/src/clef.rs), [Laya execution](../crates/huncho-backend/src/candle.rs), [Kev metadata validation](../crates/huncho-backend/src/kev.rs) and [Qwen execution](../crates/huncho-backend/src/qwen3_5.rs).

## Running a candidate

Build the CPU runtimes and qualification support:

```bash
cargo build --release -p huncho-cli --features clef,qualification
```

Choose one checked repository and pin its full revision:

```bash
target/release/huncho serve \
  --model jaredpalmer/kev-0.8b \
  --revision bf75a6a8848ea6960ff2ed108d9ed44c2941174f
```

Huncho automatically discovers and checks the reference golden file from the downloaded package; its unlabeled vectors qualify the `numerical` tier, which the default trial mode serves with an honest status. To reach full qualification (required by `--require-qualified`), collect observed outcomes:

```bash
huncho qualify --model jaredpalmer/kev-0.8b \
  --revision bf75a6a8848ea6960ff2ed108d9ed44c2941174f \
  --template labeled-golden.json
# fill in each case's targets, then:
huncho serve --model jaredpalmer/kev-0.8b --require-qualified \
  --revision bf75a6a8848ea6960ff2ed108d9ed44c2941174f \
  --qualification-golden 'kev-0.8b=/absolute/path/to/labeled-golden.json'
```

If your model package lacks a golden or you need a custom evaluation set, override it with `--qualification-golden 'kev-0.8b=/absolute/path/to/pinned-labeled-golden.json'`. Clef additionally starts with pending calibration. See [its guide](clef.md) for fitting and device settings.

## Complete compatible candidate inventory

| Family | Repositories | Runtime |
|---|---:|---|
| F1 | 323 | Candle / ModernBERT typed Laya head |
| F2 | 38 | Candle / Qwen3.5 Kev pointer head |
| F3 | 4 | Candle / Qwen3.5 Nimble candidate logits |
| F5 | 18 | Native Clef / joint schema head |

### F1: Laya-compatible checkpoints

| Repository | Declared license | Checked revision |
|---|---|---|
| [0xSojalSec/laya-System1](https://huggingface.co/0xSojalSec/laya-System1) | apache-2.0 | [325ef6959685](https://huggingface.co/0xSojalSec/laya-System1/tree/325ef69596857536cb456277f9f7736d0deb1b93) |
| [10vivo01/laya-ru-tickets](https://huggingface.co/10vivo01/laya-ru-tickets) | other | [090a89f417d0](https://huggingface.co/10vivo01/laya-ru-tickets/tree/090a89f417d035dc83f86c6956efc8fabe0744dc) |
| [11wanqi/laya-command-safety-ft](https://huggingface.co/11wanqi/laya-command-safety-ft) | mit | [a9f42a0b877f](https://huggingface.co/11wanqi/laya-command-safety-ft/tree/a9f42a0b877fd33b849ab1ed32839e7322fefe03) |
| [16sulphur/laya-prompt-guard](https://huggingface.co/16sulphur/laya-prompt-guard) | apache-2.0 | [1c36ead5ccca](https://huggingface.co/16sulphur/laya-prompt-guard/tree/1c36ead5ccca7568fd105b3b0772dc5171a56bd0) |
| [2045max/laya-custom-model](https://huggingface.co/2045max/laya-custom-model) | not declared in Hub tags | [ff010f48278f](https://huggingface.co/2045max/laya-custom-model/tree/ff010f48278fac41311247684ef9ce72fcf64eb6) |
| [2nugu/laya-ko](https://huggingface.co/2nugu/laya-ko) | apache-2.0 | [386561111d95](https://huggingface.co/2nugu/laya-ko/tree/386561111d95a85f6025882337a3cfd9e54c03ef) |
| [3p3r/sentinel-laya](https://huggingface.co/3p3r/sentinel-laya) | apache-2.0 | [3efebe799179](https://huggingface.co/3p3r/sentinel-laya/tree/3efebe799179a16c805eab272bd727000fea6c07) |
| [3p3r/sentinel-laya-multilingual](https://huggingface.co/3p3r/sentinel-laya-multilingual) | apache-2.0 | [a7539113482f](https://huggingface.co/3p3r/sentinel-laya-multilingual/tree/a7539113482f2631fd733d3f20a5b027ce1b617e) |
| [aalok101singh/trident-laya-process-security-v1](https://huggingface.co/aalok101singh/trident-laya-process-security-v1) | not declared in Hub tags | [3fcaabc41bd1](https://huggingface.co/aalok101singh/trident-laya-process-security-v1/tree/3fcaabc41bd1a7baead882f25217fe8b0b6e4e71) |
| [aaroncool9/laya](https://huggingface.co/aaroncool9/laya) | apache-2.0 | [55fd0220a173](https://huggingface.co/aaroncool9/laya/tree/55fd0220a1734502dee42d31cb0794b1b56ad17e) |
| [aaung/laya-tetris-placement](https://huggingface.co/aaung/laya-tetris-placement) | apache-2.0 | [8a1a828f2d0b](https://huggingface.co/aaung/laya-tetris-placement/tree/8a1a828f2d0b7b8e1330a1d7d610ce6abfa63daf) |
| [abedinia/laya-web-agent](https://huggingface.co/abedinia/laya-web-agent) | apache-2.0 | [88d48cbb142e](https://huggingface.co/abedinia/laya-web-agent/tree/88d48cbb142e63718c0775bc9bb0debf61ca12c1) |
| [Abhi268170/bulwark](https://huggingface.co/Abhi268170/bulwark) | apache-2.0 | [7031b4c9a4a6](https://huggingface.co/Abhi268170/bulwark/tree/7031b4c9a4a699ee53ed5573a3003cf362edeb05) |
| [abhijithsoman27/laya-email-resume-classifier-1](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-1) | apache-2.0 | [96fb5bdb02d3](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-1/tree/96fb5bdb02d34f36563579f48c079f3c5ad589a0) |
| [abhijithsoman27/laya-email-resume-classifier-2](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-2) | apache-2.0 | [eea12c93b52f](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-2/tree/eea12c93b52f001fb27dc12d5e463c9f24f3fb5b) |
| [abhijithsoman27/laya-email-resume-classifier-3](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-3) | apache-2.0 | [4443bc92d4ae](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-3/tree/4443bc92d4aec0bf4f50a018a28ecc8cdfedb12a) |
| [abhijithsoman27/laya-email-resume-classifier-4](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-4) | apache-2.0 | [2fb08f254f0c](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-4/tree/2fb08f254f0cc97c845edad2b33b1931b155d6cf) |
| [abhijithsoman27/laya-email-resume-classifier-v2](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-v2) | apache-2.0 | [f7b42cf2d95b](https://huggingface.co/abhijithsoman27/laya-email-resume-classifier-v2/tree/f7b42cf2d95b0afb5a9ce530c0a40686eec04b01) |
| [abhijithsoman27/laya-recruiting-email-classifier](https://huggingface.co/abhijithsoman27/laya-recruiting-email-classifier) | not declared in Hub tags | [6500c0ce3d33](https://huggingface.co/abhijithsoman27/laya-recruiting-email-classifier/tree/6500c0ce3d33342c2feda61b558f2578705b9f42) |
| [adambloebaum/laya-blackjack](https://huggingface.co/adambloebaum/laya-blackjack) | apache-2.0 | [14af6820702b](https://huggingface.co/adambloebaum/laya-blackjack/tree/14af6820702b3d6eb47d5f9e395f7852af3767ec) |
| [Aditharavind/laya-neuro](https://huggingface.co/Aditharavind/laya-neuro) | apache-2.0 | [82b1fe9131d4](https://huggingface.co/Aditharavind/laya-neuro/tree/82b1fe9131d4d6afb6034c107a755aee6a49ddcf) |
| [Aditharavind/swarm-laya](https://huggingface.co/Aditharavind/swarm-laya) | apache-2.0 | [9391402725a4](https://huggingface.co/Aditharavind/swarm-laya/tree/9391402725a4cc82c3cac9d6482beaf9645e18fd) |
| [agk4444/laya-typed-decisions](https://huggingface.co/agk4444/laya-typed-decisions) | apache-2.0 | [dfebdeee6873](https://huggingface.co/agk4444/laya-typed-decisions/tree/dfebdeee687342a8cd5a2c04a58cb493c38f8328) |
| [AhmetSemih/laya-turkish-mmlu](https://huggingface.co/AhmetSemih/laya-turkish-mmlu) | apache-2.0 | [20285ce35e97](https://huggingface.co/AhmetSemih/laya-turkish-mmlu/tree/20285ce35e97b08fe4d9e6a2192a60487f7740b8) |
| [ai-ml-space/laya](https://huggingface.co/ai-ml-space/laya) | apache-2.0 | [6b481ab01da2](https://huggingface.co/ai-ml-space/laya/tree/6b481ab01da2d74d27be9bf7ce1b6a8aaa3ba68e) |
| [AIMultiple/laya-typed-decisions-browser-ft](https://huggingface.co/AIMultiple/laya-typed-decisions-browser-ft) | apache-2.0 | [e15214b809a1](https://huggingface.co/AIMultiple/laya-typed-decisions-browser-ft/tree/e15214b809a135d2fe71504f66af1b96d7238a89) |
| [akpsahan/laya](https://huggingface.co/akpsahan/laya) | apache-2.0 | [d96df1649680](https://huggingface.co/akpsahan/laya/tree/d96df1649680e4acd18f7f20c5dc6d2b1bd1143d) |
| [akshatbindal/laya-typed-decisions](https://huggingface.co/akshatbindal/laya-typed-decisions) | apache-2.0 | [fea3307fbd5b](https://huggingface.co/akshatbindal/laya-typed-decisions/tree/fea3307fbd5be9937648f9acd22ea612c61fbd83) |
| [alfred361/laya-multilingual-typed-decisions](https://huggingface.co/alfred361/laya-multilingual-typed-decisions) | apache-2.0 | [60ce5be491a7](https://huggingface.co/alfred361/laya-multilingual-typed-decisions/tree/60ce5be491a7723df937b757e43b254a08898f8e) |
| [alihmaou/laya-nao-mobilite](https://huggingface.co/alihmaou/laya-nao-mobilite) | apache-2.0 | [6c8b416e9b64](https://huggingface.co/alihmaou/laya-nao-mobilite/tree/6c8b416e9b645d17b84b3025007600f58047a66a) |
| [ameerhmz5/laya-modernbert-decision-90pct](https://huggingface.co/ameerhmz5/laya-modernbert-decision-90pct) | apache-2.0 | [c2caad4eebb0](https://huggingface.co/ameerhmz5/laya-modernbert-decision-90pct/tree/c2caad4eebb0fef37f6d6011dc79bd539f484ca9) |
| [ampali/laya-genie-judge](https://huggingface.co/ampali/laya-genie-judge) | apache-2.0 | [93a1c0edce2a](https://huggingface.co/ampali/laya-genie-judge/tree/93a1c0edce2a3511e50b23b7568559082c88c6e8) |
| [Ankit1106/laya-typed-decisions](https://huggingface.co/Ankit1106/laya-typed-decisions) | apache-2.0 | [66172d2dfdd5](https://huggingface.co/Ankit1106/laya-typed-decisions/tree/66172d2dfdd5d5c31f0aa5cc999ed2cf2f4b78a2) |
| [anshc02222/laya-cc](https://huggingface.co/anshc02222/laya-cc) | apache-2.0 | [852ee8bad969](https://huggingface.co/anshc02222/laya-cc/tree/852ee8bad969d5857647d0d743b136260d9b9728) |
| [anyforge/ruhui](https://huggingface.co/anyforge/ruhui) | apache-2.0 | [9e82344401aa](https://huggingface.co/anyforge/ruhui/tree/9e82344401aa0c2033e558f6083fa750b307c03b) |
| [api-service-sac/s1-code-v1](https://huggingface.co/api-service-sac/s1-code-v1) | apache-2.0 | [fba6dd050673](https://huggingface.co/api-service-sac/s1-code-v1/tree/fba6dd0506737bbdc324b11c736a9aa902e7d74d) |
| [api-service-sac/s1-code-v2](https://huggingface.co/api-service-sac/s1-code-v2) | apache-2.0 | [37b1c5c595b6](https://huggingface.co/api-service-sac/s1-code-v2/tree/37b1c5c595b64195a6cfa84748cebf46b4704f79) |
| [api-service-sac/s1-code-v3](https://huggingface.co/api-service-sac/s1-code-v3) | apache-2.0 | [9e5cdd6820c0](https://huggingface.co/api-service-sac/s1-code-v3/tree/9e5cdd6820c068d92a7a446191f1c1647692267a) |
| [app2610/laya](https://huggingface.co/app2610/laya) | apache-2.0 | [6495adafec72](https://huggingface.co/app2610/laya/tree/6495adafec72d8b109f5104d4f221556cf70719e) |
| [ariacompute/afm-de](https://huggingface.co/ariacompute/afm-de) | mit | [e80feeb8cdb1](https://huggingface.co/ariacompute/afm-de/tree/e80feeb8cdb1b10a276b252f7bfdcb81124ac953) |
| [assafvayner/laya-dup](https://huggingface.co/assafvayner/laya-dup) | apache-2.0 | [68e12fffb779](https://huggingface.co/assafvayner/laya-dup/tree/68e12fffb7792db612657eba7dc662da7d9c3c7a) |
| [AugustJW/laya-sft-amazon-reviews](https://huggingface.co/AugustJW/laya-sft-amazon-reviews) | apache-2.0 | [b4fc4a486917](https://huggingface.co/AugustJW/laya-sft-amazon-reviews/tree/b4fc4a486917164f22e742efc12d90f0879ac509) |
| [aungthuhein-dev/laya-burmese-sib200](https://huggingface.co/aungthuhein-dev/laya-burmese-sib200) | apache-2.0 | [407ae7276528](https://huggingface.co/aungthuhein-dev/laya-burmese-sib200/tree/407ae7276528aa6fb85968288974c2cbb19a3f89) |
| [azalio/laya-doom-v3](https://huggingface.co/azalio/laya-doom-v3) | apache-2.0 | [d27276e3bf82](https://huggingface.co/azalio/laya-doom-v3/tree/d27276e3bf8275cdbc9a7a72d80cad656aa87bd7) |
| [baharedorsa/laya](https://huggingface.co/baharedorsa/laya) | apache-2.0 | [e476be9310a0](https://huggingface.co/baharedorsa/laya/tree/e476be9310a0a64e7c1a5b22f5cccf3b55750226) |
| [binoykunhi/laya-typed-decisions-endpoint](https://huggingface.co/binoykunhi/laya-typed-decisions-endpoint) | apache-2.0 | [fbefb2d175df](https://huggingface.co/binoykunhi/laya-typed-decisions-endpoint/tree/fbefb2d175df7314582b29c4e3168c5a1c5f0b7e) |
| [BlueAquilae/laya-pack-shaped](https://huggingface.co/BlueAquilae/laya-pack-shaped) | not declared in Hub tags | [befbdb518ac0](https://huggingface.co/BlueAquilae/laya-pack-shaped/tree/befbdb518ac0aa0bdee9c8f3b061dc23d129d93b) |
| [BoogieKn/laya-plays-google-snake](https://huggingface.co/BoogieKn/laya-plays-google-snake) | apache-2.0 | [a1115cbd608d](https://huggingface.co/BoogieKn/laya-plays-google-snake/tree/a1115cbd608d3be58a9be67772f12b35f20d48e9) |
| [Borcherding/laya-multilingual-calculator-router](https://huggingface.co/Borcherding/laya-multilingual-calculator-router) | apache-2.0 | [057a160d736c](https://huggingface.co/Borcherding/laya-multilingual-calculator-router/tree/057a160d736ce3402ad404af26afd06a9023fa46) |
| [BryanSnappCTO/laya-fallacies](https://huggingface.co/BryanSnappCTO/laya-fallacies) | apache-2.0 | [356ec596ee4a](https://huggingface.co/BryanSnappCTO/laya-fallacies/tree/356ec596ee4aeb5656a4be59ac2e8d2c3a3d3346) |
| [ByunByun/laya-android](https://huggingface.co/ByunByun/laya-android) | apache-2.0 | [5a03b2bc4220](https://huggingface.co/ByunByun/laya-android/tree/5a03b2bc4220ec786a00a87eacdaa75cb844bd6b) |
| [canbingol/laya-typed-decisions-turkish-mmlu-10k](https://huggingface.co/canbingol/laya-typed-decisions-turkish-mmlu-10k) | apache-2.0 | [9c4b31347c8d](https://huggingface.co/canbingol/laya-typed-decisions-turkish-mmlu-10k/tree/9c4b31347c8de263fb4d821a1197ed962ac98e10) |
| [capsoul/laya](https://huggingface.co/capsoul/laya) | apache-2.0 | [4a4ade27e5cc](https://huggingface.co/capsoul/laya/tree/4a4ade27e5cc5442ef55118a6b1bfbf54999ca31) |
| [chand1012/laya-instinct](https://huggingface.co/chand1012/laya-instinct) | apache-2.0 | [c1db6bc33e49](https://huggingface.co/chand1012/laya-instinct/tree/c1db6bc33e497f89a5737b1662af9512d87d1d8d) |
| [chand1012/laya-task-area-v2](https://huggingface.co/chand1012/laya-task-area-v2) | apache-2.0 | [14b6e0149aac](https://huggingface.co/chand1012/laya-task-area-v2/tree/14b6e0149aac2b1be9327859d6ef693ac207ad7b) |
| [chand1012/laya-task-tags-v2](https://huggingface.co/chand1012/laya-task-tags-v2) | apache-2.0 | [0a130ccd0e76](https://huggingface.co/chand1012/laya-task-tags-v2/tree/0a130ccd0e767099d2f62ebe458731716b06eab0) |
| [chand1012/laya-task-unified-v2](https://huggingface.co/chand1012/laya-task-unified-v2) | apache-2.0 | [01e9d2d84a42](https://huggingface.co/chand1012/laya-task-unified-v2/tree/01e9d2d84a42898448453f30e39e7f3ed3ae631a) |
| [chaoliangUNSW/MacJev-322M-4K-Laya](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya) | apache-2.0 | [92b182e653eb](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya/tree/92b182e653ebd8facd04ec9bddf6f63c4a7b7c22) |
| [chaoliangUNSW/MacJev-322M-4K-Laya-MLX](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya-MLX) | apache-2.0 | [f2ab36ef98d8](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya-MLX/tree/f2ab36ef98d81e9fc65949e4df35cebc93f21e14) |
| [cklxx/laya-browser](https://huggingface.co/cklxx/laya-browser) | apache-2.0 | [645cf366a2ae](https://huggingface.co/cklxx/laya-browser/tree/645cf366a2ae35f1086e8c20eff48f909bb49206) |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | apache-2.0 | [7b928d828b7b](https://huggingface.co/convaiinnovations/laya/tree/7b928d828b7b0e022f929d9bd2e44165aa270148) |
| [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | apache-2.0 | [1720e3e3357c](https://huggingface.co/convaiinnovations/laya-multilingual/tree/1720e3e3357cfe1e281542e223f8273b0890ca34) |
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | apache-2.0 | [e929ae5cf69b](https://huggingface.co/convaiinnovations/laya-typed-decisions/tree/e929ae5cf69bc34259cd2f95c9e91145b818b1f0) |
| [Cruzex/laya-typed-decisions-smoketest](https://huggingface.co/Cruzex/laya-typed-decisions-smoketest) | apache-2.0 | [a3d15d7b009e](https://huggingface.co/Cruzex/laya-typed-decisions-smoketest/tree/a3d15d7b009e29332e3452866b9c2774bf99b0b6) |
| [daliborsb/laya](https://huggingface.co/daliborsb/laya) | apache-2.0 | [cba84b663fe8](https://huggingface.co/daliborsb/laya/tree/cba84b663fe8024d86af5a9139cc2aa293a16dc6) |
| [Dat2508/rex](https://huggingface.co/Dat2508/rex) | apache-2.0 | [c5cf6789d444](https://huggingface.co/Dat2508/rex/tree/c5cf6789d444cefc0186518dde6118a29bd2d33d) |
| [Dat2508/rex-ole](https://huggingface.co/Dat2508/rex-ole) | apache-2.0 | [82ec151fbf93](https://huggingface.co/Dat2508/rex-ole/tree/82ec151fbf931fadc2a6ee3f12ad6f293be7ff1b) |
| [datafreak/laya-chess](https://huggingface.co/datafreak/laya-chess) | apache-2.0 | [9269e61202b2](https://huggingface.co/datafreak/laya-chess/tree/9269e61202b2c3517fef1cdacea83c3c23b92961) |
| [daviestechlabs/laya-homelab-router](https://huggingface.co/daviestechlabs/laya-homelab-router) | apache-2.0 | [2f5d38881405](https://huggingface.co/daviestechlabs/laya-homelab-router/tree/2f5d38881405512b61e1540da96356842b454efe) |
| [Deepdive404-3/laya](https://huggingface.co/Deepdive404-3/laya) | apache-2.0 | [d236df411bf3](https://huggingface.co/Deepdive404-3/laya/tree/d236df411bf309c27079c79e01fca25f8bf60bf3) |
| [dev-amarnadh-gandham/laya](https://huggingface.co/dev-amarnadh-gandham/laya) | apache-2.0 | [267df53a0e0b](https://huggingface.co/dev-amarnadh-gandham/laya/tree/267df53a0e0b5db1025df22287a948b56c6c97b7) |
| [developerjeremylive/laya-etheroi](https://huggingface.co/developerjeremylive/laya-etheroi) | apache-2.0 | [26f247b65a68](https://huggingface.co/developerjeremylive/laya-etheroi/tree/26f247b65a68cad7f7240e9cdd1372d87ce76c52) |
| [dimitrisdais/laya-banking77-typed-decisions](https://huggingface.co/dimitrisdais/laya-banking77-typed-decisions) | apache-2.0 | [680ab36bc677](https://huggingface.co/dimitrisdais/laya-banking77-typed-decisions/tree/680ab36bc6776774910ae2fb2dc07fe6d7d29ea8) |
| [Dipto084/safe_laya](https://huggingface.co/Dipto084/safe_laya) | apache-2.0 | [dc1a2bee2862](https://huggingface.co/Dipto084/safe_laya/tree/dc1a2bee2862ea827756f0b2e1e3634996767d18) |
| [djairguilherme/laya-news-scorer](https://huggingface.co/djairguilherme/laya-news-scorer) | apache-2.0 | [cfe02d296ce6](https://huggingface.co/djairguilherme/laya-news-scorer/tree/cfe02d296ce60b1ca4beda52144e5d4a5261aa32) |
| [doofz/THX-01](https://huggingface.co/doofz/THX-01) | apache-2.0 | [bef2e2478be7](https://huggingface.co/doofz/THX-01/tree/bef2e2478be7acd79429324a658558a57552a2e8) |
| [duclvQ/vi-law-laya](https://huggingface.co/duclvQ/vi-law-laya) | apache-2.0 | [fd2e62dc3689](https://huggingface.co/duclvQ/vi-law-laya/tree/fd2e62dc36894b3c534c8508da93866949bfb91d) |
| [EduardoHYM/citeverify-laya-s1](https://huggingface.co/EduardoHYM/citeverify-laya-s1) | apache-2.0 | [4b3f359d7a57](https://huggingface.co/EduardoHYM/citeverify-laya-s1/tree/4b3f359d7a576998356b9a535a532377db1ec912) |
| [elnachto/laya-triage-en](https://huggingface.co/elnachto/laya-triage-en) | apache-2.0 | [9596364eddd7](https://huggingface.co/elnachto/laya-triage-en/tree/9596364eddd73379a2bd9e6957463d6471540014) |
| [elnachto/laya-triage-multilingual](https://huggingface.co/elnachto/laya-triage-multilingual) | apache-2.0 | [6c2dad2d8b70](https://huggingface.co/elnachto/laya-triage-multilingual/tree/6c2dad2d8b70b6cdab8f217f14b37908074f35ca) |
| [eltonssouza/relay-laya](https://huggingface.co/eltonssouza/relay-laya) | mit | [a09e691d3cdc](https://huggingface.co/eltonssouza/relay-laya/tree/a09e691d3cdc45058e5fa4eb2497ae304e20f772) |
| [Emerald7664/laya](https://huggingface.co/Emerald7664/laya) | apache-2.0 | [59bce85b0afa](https://huggingface.co/Emerald7664/laya/tree/59bce85b0afa0df19e8447a11cef68fe18e37c4c) |
| [eric-ml-nlp/Delveta-LayaChoice-v1](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1) | not declared in Hub tags | [824f4684105e](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1/tree/824f4684105e52f15eb9e58a8d6ae856935cda37) |
| [eric-ml-nlp/Delveta-LayaChoice-v2](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2) | not declared in Hub tags | [417e3627fc0b](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2/tree/417e3627fc0b593e08b89f0c4e21f3bcc7c9a16c) |
| [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1) | cc-by-sa-4.0 | [c2b18d33d1e4](https://huggingface.co/faall7479/laya-idjvsuen-v1/tree/c2b18d33d1e4cfdb41211d7b23fa4ca961dab46e) |
| [faall7479/laya-idjvsuen-v3](https://huggingface.co/faall7479/laya-idjvsuen-v3) | cc-by-sa-4.0 | [30d60e31c71e](https://huggingface.co/faall7479/laya-idjvsuen-v3/tree/30d60e31c71eb83e4797b2e839ed942a0cff1647) |
| [faall7479/laya-idjvsuen-v4](https://huggingface.co/faall7479/laya-idjvsuen-v4) | cc-by-sa-4.0 | [f55b2d170704](https://huggingface.co/faall7479/laya-idjvsuen-v4/tree/f55b2d170704f285d0dbaad4e64617256bb482b0) |
| [FabioCeleste/laya-mtg-deckbuild](https://huggingface.co/FabioCeleste/laya-mtg-deckbuild) | apache-2.0 | [f82726490b27](https://huggingface.co/FabioCeleste/laya-mtg-deckbuild/tree/f82726490b273bdaf6ddc8ddc2950a7b45a96d80) |
| [FabioCeleste/laya-mtg-draft-picks](https://huggingface.co/FabioCeleste/laya-mtg-draft-picks) | apache-2.0 | [92d4dc527d37](https://huggingface.co/FabioCeleste/laya-mtg-draft-picks/tree/92d4dc527d377ff7a20bb5d28bf16a2187d0aa7a) |
| [FAISALFAZALHUSSAIN/laya](https://huggingface.co/FAISALFAZALHUSSAIN/laya) | apache-2.0 | [6f8b95669e4c](https://huggingface.co/FAISALFAZALHUSSAIN/laya/tree/6f8b95669e4c20049318ecd9bd85635064b55ddb) |
| [fcelabs/laya-glendesk-issues](https://huggingface.co/fcelabs/laya-glendesk-issues) | apache-2.0 | [02d85a90718e](https://huggingface.co/fcelabs/laya-glendesk-issues/tree/02d85a90718eee6639bb77fcc85edb4eb18588a3) |
| [featherless-ai/laya-plus-classifier](https://huggingface.co/featherless-ai/laya-plus-classifier) | not declared in Hub tags | [d4c9aeee6547](https://huggingface.co/featherless-ai/laya-plus-classifier/tree/d4c9aeee6547559b20625c76b4740e71d5a16775) |
| [flaukowski/laya-kannaka-evidence-gate](https://huggingface.co/flaukowski/laya-kannaka-evidence-gate) | apache-2.0 | [efd99768b556](https://huggingface.co/flaukowski/laya-kannaka-evidence-gate/tree/efd99768b5569aee95ea05cc5e91b47b776266ed) |
| [flaukowski/laya-kannaka-supersession](https://huggingface.co/flaukowski/laya-kannaka-supersession) | apache-2.0 | [c494ee705e5c](https://huggingface.co/flaukowski/laya-kannaka-supersession/tree/c494ee705e5cd4400ed702233b14993732114da7) |
| [GeekyAbs/laya](https://huggingface.co/GeekyAbs/laya) | apache-2.0 | [b65d05b4d9eb](https://huggingface.co/GeekyAbs/laya/tree/b65d05b4d9eb716b163e4348026c07fbdff615bd) |
| [Gjusev/laya-triage-banking77](https://huggingface.co/Gjusev/laya-triage-banking77) | apache-2.0 | [1f223a8b034e](https://huggingface.co/Gjusev/laya-triage-banking77/tree/1f223a8b034e93e1452266a434a2deae68752de3) |
| [goktugoguz/laya-multilingual-arc](https://huggingface.co/goktugoguz/laya-multilingual-arc) | apache-2.0 | [6f595ca56055](https://huggingface.co/goktugoguz/laya-multilingual-arc/tree/6f595ca56055165829ecd923db241115b505d56e) |
| [goktugoguz/laya-multilingual-stellar](https://huggingface.co/goktugoguz/laya-multilingual-stellar) | apache-2.0 | [189883aa9a7c](https://huggingface.co/goktugoguz/laya-multilingual-stellar/tree/189883aa9a7c22f36a48ede5a6cc47ecff43d4c4) |
| [gtm-k/laya-maintenance-decision-assist](https://huggingface.co/gtm-k/laya-maintenance-decision-assist) | apache-2.0 | [8989e4ed2106](https://huggingface.co/gtm-k/laya-maintenance-decision-assist/tree/8989e4ed2106315ed15692a9228aee9e509f70ff) |
| [gtm-k/laya-product-decision-assist](https://huggingface.co/gtm-k/laya-product-decision-assist) | apache-2.0 | [91294410a613](https://huggingface.co/gtm-k/laya-product-decision-assist/tree/91294410a613e3cc7a80f80402e258df4716d472) |
| [Gtrkrsk/laya](https://huggingface.co/Gtrkrsk/laya) | apache-2.0 | [f6029af5253c](https://huggingface.co/Gtrkrsk/laya/tree/f6029af5253c735c891f0678ba49879469ba4cf8) |
| [hackoffice/laya-multilingual](https://huggingface.co/hackoffice/laya-multilingual) | apache-2.0 | [b525d41914bd](https://huggingface.co/hackoffice/laya-multilingual/tree/b525d41914bd33b3672b4b5e3d1fb3a75433fff0) |
| [Hagwell/laya](https://huggingface.co/Hagwell/laya) | apache-2.0 | [01b0a6f0b09f](https://huggingface.co/Hagwell/laya/tree/01b0a6f0b09f6eb97531d645c9f920d01111c304) |
| [hanish-builds/laya-browser-v1](https://huggingface.co/hanish-builds/laya-browser-v1) | not declared in Hub tags | [32c5ee734fdd](https://huggingface.co/hanish-builds/laya-browser-v1/tree/32c5ee734fdd39eadac5c3675ebd167ac26bf119) |
| [harikarthikmanyam/laya-issue-triage](https://huggingface.co/harikarthikmanyam/laya-issue-triage) | apache-2.0 | [ad004d061120](https://huggingface.co/harikarthikmanyam/laya-issue-triage/tree/ad004d06112072b4ee9d390aa5c47ae2249209c1) |
| [harrissondutra/laya-audiovitrine](https://huggingface.co/harrissondutra/laya-audiovitrine) | not declared in Hub tags | [47ebcc18debd](https://huggingface.co/harrissondutra/laya-audiovitrine/tree/47ebcc18debd4d234f15bbe2767bd9090ad513b9) |
| [harrissondutra/laya-supervitrine](https://huggingface.co/harrissondutra/laya-supervitrine) | not declared in Hub tags | [6306c3cb6853](https://huggingface.co/harrissondutra/laya-supervitrine/tree/6306c3cb6853e8d84df3d4d9ea5feef633ff857d) |
| [Harsh1312/laya-pr-triage](https://huggingface.co/Harsh1312/laya-pr-triage) | apache-2.0 | [56343f095893](https://huggingface.co/Harsh1312/laya-pr-triage/tree/56343f095893f7a4b602f350fe9639332d827f8e) |
| [hayriyigit/laya-tr](https://huggingface.co/hayriyigit/laya-tr) | other | [3522e37ce68b](https://huggingface.co/hayriyigit/laya-tr/tree/3522e37ce68bb676b3a7a8843158b2796d33e6cd) |
| [hayriyigit/laya-tr-karar](https://huggingface.co/hayriyigit/laya-tr-karar) | other | [24e50b23353d](https://huggingface.co/hayriyigit/laya-tr-karar/tree/24e50b23353dfeddcd8fcd79f7c3d08553c5d32a) |
| [helmo/laya](https://huggingface.co/helmo/laya) | apache-2.0 | [59c9e3cdd4c0](https://huggingface.co/helmo/laya/tree/59c9e3cdd4c0d00081eb8028d31d0126826c6b9c) |
| [henilchopada/laya-v](https://huggingface.co/henilchopada/laya-v) | apache-2.0 | [5c56367ba43a](https://huggingface.co/henilchopada/laya-v/tree/5c56367ba43ad790a2ba8c1491e1bcc9b4cca56c) |
| [hiepho/viya-base](https://huggingface.co/hiepho/viya-base) | apache-2.0 | [c806a2873ecc](https://huggingface.co/hiepho/viya-base/tree/c806a2873ecc4b63d4db0fa68c3ff7d1bf87c477) |
| [HimanshuPathak/laya](https://huggingface.co/HimanshuPathak/laya) | apache-2.0 | [9158f1271b3a](https://huggingface.co/HimanshuPathak/laya/tree/9158f1271b3a832be6850a4d0c1a2ce4c2ad1520) |
| [Himvans/Saarthi](https://huggingface.co/Himvans/Saarthi) | apache-2.0 | [a94022751c7b](https://huggingface.co/Himvans/Saarthi/tree/a94022751c7b38604ef0bbae859c16806e223688) |
| [hravi/laya-finance](https://huggingface.co/hravi/laya-finance) | other | [6b9cb3a164fd](https://huggingface.co/hravi/laya-finance/tree/6b9cb3a164fda0ab442d2c05993eb5fb1bb243b4) |
| [Hspro430/laya-f1-finetuned](https://huggingface.co/Hspro430/laya-f1-finetuned) | not declared in Hub tags | [af90788652c9](https://huggingface.co/Hspro430/laya-f1-finetuned/tree/af90788652c9639c1786258f254ff690a50c378a) |
| [huanggengwei/laya-typed-decisions](https://huggingface.co/huanggengwei/laya-typed-decisions) | apache-2.0 | [14cc9a5e4e3c](https://huggingface.co/huanggengwei/laya-typed-decisions/tree/14cc9a5e4e3c3c3d6ccbb0cbdb7848d40dfdc977) |
| [hxrikp/laya-arxiv-topic-router](https://huggingface.co/hxrikp/laya-arxiv-topic-router) | other | [aa933e557cc2](https://huggingface.co/hxrikp/laya-arxiv-topic-router/tree/aa933e557cc26eafc6f6045a78515e3b297cb112) |
| [hxrikp/laya-session-guard-followup-original](https://huggingface.co/hxrikp/laya-session-guard-followup-original) | apache-2.0 | [6a3241690476](https://huggingface.co/hxrikp/laya-session-guard-followup-original/tree/6a3241690476389c8447d6a8b28840852156b209) |
| [hxrikp/laya-session-guard-pilot](https://huggingface.co/hxrikp/laya-session-guard-pilot) | apache-2.0 | [d8f65c3c6816](https://huggingface.co/hxrikp/laya-session-guard-pilot/tree/d8f65c3c68162163b7fe0f8c06bc40cd8d12beed) |
| [hxrikp/laya-wikirace-wikispeedia](https://huggingface.co/hxrikp/laya-wikirace-wikispeedia) | apache-2.0 | [f967d3656611](https://huggingface.co/hxrikp/laya-wikirace-wikispeedia/tree/f967d3656611e5410a30b6262f9b5a8a362ee0d1) |
| [igserviceu/laya](https://huggingface.co/igserviceu/laya) | apache-2.0 | [f0fdc7cf02cc](https://huggingface.co/igserviceu/laya/tree/f0fdc7cf02cc4e1158a9f76494f548e5fca780bb) |
| [ikaanakkok/laya-browser-v1](https://huggingface.co/ikaanakkok/laya-browser-v1) | mit | [352d2442e84c](https://huggingface.co/ikaanakkok/laya-browser-v1/tree/352d2442e84c9b64459a0234c6d516c86194787b) |
| [ikaanakkok/laya-desktop-v1](https://huggingface.co/ikaanakkok/laya-desktop-v1) | mit | [b6b6a5f75c08](https://huggingface.co/ikaanakkok/laya-desktop-v1/tree/b6b6a5f75c083dc4b5da766c5751be54ea7896fa) |
| [impacte/bunker-laya](https://huggingface.co/impacte/bunker-laya) | apache-2.0 | [6dee05e17edb](https://huggingface.co/impacte/bunker-laya/tree/6dee05e17edbda725d859e6a3e8c7aec92448462) |
| [impacte/laya-medical-escalation](https://huggingface.co/impacte/laya-medical-escalation) | apache-2.0 | [91c869598a4c](https://huggingface.co/impacte/laya-medical-escalation/tree/91c869598a4c29ec4e68cfea40ac3761f609ca54) |
| [impacte/mimir-laya-router](https://huggingface.co/impacte/mimir-laya-router) | apache-2.0 | [441057d7578b](https://huggingface.co/impacte/mimir-laya-router/tree/441057d7578bf6a6ee8cf207a44b00cf01a0738e) |
| [Infin8-AI/laya](https://huggingface.co/Infin8-AI/laya) | apache-2.0 | [e9012d95a9af](https://huggingface.co/Infin8-AI/laya/tree/e9012d95a9af8ff33cddf5745ab6003a4b5b2557) |
| [jaswanthsanjay88/rev-decision-model](https://huggingface.co/jaswanthsanjay88/rev-decision-model) | apache-2.0 | [b7d8eb7f7fee](https://huggingface.co/jaswanthsanjay88/rev-decision-model/tree/b7d8eb7f7feede28c720d88fe022df71fd0a02e9) |
| [jayzou3773/laya-ra2-v33](https://huggingface.co/jayzou3773/laya-ra2-v33) | apache-2.0 | [51d2f5a0b36c](https://huggingface.co/jayzou3773/laya-ra2-v33/tree/51d2f5a0b36c0a9921a450f14a5afb7cb6d30685) |
| [jeremierostan/laya-pairmatch](https://huggingface.co/jeremierostan/laya-pairmatch) | apache-2.0 | [a2906497d12e](https://huggingface.co/jeremierostan/laya-pairmatch/tree/a2906497d12e65fce76a66f3432a346f2d472a66) |
| [jeremierostan/laya-student-match](https://huggingface.co/jeremierostan/laya-student-match) | apache-2.0 | [8562f27a23c2](https://huggingface.co/jeremierostan/laya-student-match/tree/8562f27a23c2cf687b9263dcc0218af9c8db448d) |
| [jerichosiahaya/laya-spam-indonesian](https://huggingface.co/jerichosiahaya/laya-spam-indonesian) | apache-2.0 | [38d520b0ab55](https://huggingface.co/jerichosiahaya/laya-spam-indonesian/tree/38d520b0ab5502610ac5c402a931ee06934ae5f7) |
| [Jojoarumugam/laya-agentguard](https://huggingface.co/Jojoarumugam/laya-agentguard) | apache-2.0 | [e255b9386560](https://huggingface.co/Jojoarumugam/laya-agentguard/tree/e255b93865605420bc7bd7efacadadf572e6d01e) |
| [juteq/arbit-base](https://huggingface.co/juteq/arbit-base) | apache-2.0 | [36d4513c3ed1](https://huggingface.co/juteq/arbit-base/tree/36d4513c3ed1007f9ec712f94018afd0de0a5b37) |
| [kevago/calvino-laya-ft](https://huggingface.co/kevago/calvino-laya-ft) | apache-2.0 | [464ee356fe2f](https://huggingface.co/kevago/calvino-laya-ft/tree/464ee356fe2f9fea081bd0a766cc12646c0e8f80) |
| [kgrozdanovski/caldec-v1-laya](https://huggingface.co/kgrozdanovski/caldec-v1-laya) | apache-2.0 | [6a7e10eb7958](https://huggingface.co/kgrozdanovski/caldec-v1-laya/tree/6a7e10eb795806dd3a8e80c2ae8d1fa803991757) |
| [knpatil/laya-browser-agent](https://huggingface.co/knpatil/laya-browser-agent) | apache-2.0 | [321117669766](https://huggingface.co/knpatil/laya-browser-agent/tree/3211176697668976dfe84b5c3d978d8d1c88d9b8) |
| [kp-forks/laya](https://huggingface.co/kp-forks/laya) | apache-2.0 | [67108e4ee27b](https://huggingface.co/kp-forks/laya/tree/67108e4ee27bef55833fe6f11003de73134b6e28) |
| [Krazher/laya-naryad](https://huggingface.co/Krazher/laya-naryad) | other | [f1b9a64243c5](https://huggingface.co/Krazher/laya-naryad/tree/f1b9a64243c5c4911c2d676521d076d74edbebda) |
| [ksthink/laya-typed-decisions](https://huggingface.co/ksthink/laya-typed-decisions) | apache-2.0 | [14b612461736](https://huggingface.co/ksthink/laya-typed-decisions/tree/14b612461736ec7b776cf9377838efcdabaeb9d6) |
| [Kuramajiulama/laya](https://huggingface.co/Kuramajiulama/laya) | apache-2.0 | [9efdb463374f](https://huggingface.co/Kuramajiulama/laya/tree/9efdb463374f1184d4168865824eaffdcb04a146) |
| [ledio/laya-typed-decisions](https://huggingface.co/ledio/laya-typed-decisions) | apache-2.0 | [3ad96d4903be](https://huggingface.co/ledio/laya-typed-decisions/tree/3ad96d4903be5db6f73334aad95a26238e8e8418) |
| [leloss/LayaCrime-v0](https://huggingface.co/leloss/LayaCrime-v0) | not declared in Hub tags | [b6008e06de83](https://huggingface.co/leloss/LayaCrime-v0/tree/b6008e06de83c28291f5939d50e00905a45940c7) |
| [liskasYR/laya](https://huggingface.co/liskasYR/laya) | apache-2.0 | [9baf31afe1b2](https://huggingface.co/liskasYR/laya/tree/9baf31afe1b29b31425fcc2e1c27fcfdeaa70964) |
| [liskasYR/laya-typed-decisions](https://huggingface.co/liskasYR/laya-typed-decisions) | apache-2.0 | [12522a05c214](https://huggingface.co/liskasYR/laya-typed-decisions/tree/12522a05c214ce848b80d1cf1516c9f8c6fb6ff1) |
| [loxic68361/laya-driving](https://huggingface.co/loxic68361/laya-driving) | apache-2.0 | [7c973b8b2043](https://huggingface.co/loxic68361/laya-driving/tree/7c973b8b2043eee24a2c7aa7ed0be0c809754910) |
| [LucasRabay/lance-a-laya](https://huggingface.co/LucasRabay/lance-a-laya) | apache-2.0 | [a0de0fcbf071](https://huggingface.co/LucasRabay/lance-a-laya/tree/a0de0fcbf07103e7015f69264b40a163f86c87db) |
| [Luni/laya-grounded](https://huggingface.co/Luni/laya-grounded) | cc-by-nc-4.0 | [adf7152c72af](https://huggingface.co/Luni/laya-grounded/tree/adf7152c72af118bf90b4e5e9c5b158c70e95ec0) |
| [madhavbiplov/laya-snake-mlx](https://huggingface.co/madhavbiplov/laya-snake-mlx) | apache-2.0 | [1a7117493328](https://huggingface.co/madhavbiplov/laya-snake-mlx/tree/1a7117493328f1e649c23c149daebfe0f678b621) |
| [MainzelMennchen/LayaVoxprompt](https://huggingface.co/MainzelMennchen/LayaVoxprompt) | apache-2.0 | [654fde85fa2b](https://huggingface.co/MainzelMennchen/LayaVoxprompt/tree/654fde85fa2b8df202888fba06bb5283c74d4191) |
| [maleshku/laya-accounting-canada](https://huggingface.co/maleshku/laya-accounting-canada) | not declared in Hub tags | [ef9546acb416](https://huggingface.co/maleshku/laya-accounting-canada/tree/ef9546acb4168b719864e0fd1f2081723d3ad4bb) |
| [Marianinpb/laya-context-router](https://huggingface.co/Marianinpb/laya-context-router) | not declared in Hub tags | [3e25af77c951](https://huggingface.co/Marianinpb/laya-context-router/tree/3e25af77c95186aaa44f00abe5ab75a6d5329483) |
| [Marianinpb/laya-toolrag-router](https://huggingface.co/Marianinpb/laya-toolrag-router) | not declared in Hub tags | [687ed637f914](https://huggingface.co/Marianinpb/laya-toolrag-router/tree/687ed637f914b5159781a384f08877e0584d982d) |
| [marioluciofjr/layakedin](https://huggingface.co/marioluciofjr/layakedin) | apache-2.0 | [9f1eff235bdc](https://huggingface.co/marioluciofjr/layakedin/tree/9f1eff235bdc65a6a3f913a11babfe9327f312a5) |
| [Masa557/laya-pii-ja](https://huggingface.co/Masa557/laya-pii-ja) | apache-2.0 | [2cda342cd64b](https://huggingface.co/Masa557/laya-pii-ja/tree/2cda342cd64b153db42f58a37884d85892d7718b) |
| [Mass121/laya-typed-decisions](https://huggingface.co/Mass121/laya-typed-decisions) | apache-2.0 | [6d9d73bfcf4e](https://huggingface.co/Mass121/laya-typed-decisions/tree/6d9d73bfcf4e16d176b3a76dd66c74b64e406a77) |
| [Mass121/laya-typed-decisions-v2](https://huggingface.co/Mass121/laya-typed-decisions-v2) | apache-2.0 | [6fb41711ea5f](https://huggingface.co/Mass121/laya-typed-decisions-v2/tree/6fb41711ea5ff81097e6fc09ccebf31138b06e4d) |
| [Mauceric/laya-tiron-router](https://huggingface.co/Mauceric/laya-tiron-router) | apache-2.0 | [b2e36fb1f43d](https://huggingface.co/Mauceric/laya-tiron-router/tree/b2e36fb1f43dad32e49f5a2a08a45c5b98fcb6a7) |
| [mazenDDr/laya-crisis-triage](https://huggingface.co/mazenDDr/laya-crisis-triage) | cc-by-nc-sa-4.0 | [cfa0d6bfd9ba](https://huggingface.co/mazenDDr/laya-crisis-triage/tree/cfa0d6bfd9bae3672bda05feedca12426a4b8ec4) |
| [mgoeckel/oscar-1-150m](https://huggingface.co/mgoeckel/oscar-1-150m) | apache-2.0 | [1015773b7b1e](https://huggingface.co/mgoeckel/oscar-1-150m/tree/1015773b7b1e3dcab42029db39efeffaed31cfce) |
| [mgoeckel/oscar-1-17m](https://huggingface.co/mgoeckel/oscar-1-17m) | apache-2.0 | [a443c1810753](https://huggingface.co/mgoeckel/oscar-1-17m/tree/a443c18107534cb3148941de601e15a89a2c6da3) |
| [mgoeckel/oscar-1-32m](https://huggingface.co/mgoeckel/oscar-1-32m) | apache-2.0 | [6b4289e97e3b](https://huggingface.co/mgoeckel/oscar-1-32m/tree/6b4289e97e3bda37dad070bb560affb75107250f) |
| [mgoeckel/oscar-1-400m](https://huggingface.co/mgoeckel/oscar-1-400m) | apache-2.0 | [d0a87f7adcf5](https://huggingface.co/mgoeckel/oscar-1-400m/tree/d0a87f7adcf5ad3dbb8616eafe6a1c2d2e667f06) |
| [mgoeckel/oscar-1-68m](https://huggingface.co/mgoeckel/oscar-1-68m) | apache-2.0 | [9d9a246edd04](https://huggingface.co/mgoeckel/oscar-1-68m/tree/9d9a246edd041bda0531d28000ccf044bf5703fc) |
| [michaelfeil/laya-typed-decisions](https://huggingface.co/michaelfeil/laya-typed-decisions) | apache-2.0 | [2741340228fc](https://huggingface.co/michaelfeil/laya-typed-decisions/tree/2741340228fc0c982ae9dd63b34dcd5a9e7ce3ce) |
| [minhquan2310/dispositio](https://huggingface.co/minhquan2310/dispositio) | apache-2.0 | [062289b671bb](https://huggingface.co/minhquan2310/dispositio/tree/062289b671bbeeed97a3e58405c6fae825ea6e4c) |
| [misukisu/laya-fi-multilingual](https://huggingface.co/misukisu/laya-fi-multilingual) | apache-2.0 | [b34c70eed1b2](https://huggingface.co/misukisu/laya-fi-multilingual/tree/b34c70eed1b2e6eda460cfdf4e7a77eefb169771) |
| [mmahdi-sz/Laya-fa](https://huggingface.co/mmahdi-sz/Laya-fa) | not declared in Hub tags | [4f8a1b4fc6ee](https://huggingface.co/mmahdi-sz/Laya-fa/tree/4f8a1b4fc6ee887a12dddd6be0fe77395d90b682) |
| [mmahdi-sz/Laya-fa-universal-support](https://huggingface.co/mmahdi-sz/Laya-fa-universal-support) | mit | [ec31f11383e6](https://huggingface.co/mmahdi-sz/Laya-fa-universal-support/tree/ec31f11383e66a0494b3e2dc47282194d97b3202) |
| [mo-shadfar/laya-fa-general](https://huggingface.co/mo-shadfar/laya-fa-general) | apache-2.0 | [2b86ca01851f](https://huggingface.co/mo-shadfar/laya-fa-general/tree/2b86ca01851f7fc44a77a284301bfc644d9ef9d8) |
| [mo-shadfar/laya-fa-intent-routing](https://huggingface.co/mo-shadfar/laya-fa-intent-routing) | apache-2.0 | [9643d466d219](https://huggingface.co/mo-shadfar/laya-fa-intent-routing/tree/9643d466d219a4a5dc2308c7b17602d89c60cb3d) |
| [mo-shadfar/laya-fa-retail-shopping](https://huggingface.co/mo-shadfar/laya-fa-retail-shopping) | apache-2.0 | [8d7cf55ff167](https://huggingface.co/mo-shadfar/laya-fa-retail-shopping/tree/8d7cf55ff167230c49f4b4b1eccce7b17c86e1e1) |
| [mo-shadfar/laya-fa-semtiment](https://huggingface.co/mo-shadfar/laya-fa-semtiment) | apache-2.0 | [2f556c156c49](https://huggingface.co/mo-shadfar/laya-fa-semtiment/tree/2f556c156c49ac84d9a12672b4822f5ca8487973) |
| [mo-shadfar/laya-fa-spam-detectiom](https://huggingface.co/mo-shadfar/laya-fa-spam-detectiom) | apache-2.0 | [8e5419a249bd](https://huggingface.co/mo-shadfar/laya-fa-spam-detectiom/tree/8e5419a249bd7e08a26a0b05f152c7da18c34424) |
| [mo-shadfar/laya-fa-spam-detection](https://huggingface.co/mo-shadfar/laya-fa-spam-detection) | apache-2.0 | [6a180ee48839](https://huggingface.co/mo-shadfar/laya-fa-spam-detection/tree/6a180ee48839b4ea69aa57d635af11eeca7183bf) |
| [Mohammadabdi96/Laya_assessment](https://huggingface.co/Mohammadabdi96/Laya_assessment) | apache-2.0 | [36da8a04dba0](https://huggingface.co/Mohammadabdi96/Laya_assessment/tree/36da8a04dba04c537c6e3c90be02318cd5e18549) |
| [mohanvarma22/laya-safety-beavertails](https://huggingface.co/mohanvarma22/laya-safety-beavertails) | not declared in Hub tags | [09c4619740c4](https://huggingface.co/mohanvarma22/laya-safety-beavertails/tree/09c4619740c46a09c04f1ca687c3bb44ff37ee1e) |
| [muhammadnoman76/jev-urdu](https://huggingface.co/muhammadnoman76/jev-urdu) | apache-2.0 | [058ea3f7a86c](https://huggingface.co/muhammadnoman76/jev-urdu/tree/058ea3f7a86c4d7c5d95fb06d696eefb1319384e) |
| [muthuishere/openjevx](https://huggingface.co/muthuishere/openjevx) | apache-2.0 | [5399cb82a1ea](https://huggingface.co/muthuishere/openjevx/tree/5399cb82a1eafad35575a6eaff5e2c5de5a8fc7e) |
| [mvilacad/laya-conductor](https://huggingface.co/mvilacad/laya-conductor) | apache-2.0 | [04116e680834](https://huggingface.co/mvilacad/laya-conductor/tree/04116e6808345ee4aa9477afa5e405cd48ab8421) |
| [mykytaproks/mykytaproks](https://huggingface.co/mykytaproks/mykytaproks) | apache-2.0 | [9fbcd3c81a56](https://huggingface.co/mykytaproks/mykytaproks/tree/9fbcd3c81a56303b671d20cccded83a6754f53da) |
| [nafiullah/laya-multilingual-bn-ecom-voice](https://huggingface.co/nafiullah/laya-multilingual-bn-ecom-voice) | cc-by-nc-sa-4.0 | [31ee550a47c7](https://huggingface.co/nafiullah/laya-multilingual-bn-ecom-voice/tree/31ee550a47c74b49c0e22e2df0e60998afd7b661) |
| [nathanribeiroo/laya-cartoes-pt](https://huggingface.co/nathanribeiroo/laya-cartoes-pt) | apache-2.0 | [e0fda95d1e70](https://huggingface.co/nathanribeiroo/laya-cartoes-pt/tree/e0fda95d1e705ffa089eca611b004e8e9889a115) |
| [Nemo601/laya-jev-chat](https://huggingface.co/Nemo601/laya-jev-chat) | not declared in Hub tags | [b9b8d1e2e245](https://huggingface.co/Nemo601/laya-jev-chat/tree/b9b8d1e2e245a4dfdf86df5a252e48b58a76f3bd) |
| [ngdghfdc/head-arb-gold](https://huggingface.co/ngdghfdc/head-arb-gold) | apache-2.0 | [e98373e07b91](https://huggingface.co/ngdghfdc/head-arb-gold/tree/e98373e07b91f6e0ae711f894c76925c643eea8a) |
| [ngdghfdc/head-bbox-gold](https://huggingface.co/ngdghfdc/head-bbox-gold) | apache-2.0 | [3116fadee318](https://huggingface.co/ngdghfdc/head-bbox-gold/tree/3116fadee318d69c4c40d6aff8b77727bc743326) |
| [ngdghfdc/head-comp-gold](https://huggingface.co/ngdghfdc/head-comp-gold) | apache-2.0 | [c47f74d2f323](https://huggingface.co/ngdghfdc/head-comp-gold/tree/c47f74d2f323e1d7a7b14ba987efcba1ec5d8bcd) |
| [ngdghfdc/head-cons-gold](https://huggingface.co/ngdghfdc/head-cons-gold) | apache-2.0 | [a887c4e50814](https://huggingface.co/ngdghfdc/head-cons-gold/tree/a887c4e508149a3247478f55f839cc45cb585ab1) |
| [ngdghfdc/head-exit-gold](https://huggingface.co/ngdghfdc/head-exit-gold) | apache-2.0 | [f605aeb0cd94](https://huggingface.co/ngdghfdc/head-exit-gold/tree/f605aeb0cd94730eee4888a454b09ee3e46c2cbe) |
| [ngdghfdc/head-loc-gold](https://huggingface.co/ngdghfdc/head-loc-gold) | apache-2.0 | [d20c228fe17c](https://huggingface.co/ngdghfdc/head-loc-gold/tree/d20c228fe17c227bb6d2ee8e870e1497ec01493f) |
| [ngdghfdc/head-prio-gold](https://huggingface.co/ngdghfdc/head-prio-gold) | apache-2.0 | [77bb2026512e](https://huggingface.co/ngdghfdc/head-prio-gold/tree/77bb2026512e2c35a64a05cccd7f24e8cf133295) |
| [ngdghfdc/head-rout-gold](https://huggingface.co/ngdghfdc/head-rout-gold) | apache-2.0 | [a3622f2a9263](https://huggingface.co/ngdghfdc/head-rout-gold/tree/a3622f2a9263e70f0e8558eb5bfd219724d62df2) |
| [ngdghfdc/head-triage-gold](https://huggingface.co/ngdghfdc/head-triage-gold) | apache-2.0 | [cd15bde9e65a](https://huggingface.co/ngdghfdc/head-triage-gold/tree/cd15bde9e65afd3f12cae1582f7221a1dfb7822f) |
| [nishanth-kj/laya](https://huggingface.co/nishanth-kj/laya) | apache-2.0 | [933512bd5b1d](https://huggingface.co/nishanth-kj/laya/tree/933512bd5b1d288fd9174299764a0134fd957ce2) |
| [ntraidmode/laya-x64-v4](https://huggingface.co/ntraidmode/laya-x64-v4) | apache-2.0 | [1cb313f54a13](https://huggingface.co/ntraidmode/laya-x64-v4/tree/1cb313f54a13f127ccdeb4d794402b2e568e1dc3) |
| [nuhmanpk/pawn-17m](https://huggingface.co/nuhmanpk/pawn-17m) | apache-2.0 | [954140accc79](https://huggingface.co/nuhmanpk/pawn-17m/tree/954140accc792ea22e33398f95d183814eea4520) |
| [nuhmanpk/pawn-32m](https://huggingface.co/nuhmanpk/pawn-32m) | apache-2.0 | [c67412bec3b5](https://huggingface.co/nuhmanpk/pawn-32m/tree/c67412bec3b554a5451f9a88a1422c22cb34557b) |
| [nuhmanpk/pawn-68m](https://huggingface.co/nuhmanpk/pawn-68m) | apache-2.0 | [656eec2951b5](https://huggingface.co/nuhmanpk/pawn-68m/tree/656eec2951b55e906ede3fae3d863dde97ee639f) |
| [nuhmanpk/rook-1](https://huggingface.co/nuhmanpk/rook-1) | apache-2.0 | [9285e3cddff8](https://huggingface.co/nuhmanpk/rook-1/tree/9285e3cddff8bc7600ff2ce0ec1b0b2523bf484a) |
| [Omarmahmoud83/laya_compliance](https://huggingface.co/Omarmahmoud83/laya_compliance) | not declared in Hub tags | [194e82f098df](https://huggingface.co/Omarmahmoud83/laya_compliance/tree/194e82f098df324ae6aec83d3ad6c331c9de599e) |
| [Omarmahmoud83/laya_type_decision](https://huggingface.co/Omarmahmoud83/laya_type_decision) | not declared in Hub tags | [84bc704cfc54](https://huggingface.co/Omarmahmoud83/laya_type_decision/tree/84bc704cfc54b7454f478c64ee5fc15976abe60f) |
| [opg13/laya](https://huggingface.co/opg13/laya) | apache-2.0 | [99175af5d679](https://huggingface.co/opg13/laya/tree/99175af5d67919ae93b7ffc1a1db4835af14245e) |
| [ottosulin/safe-laya](https://huggingface.co/ottosulin/safe-laya) | apache-2.0 | [02ccb35886c9](https://huggingface.co/ottosulin/safe-laya/tree/02ccb35886c95f334428756a65d5debccefe3489) |
| [OwaisAli10/laya-snake](https://huggingface.co/OwaisAli10/laya-snake) | apache-2.0 | [21057b19696f](https://huggingface.co/OwaisAli10/laya-snake/tree/21057b19696fb3e27c4f4c644e6cb872ed866486) |
| [ParkJH09/ownd-laya-search](https://huggingface.co/ParkJH09/ownd-laya-search) | not declared in Hub tags | [7158dd379d6e](https://huggingface.co/ParkJH09/ownd-laya-search/tree/7158dd379d6e2cfe9d54b4f5592f6593ad59c53a) |
| [pashwinkrish22/laya-email-resume-classifier](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier) | apache-2.0 | [25da14a9ce7e](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier/tree/25da14a9ce7e1090d57c9cfbe06dd753afe6fb7d) |
| [pashwinkrish22/laya-email-resume-classifier-claude](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-claude) | apache-2.0 | [860d27cfc1b3](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-claude/tree/860d27cfc1b38199a15d4424a467dc5d83770c00) |
| [pashwinkrish22/laya-email-resume-classifier-it](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-it) | apache-2.0 | [e999e486b996](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-it/tree/e999e486b99662515d6a5ce1f0157ca193015d04) |
| [pashwinkrish22/laya-email-resume-classifier-it1](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-it1) | apache-2.0 | [02c38f6433f2](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-it1/tree/02c38f6433f299ead661dd84a6132db0f7bc0edc) |
| [pashwinkrish22/laya-email-resume-classifier-multi](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-multi) | apache-2.0 | [db028a0ccbb7](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-multi/tree/db028a0ccbb73ebd2917dc1d62000a38e3f57b9b) |
| [pashwinkrish22/laya-email-resume-classifier-typed](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-typed) | apache-2.0 | [1565b8509fcd](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-typed/tree/1565b8509fcd7a346e61e4ba66ce406314c2310b) |
| [pashwinkrish22/laya-email-resume-classifier-v2](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-v2) | apache-2.0 | [af77bc2a05e0](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-v2/tree/af77bc2a05e0568621453fa0213afb1d1017af35) |
| [pashwinkrish22/laya-email-resume-classifier-v3](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-v3) | apache-2.0 | [b627ca7c8702](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-v3/tree/b627ca7c8702a647987cfa7482196d8a401ba6cb) |
| [pashwinkrish22/laya-email-resume-classifier-v4](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-v4) | apache-2.0 | [cdb554864a71](https://huggingface.co/pashwinkrish22/laya-email-resume-classifier-v4/tree/cdb554864a7167cb670d1639d076e85caa0772f1) |
| [peter2000/laya-vulnerability-groups](https://huggingface.co/peter2000/laya-vulnerability-groups) | apache-2.0 | [ffe575092a85](https://huggingface.co/peter2000/laya-vulnerability-groups/tree/ffe575092a856bc8d9b2a0502aaed578b94dbfa7) |
| [peter2000/laya-vulnerability-groups-v2](https://huggingface.co/peter2000/laya-vulnerability-groups-v2) | apache-2.0 | [5d4b2227fa63](https://huggingface.co/peter2000/laya-vulnerability-groups-v2/tree/5d4b2227fa63fbbe6a1ee4d67b2349b78ec04362) |
| [philipsubhan/laya-typed-decisions](https://huggingface.co/philipsubhan/laya-typed-decisions) | apache-2.0 | [aebac0f5a277](https://huggingface.co/philipsubhan/laya-typed-decisions/tree/aebac0f5a2770b48a5c3cd2b5fdf9ded9c6a4631) |
| [Phocinae/Phocinae-Largha-150M-v1](https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1) | apache-2.0 | [9a37d71465be](https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1/tree/9a37d71465be4b2abb8a64c5270008c08897cd33) |
| [piratehack009/laya-cn-finance-relevance](https://huggingface.co/piratehack009/laya-cn-finance-relevance) | not declared in Hub tags | [6ee4b120b3f9](https://huggingface.co/piratehack009/laya-cn-finance-relevance/tree/6ee4b120b3f9fbb155450a04527b5a02b836971d) |
| [piratehack009/laya-cn-flash-triage](https://huggingface.co/piratehack009/laya-cn-flash-triage) | not declared in Hub tags | [b4b6e63a242b](https://huggingface.co/piratehack009/laya-cn-flash-triage/tree/b4b6e63a242b442a405c0ab0d1be6ff7256f1e69) |
| [piratehack009/laya-cn-retail-sentiment](https://huggingface.co/piratehack009/laya-cn-retail-sentiment) | not declared in Hub tags | [67f2b467b10d](https://huggingface.co/piratehack009/laya-cn-retail-sentiment/tree/67f2b467b10da8ffe389d8eb78e46762e4a8276b) |
| [PIXELZX/XERON-0.1](https://huggingface.co/PIXELZX/XERON-0.1) | apache-2.0 | [a077ab80d30a](https://huggingface.co/PIXELZX/XERON-0.1/tree/a077ab80d30a6c4c595de7cf0b725fddf7dc07a8) |
| [PIXELZX/XERON-0.2](https://huggingface.co/PIXELZX/XERON-0.2) | apache-2.0 | [756131915e40](https://huggingface.co/PIXELZX/XERON-0.2/tree/756131915e40b1f04aa37b35e2281fd3b91bb6a9) |
| [PIXELZX/XERON-0.4](https://huggingface.co/PIXELZX/XERON-0.4) | apache-2.0 | [99a6e116250d](https://huggingface.co/PIXELZX/XERON-0.4/tree/99a6e116250dbd8ea2ce83341b402230d69efa01) |
| [Prasanna85/laya-issue-triage](https://huggingface.co/Prasanna85/laya-issue-triage) | apache-2.0 | [da82718ac473](https://huggingface.co/Prasanna85/laya-issue-triage/tree/da82718ac473b1b4148ec47b3dd7c5bd897d6bbf) |
| [Priceman614/x402-demand](https://huggingface.co/Priceman614/x402-demand) | apache-2.0 | [e187b8739664](https://huggingface.co/Priceman614/x402-demand/tree/e187b873966482c3dfcd8d6abeb48a8bb8f17ece) |
| [pythainlp/laya-multilingual](https://huggingface.co/pythainlp/laya-multilingual) | apache-2.0 | [1ce8695f73c7](https://huggingface.co/pythainlp/laya-multilingual/tree/1ce8695f73c7bd6f9cba003dc9164ea07489fca7) |
| [PythiaFinance/8ball](https://huggingface.co/PythiaFinance/8ball) | apache-2.0 | [76c4106458f7](https://huggingface.co/PythiaFinance/8ball/tree/76c4106458f708399211b709cc18d2df39ebcca3) |
| [Qalarc/Laya-GUT-Finetune](https://huggingface.co/Qalarc/Laya-GUT-Finetune) | cc-by-nc-4.0 | [2464dcd23ae2](https://huggingface.co/Qalarc/Laya-GUT-Finetune/tree/2464dcd23ae207fcad21ab1051db6ff743af4d09) |
| [Quantum08/laya-browser-mind2web](https://huggingface.co/Quantum08/laya-browser-mind2web) | apache-2.0 | [de7369e52665](https://huggingface.co/Quantum08/laya-browser-mind2web/tree/de7369e52665b53cf27bf17332d737090a0ae3da) |
| [RagedLSDD/laya-typed-decisions-finetuned](https://huggingface.co/RagedLSDD/laya-typed-decisions-finetuned) | apache-2.0 | [ccff91651b90](https://huggingface.co/RagedLSDD/laya-typed-decisions-finetuned/tree/ccff91651b9024a94bfe861e861cd998eac6ac4a) |
| [Ramg77/laya-sentiment-multilingual](https://huggingface.co/Ramg77/laya-sentiment-multilingual) | apache-2.0 | [c4a0187066fb](https://huggingface.co/Ramg77/laya-sentiment-multilingual/tree/c4a0187066fb766f4a6bc291974c2c4875f45fb6) |
| [Ramg77/laya-sentiment-multilingual-v2](https://huggingface.co/Ramg77/laya-sentiment-multilingual-v2) | apache-2.0 | [2d9135643595](https://huggingface.co/Ramg77/laya-sentiment-multilingual-v2/tree/2d913564359589e957358b2b28f4a18e5be48258) |
| [rao254/laya-manual-finetuned](https://huggingface.co/rao254/laya-manual-finetuned) | not declared in Hub tags | [754500bb451e](https://huggingface.co/rao254/laya-manual-finetuned/tree/754500bb451e6ad8f472cc64ca583c3665505e76) |
| [rayss868123/laya](https://huggingface.co/rayss868123/laya) | apache-2.0 | [835c76eb49a7](https://huggingface.co/rayss868123/laya/tree/835c76eb49a7e91ba9d9378524eacc28a1dd347b) |
| [rayss868123/laya-multilingual](https://huggingface.co/rayss868123/laya-multilingual) | apache-2.0 | [c2e7b65878b3](https://huggingface.co/rayss868123/laya-multilingual/tree/c2e7b65878b3cc2ee4ac54321ded62edf56f6473) |
| [rayss868123/laya-typed-decisions](https://huggingface.co/rayss868123/laya-typed-decisions) | apache-2.0 | [9b48a64847f5](https://huggingface.co/rayss868123/laya-typed-decisions/tree/9b48a64847f55ed79cb94d23668792b00557e20f) |
| [rehman-ali/laya-tetris](https://huggingface.co/rehman-ali/laya-tetris) | apache-2.0 | [e3361fd5930c](https://huggingface.co/rehman-ali/laya-tetris/tree/e3361fd5930c211ca9671ce6d78844fa241117a0) |
| [rehman-ali/laya-tetris-4line](https://huggingface.co/rehman-ali/laya-tetris-4line) | apache-2.0 | [0431971fbdd2](https://huggingface.co/rehman-ali/laya-tetris-4line/tree/0431971fbdd2e2ae55d53ea5f62b84d0fecd1be2) |
| [ricardocs99/attributed-decision-decider](https://huggingface.co/ricardocs99/attributed-decision-decider) | cc-by-nc-4.0 | [228fa0122bb0](https://huggingface.co/ricardocs99/attributed-decision-decider/tree/228fa0122bb0421b38ec2511ce57b8d7f1c95f38) |
| [ricky136/laya-chat](https://huggingface.co/ricky136/laya-chat) | apache-2.0 | [4f8180a7ae61](https://huggingface.co/ricky136/laya-chat/tree/4f8180a7ae6118c19dfb1e00b4cb6a1d9b80c789) |
| [ringwring/laya-vn](https://huggingface.co/ringwring/laya-vn) | apache-2.0 | [d5cdbca2585f](https://huggingface.co/ringwring/laya-vn/tree/d5cdbca2585f844fbac040c5c2d360c1f6904d5e) |
| [roadius/peewee-mix-v1](https://huggingface.co/roadius/peewee-mix-v1) | apache-2.0 | [7df5abd45dbf](https://huggingface.co/roadius/peewee-mix-v1/tree/7df5abd45dbfa6d5be3cf13175de0b35dd70bc10) |
| [RodrigoBrasil2024/laya-concessionaria-pt](https://huggingface.co/RodrigoBrasil2024/laya-concessionaria-pt) | apache-2.0 | [1797597061f4](https://huggingface.co/RodrigoBrasil2024/laya-concessionaria-pt/tree/1797597061f4aa8647acd84f6ce6d942e232899e) |
| [RomanRG008/gyra](https://huggingface.co/RomanRG008/gyra) | apache-2.0 | [cd00e8df36b9](https://huggingface.co/RomanRG008/gyra/tree/cd00e8df36b98e5d9e649cc19aede5170c521655) |
| [ruddh/agent-airlock-laya](https://huggingface.co/ruddh/agent-airlock-laya) | not declared in Hub tags | [afc063143a2a](https://huggingface.co/ruddh/agent-airlock-laya/tree/afc063143a2a0a17028761e9f1676a20e8a403d6) |
| [S4MPL3BI4S/Coding_Decision_Agent](https://huggingface.co/S4MPL3BI4S/Coding_Decision_Agent) | apache-2.0 | [fbc40c2e4b62](https://huggingface.co/S4MPL3BI4S/Coding_Decision_Agent/tree/fbc40c2e4b625519de1c1f617724449a636d08ff) |
| [safecircleai/vigil](https://huggingface.co/safecircleai/vigil) | other | [adbdb5893d1d](https://huggingface.co/safecircleai/vigil/tree/adbdb5893d1da484b0dee6a0b913a6b5e1eacc18) |
| [SAIFIINDUSTRIES/laya](https://huggingface.co/SAIFIINDUSTRIES/laya) | apache-2.0 | [2ba530c97c3e](https://huggingface.co/SAIFIINDUSTRIES/laya/tree/2ba530c97c3ef1937eeea6a1282aedd533d10217) |
| [salabh-an/laya-typed-decisions](https://huggingface.co/salabh-an/laya-typed-decisions) | apache-2.0 | [4363f4c8417d](https://huggingface.co/salabh-an/laya-typed-decisions/tree/4363f4c8417da865db0c572887ddcfc1e05aabe3) |
| [sallout/laya-multitask-intents](https://huggingface.co/sallout/laya-multitask-intents) | apache-2.0 | [a8fbb6320fed](https://huggingface.co/sallout/laya-multitask-intents/tree/a8fbb6320fedb1b676256a6efe5ac75fdca7e37d) |
| [saurabh1112/laya-intent-classifier](https://huggingface.co/saurabh1112/laya-intent-classifier) | not declared in Hub tags | [fc6fb94afeae](https://huggingface.co/saurabh1112/laya-intent-classifier/tree/fc6fb94afeae821ed0bf1cd99c1ff79d875a3328) |
| [scottrlamkin/laya-multilingual](https://huggingface.co/scottrlamkin/laya-multilingual) | apache-2.0 | [c5283e910a9c](https://huggingface.co/scottrlamkin/laya-multilingual/tree/c5283e910a9c19f6c9e7126bcc59f6c31721f265) |
| [sekkit/laya](https://huggingface.co/sekkit/laya) | apache-2.0 | [97f34fefc62b](https://huggingface.co/sekkit/laya/tree/97f34fefc62be9bf1170a48f408d6efb81c19cf2) |
| [SGK86/laya-multilingual-typed-decisions](https://huggingface.co/SGK86/laya-multilingual-typed-decisions) | apache-2.0 | [8fc29deeba77](https://huggingface.co/SGK86/laya-multilingual-typed-decisions/tree/8fc29deeba77c74004756dfd64217287b1e52e47) |
| [shakeel143/jev-multilingual](https://huggingface.co/shakeel143/jev-multilingual) | apache-2.0 | [1dd32a45462e](https://huggingface.co/shakeel143/jev-multilingual/tree/1dd32a45462e7e7db5b87abaf579f4ed767d05a3) |
| [shanaka95/laya-fintiment](https://huggingface.co/shanaka95/laya-fintiment) | apache-2.0 | [a90e51273cb8](https://huggingface.co/shanaka95/laya-fintiment/tree/a90e51273cb872e370f77383731a9976e806f2ce) |
| [Silversys/laya-eurusd](https://huggingface.co/Silversys/laya-eurusd) | not declared in Hub tags | [0d7042ec8e81](https://huggingface.co/Silversys/laya-eurusd/tree/0d7042ec8e81e0a72cce41116a6c08e9d576335f) |
| [sir-danewhitman/laya-town-ft](https://huggingface.co/sir-danewhitman/laya-town-ft) | not declared in Hub tags | [be932febba3e](https://huggingface.co/sir-danewhitman/laya-town-ft/tree/be932febba3e39edb30caeaf39efd4b48c2e9a51) |
| [sivasub987/mandate-1-laya](https://huggingface.co/sivasub987/mandate-1-laya) | apache-2.0 | [5039fc2e116e](https://huggingface.co/sivasub987/mandate-1-laya/tree/5039fc2e116e9e2636be13d4808f4d3621b405ab) |
| [skm14747/custom_lid_lsd](https://huggingface.co/skm14747/custom_lid_lsd) | apache-2.0 | [2d85dd45bb81](https://huggingface.co/skm14747/custom_lid_lsd/tree/2d85dd45bb813ae053903d4771c84ea9979c843d) |
| [slow-stack/laya-nli-conflict-v10-l1](https://huggingface.co/slow-stack/laya-nli-conflict-v10-l1) | apache-2.0 | [ee9bf5111f0e](https://huggingface.co/slow-stack/laya-nli-conflict-v10-l1/tree/ee9bf5111f0e94de8207171f5f485294ec011b12) |
| [slow-stack/laya-nli-conflict-v10-l2](https://huggingface.co/slow-stack/laya-nli-conflict-v10-l2) | apache-2.0 | [3c054b2842a7](https://huggingface.co/slow-stack/laya-nli-conflict-v10-l2/tree/3c054b2842a71436b20d71de7540fc28049ba997) |
| [slow-stack/laya-nli-conflict-v10-s2](https://huggingface.co/slow-stack/laya-nli-conflict-v10-s2) | apache-2.0 | [3d28f96cb6e9](https://huggingface.co/slow-stack/laya-nli-conflict-v10-s2/tree/3d28f96cb6e9597732f29c8b7f1bc07c1cef2015) |
| [slow-stack/laya-nli-conflict-v12](https://huggingface.co/slow-stack/laya-nli-conflict-v12) | apache-2.0 | [f48a0a0d5ce7](https://huggingface.co/slow-stack/laya-nli-conflict-v12/tree/f48a0a0d5ce730b979d5f609cd935a9882ca3b00) |
| [slow-stack/laya-nli-conflict-v12-r1](https://huggingface.co/slow-stack/laya-nli-conflict-v12-r1) | apache-2.0 | [fb7655d67bb3](https://huggingface.co/slow-stack/laya-nli-conflict-v12-r1/tree/fb7655d67bb355b7134e43daf4427d2629e04680) |
| [slow-stack/laya-nli-conflict-v12-r3](https://huggingface.co/slow-stack/laya-nli-conflict-v12-r3) | apache-2.0 | [a6d5b2679855](https://huggingface.co/slow-stack/laya-nli-conflict-v12-r3/tree/a6d5b26798552c2610c9818f0017028f0ef642e0) |
| [slow-stack/laya-nli-conflict-v5](https://huggingface.co/slow-stack/laya-nli-conflict-v5) | apache-2.0 | [c4d7c7b38cf6](https://huggingface.co/slow-stack/laya-nli-conflict-v5/tree/c4d7c7b38cf639a44ef3a70774469f18e5039c78) |
| [slow-stack/laya-nli-conflict-v5-ce](https://huggingface.co/slow-stack/laya-nli-conflict-v5-ce) | apache-2.0 | [694a9f1de26d](https://huggingface.co/slow-stack/laya-nli-conflict-v5-ce/tree/694a9f1de26d8139f93f854a78b13d22cb728cfa) |
| [slow-stack/laya-nli-conflict-v6](https://huggingface.co/slow-stack/laya-nli-conflict-v6) | apache-2.0 | [36c0b3b18659](https://huggingface.co/slow-stack/laya-nli-conflict-v6/tree/36c0b3b1865936dc6a7bc543807b9bdd2f694b92) |
| [slow-stack/laya-nli-conflict-v7](https://huggingface.co/slow-stack/laya-nli-conflict-v7) | apache-2.0 | [f8917ea1dda5](https://huggingface.co/slow-stack/laya-nli-conflict-v7/tree/f8917ea1dda5f647ac0adc066a676fecd2b494f3) |
| [slow-stack/laya-nli-conflict-v8](https://huggingface.co/slow-stack/laya-nli-conflict-v8) | apache-2.0 | [5c7e7bf3076c](https://huggingface.co/slow-stack/laya-nli-conflict-v8/tree/5c7e7bf3076c309c14e7f9795488f45947063c83) |
| [slow-stack/laya-nli-conflict-v9](https://huggingface.co/slow-stack/laya-nli-conflict-v9) | apache-2.0 | [c9cbad30b18f](https://huggingface.co/slow-stack/laya-nli-conflict-v9/tree/c9cbad30b18f124a62f98d647c933e2082df577f) |
| [slow-stack/laya-nli-memory-conflict](https://huggingface.co/slow-stack/laya-nli-memory-conflict) | apache-2.0 | [4d016a562565](https://huggingface.co/slow-stack/laya-nli-memory-conflict/tree/4d016a56256595fd966546af0ec90d78b6fd6483) |
| [slow-stack/laya-typed-decisions-multilingual](https://huggingface.co/slow-stack/laya-typed-decisions-multilingual) | apache-2.0 | [8de52d4d7ec9](https://huggingface.co/slow-stack/laya-typed-decisions-multilingual/tree/8de52d4d7ec96423639882ca6faa93490fb81c90) |
| [smolnikov/kivok-0.3b](https://huggingface.co/smolnikov/kivok-0.3b) | apache-2.0 | [5739091664dc](https://huggingface.co/smolnikov/kivok-0.3b/tree/5739091664dc79eaff3c76396d29279e6cd893d5) |
| [SofiTesfay2010/Alpha](https://huggingface.co/SofiTesfay2010/Alpha) | apache-2.0 | [af6e981db0ca](https://huggingface.co/SofiTesfay2010/Alpha/tree/af6e981db0ca2a84159b97ca3dfbe6202d887d17) |
| [sothiem/laya-kvp10k-noul](https://huggingface.co/sothiem/laya-kvp10k-noul) | apache-2.0 | [2ea229cfb3ec](https://huggingface.co/sothiem/laya-kvp10k-noul/tree/2ea229cfb3ecf056371e2a7c8d656db43393d920) |
| [Suarify/laya-pdpa-my-v3](https://huggingface.co/Suarify/laya-pdpa-my-v3) | apache-2.0 | [c88002a9ba10](https://huggingface.co/Suarify/laya-pdpa-my-v3/tree/c88002a9ba10658a7ed70e12f02773f2708d4c86) |
| [Suarify/laya-pdpa-my-v4](https://huggingface.co/Suarify/laya-pdpa-my-v4) | apache-2.0 | [314b5a365f3f](https://huggingface.co/Suarify/laya-pdpa-my-v4/tree/314b5a365f3fa641cf79322b8ec52c3c80e495e1) |
| [SwagMessiah100/laya](https://huggingface.co/SwagMessiah100/laya) | apache-2.0 | [9a08077dcab5](https://huggingface.co/SwagMessiah100/laya/tree/9a08077dcab5588508ea33361c5be7982767cd37) |
| [sxtylws/laya-multilingual](https://huggingface.co/sxtylws/laya-multilingual) | apache-2.0 | [2a2b020127ef](https://huggingface.co/sxtylws/laya-multilingual/tree/2a2b020127efaffc3d849c2392f3d24c0a726d1d) |
| [syvai/claes-preview](https://huggingface.co/syvai/claes-preview) | apache-2.0 | [85bfaef5d5f7](https://huggingface.co/syvai/claes-preview/tree/85bfaef5d5f7c81d0efcb9c6b72ca6c8ee165d17) |
| [tampajohn/laya-stop-completion-judge](https://huggingface.co/tampajohn/laya-stop-completion-judge) | apache-2.0 | [c1931d4ffbb9](https://huggingface.co/tampajohn/laya-stop-completion-judge/tree/c1931d4ffbb90c3584ef936b5d8eae7bbe8c82da) |
| [tarfandoon/laya-fa-support](https://huggingface.co/tarfandoon/laya-fa-support) | apache-2.0 | [c01b8da2ef8c](https://huggingface.co/tarfandoon/laya-fa-support/tree/c01b8da2ef8c4be27b2251dfe26338eb8472bae0) |
| [team-od/laya-thai](https://huggingface.co/team-od/laya-thai) | cc-by-nc-4.0 | [85fe58c9c763](https://huggingface.co/team-od/laya-thai/tree/85fe58c9c7638d55b8708d4c7261b64d076db389) |
| [Tejassst/orcus-laya](https://huggingface.co/Tejassst/orcus-laya) | not declared in Hub tags | [c40483b5c9d5](https://huggingface.co/Tejassst/orcus-laya/tree/c40483b5c9d57c5c7901d1c16d97cb2be72a61eb) |
| [telepatia-ai/laya-pt-es-nli](https://huggingface.co/telepatia-ai/laya-pt-es-nli) | apache-2.0 | [71459abd38f4](https://huggingface.co/telepatia-ai/laya-pt-es-nli/tree/71459abd38f4e439b2f4e3a2dae87573ae40c77e) |
| [telepatia-ai/laya-pt-es-typed](https://huggingface.co/telepatia-ai/laya-pt-es-typed) | apache-2.0 | [0f23acdf2c43](https://huggingface.co/telepatia-ai/laya-pt-es-typed/tree/0f23acdf2c4325107ebb20e0b12886b39da41734) |
| [Terom/laya](https://huggingface.co/Terom/laya) | apache-2.0 | [ea577a6c2f80](https://huggingface.co/Terom/laya/tree/ea577a6c2f80515aa476fd9f98eb01a741cc760e) |
| [TextCortex/laya-cybersec](https://huggingface.co/TextCortex/laya-cybersec) | other | [41fd5a2dfced](https://huggingface.co/TextCortex/laya-cybersec/tree/41fd5a2dfced0fb357974afd780a3ed6e7c43838) |
| [TextCortex/raya](https://huggingface.co/TextCortex/raya) | apache-2.0 | [48c8658436c2](https://huggingface.co/TextCortex/raya/tree/48c8658436c268163f31720dd6c40165894e4093) |
| [thealper2/laya-multilingual-turkish-noul](https://huggingface.co/thealper2/laya-multilingual-turkish-noul) | cc-by-sa-4.0 | [10132570b40b](https://huggingface.co/thealper2/laya-multilingual-turkish-noul/tree/10132570b40b37d5de7195c2e6e1377972f2fb4b) |
| [thechristyjo/laya-model-routing](https://huggingface.co/thechristyjo/laya-model-routing) | apache-2.0 | [438e8e196e5b](https://huggingface.co/thechristyjo/laya-model-routing/tree/438e8e196e5ba7f568effa0f2127e1d8722414c3) |
| [thechristyjo/laya-model-routing-1000](https://huggingface.co/thechristyjo/laya-model-routing-1000) | apache-2.0 | [77eae7a2d67d](https://huggingface.co/thechristyjo/laya-model-routing-1000/tree/77eae7a2d67d56a205090bb7ff5c23ae27149330) |
| [thechristyjo/laya-model-routing-1000e12](https://huggingface.co/thechristyjo/laya-model-routing-1000e12) | apache-2.0 | [bcef0143dbb3](https://huggingface.co/thechristyjo/laya-model-routing-1000e12/tree/bcef0143dbb3a6643f2c37b9d362d9cb9ed8546d) |
| [thechristyjo/laya-model-routing-1000hpo](https://huggingface.co/thechristyjo/laya-model-routing-1000hpo) | apache-2.0 | [bf90ba3c9d04](https://huggingface.co/thechristyjo/laya-model-routing-1000hpo/tree/bf90ba3c9d04db9088f598eb4c91f6fa6f6daa2f) |
| [thechristyjo/laya-model-routing-500](https://huggingface.co/thechristyjo/laya-model-routing-500) | apache-2.0 | [22276ec7d720](https://huggingface.co/thechristyjo/laya-model-routing-500/tree/22276ec7d7204631adea64ae63562396df76ee48) |
| [thechristyjo/laya-model-routing-500e12](https://huggingface.co/thechristyjo/laya-model-routing-500e12) | apache-2.0 | [34d97e643006](https://huggingface.co/thechristyjo/laya-model-routing-500e12/tree/34d97e643006e14c41ac69003548fb3426e84b0e) |
| [Themrgndz/laya-erp-kategori](https://huggingface.co/Themrgndz/laya-erp-kategori) | apache-2.0 | [bd439350980f](https://huggingface.co/Themrgndz/laya-erp-kategori/tree/bd439350980f0df9f99d00579e110f0e2c726c62) |
| [TheWiderLensInitiative/laya-for-clara](https://huggingface.co/TheWiderLensInitiative/laya-for-clara) | apache-2.0 | [07ea0f3ee3e9](https://huggingface.co/TheWiderLensInitiative/laya-for-clara/tree/07ea0f3ee3e920d2a3c87b8f6d4a09ae14001ac6) |
| [tindang/laya-code](https://huggingface.co/tindang/laya-code) | apache-2.0 | [f3d6bd2344e4](https://huggingface.co/tindang/laya-code/tree/f3d6bd2344e4750dd917f95d40ceacfa81bb81db) |
| [tiuras/laya-ghostpad-v3](https://huggingface.co/tiuras/laya-ghostpad-v3) | apache-2.0 | [3e328bfcc613](https://huggingface.co/tiuras/laya-ghostpad-v3/tree/3e328bfcc613e4c4f8ef9699895b9261840885d1) |
| [TracNetwork/mayhem-catalog-convaiinnovations-laya](https://huggingface.co/TracNetwork/mayhem-catalog-convaiinnovations-laya) | apache-2.0 | [d288cbfe560a](https://huggingface.co/TracNetwork/mayhem-catalog-convaiinnovations-laya/tree/d288cbfe560a0ff904401f57e28820aa140c11e7) |
| [UNUSABLE07/laya](https://huggingface.co/UNUSABLE07/laya) | apache-2.0 | [aae24a455138](https://huggingface.co/UNUSABLE07/laya/tree/aae24a455138ea985dcced96375ef1fb3720814d) |
| [vanty120/laya-vi](https://huggingface.co/vanty120/laya-vi) | apache-2.0 | [860b2b5f145a](https://huggingface.co/vanty120/laya-vi/tree/860b2b5f145a1013161dcedd93fda7c263a3c9f0) |
| [vdaular/laya](https://huggingface.co/vdaular/laya) | apache-2.0 | [c7b09e9790b9](https://huggingface.co/vdaular/laya/tree/c7b09e9790b9b52efcf1f4599fccd56bfedc7156) |
| [Veroic/laya_test](https://huggingface.co/Veroic/laya_test) | apache-2.0 | [ccaeb57eb801](https://huggingface.co/Veroic/laya_test/tree/ccaeb57eb801b59b3e56b308b8ef8df23eaf305a) |
| [Vinay57/laya-jigsaw-moderation](https://huggingface.co/Vinay57/laya-jigsaw-moderation) | apache-2.0 | [9e558ae55944](https://huggingface.co/Vinay57/laya-jigsaw-moderation/tree/9e558ae5594479efd78d35660d12ea27444ce075) |
| [winwinwinbb/soft-decider-421m](https://huggingface.co/winwinwinbb/soft-decider-421m) | apache-2.0 | [ad67f6b0b178](https://huggingface.co/winwinwinbb/soft-decider-421m/tree/ad67f6b0b178a726a399919f5f6d443e1f772890) |
| [Wouze/laya-ara](https://huggingface.co/Wouze/laya-ara) | other | [c2a0118ce668](https://huggingface.co/Wouze/laya-ara/tree/c2a0118ce6688e24099feca80643b4075120751c) |
| [Wouze/laya-ara-rag](https://huggingface.co/Wouze/laya-ara-rag) | apache-2.0 | [731b39b7b264](https://huggingface.co/Wouze/laya-ara-rag/tree/731b39b7b264accab86c3cde06beec5f42d023b4) |
| [wuyouxiaobai/laya](https://huggingface.co/wuyouxiaobai/laya) | apache-2.0 | [a93f3c6d6417](https://huggingface.co/wuyouxiaobai/laya/tree/a93f3c6d6417a67f7bfc8f5be110c4fc8fe32b75) |
| [xamjiang/laya-burger](https://huggingface.co/xamjiang/laya-burger) | other | [0c79507c8ec3](https://huggingface.co/xamjiang/laya-burger/tree/0c79507c8ec3514398cd513abe281572c8d02761) |
| [yashchouhann/laya-typed-decisions](https://huggingface.co/yashchouhann/laya-typed-decisions) | apache-2.0 | [b01a1a413a70](https://huggingface.co/yashchouhann/laya-typed-decisions/tree/b01a1a413a70ee67ae2bdd7d6c21415f0b933201) |
| [yasserrmd/enterprise-reflux-laya](https://huggingface.co/yasserrmd/enterprise-reflux-laya) | apache-2.0 | [60bfe695199f](https://huggingface.co/yasserrmd/enterprise-reflux-laya/tree/60bfe695199f48e8eb06a263c1b49a02cc7144a7) |
| [yasserrmd/enterprise-reflux-laya-v21](https://huggingface.co/yasserrmd/enterprise-reflux-laya-v21) | apache-2.0 | [75c4407d03c0](https://huggingface.co/yasserrmd/enterprise-reflux-laya-v21/tree/75c4407d03c04aa83be690e57f36d62466fb4589) |
| [yxnxc/laya-claim-detection](https://huggingface.co/yxnxc/laya-claim-detection) | apache-2.0 | [13a535703a8f](https://huggingface.co/yxnxc/laya-claim-detection/tree/13a535703a8f73ed6b9011ed149195882a0c6d14) |
| [Zamax14/laya-context-prefilter](https://huggingface.co/Zamax14/laya-context-prefilter) | apache-2.0 | [a22e443c55d1](https://huggingface.co/Zamax14/laya-context-prefilter/tree/a22e443c55d11351df59ce961ec3b9fe1e4b4571) |
| [ZeraG07/laya-vision](https://huggingface.co/ZeraG07/laya-vision) | cc-by-nc-sa-4.0 | [54af45d3930d](https://huggingface.co/ZeraG07/laya-vision/tree/54af45d3930d4d3a368279a43a0acbc8204eaedb) |
| [zsla-oc/laya-frogger](https://huggingface.co/zsla-oc/laya-frogger) | apache-2.0 | [7f4c21a88b5e](https://huggingface.co/zsla-oc/laya-frogger/tree/7f4c21a88b5e0ca4cec2a0742022519d94a8a235) |

### F2: Kev-compatible checkpoints

| Repository | Declared license | Checked revision |
|---|---|---|
| [abhash-chakraborty/kev-mem-0.8b](https://huggingface.co/abhash-chakraborty/kev-mem-0.8b) | apache-2.0 | [4006a88f1fea](https://huggingface.co/abhash-chakraborty/kev-mem-0.8b/tree/4006a88f1fea39fbcf335291369f7f3d8f00ab9e) |
| [ai-ecoverse/kev-0.8b-vision-intent](https://huggingface.co/ai-ecoverse/kev-0.8b-vision-intent) | openrail | [8c319760331b](https://huggingface.co/ai-ecoverse/kev-0.8b-vision-intent/tree/8c319760331b1c7dbb33538b1ad94569f00f7d02) |
| [ai-ecoverse/kev-4b-vision-intent](https://huggingface.co/ai-ecoverse/kev-4b-vision-intent) | openrail | [f51a4bbe348d](https://huggingface.co/ai-ecoverse/kev-4b-vision-intent/tree/f51a4bbe348d5df82cef011b18a13da931c706e0) |
| [AIMultiple/kev-9b-browser-ft](https://huggingface.co/AIMultiple/kev-9b-browser-ft) | apache-2.0 | [8a3162f0759c](https://huggingface.co/AIMultiple/kev-9b-browser-ft/tree/8a3162f0759c258f2a3166fe48e325d4a8ef686e) |
| [akshat-OwO/nev-0.8b](https://huggingface.co/akshat-OwO/nev-0.8b) | apache-2.0 | [2cdfac694d3d](https://huggingface.co/akshat-OwO/nev-0.8b/tree/2cdfac694d3d61fe9ef1a5156ff8d521930ca350) |
| [anish-shrestha/qwen3.5-0.8b-decision-distill-4b](https://huggingface.co/anish-shrestha/qwen3.5-0.8b-decision-distill-4b) | other | [4c3b8314e261](https://huggingface.co/anish-shrestha/qwen3.5-0.8b-decision-distill-4b/tree/4c3b8314e26106ada4bb8a9978776e3171c12f8b) |
| [anish-shrestha/qwen3.5-0.8b-decision-gold-control](https://huggingface.co/anish-shrestha/qwen3.5-0.8b-decision-gold-control) | other | [ef5fd88b12df](https://huggingface.co/anish-shrestha/qwen3.5-0.8b-decision-gold-control/tree/ef5fd88b12df85be72d30de99bfea416a561d1d6) |
| [BharathGowda/kev-gym-4b](https://huggingface.co/BharathGowda/kev-gym-4b) | apache-2.0 | [639181b2634d](https://huggingface.co/BharathGowda/kev-gym-4b/tree/639181b2634d80f768f710bfa5beeae591c97244) |
| [dhtocks/malkuth-2b](https://huggingface.co/dhtocks/malkuth-2b) | cc-by-nc-4.0 | [37ae1672eebc](https://huggingface.co/dhtocks/malkuth-2b/tree/37ae1672eebcf9ee5417418bd0be49f2e53d9087) |
| [dhtocks/malkuth-4b](https://huggingface.co/dhtocks/malkuth-4b) | cc-by-nc-4.0 | [ba9221fcb1ff](https://huggingface.co/dhtocks/malkuth-4b/tree/ba9221fcb1ff450fc51dcefffc8e034ed479630d) |
| [felhen-ai/saracura-ptbr-4b](https://huggingface.co/felhen-ai/saracura-ptbr-4b) | apache-2.0 | [9b27a7be153a](https://huggingface.co/felhen-ai/saracura-ptbr-4b/tree/9b27a7be153afee62b778b61eb5d664ce978d3b1) |
| [fzkun/kev-queue-v4-4b](https://huggingface.co/fzkun/kev-queue-v4-4b) | apache-2.0 | [a8e77661d5cb](https://huggingface.co/fzkun/kev-queue-v4-4b/tree/a8e77661d5cb5324475336c91d25c6466c69d897) |
| [Glax147/kev-0.8b-ba-lora](https://huggingface.co/Glax147/kev-0.8b-ba-lora) | apache-2.0 | [6f3864febcef](https://huggingface.co/Glax147/kev-0.8b-ba-lora/tree/6f3864febcef263cd5d520d5520c0c5aa92472f0) |
| [Glax147/kev-4b-ba-lora](https://huggingface.co/Glax147/kev-4b-ba-lora) | apache-2.0 | [3d5932a57af8](https://huggingface.co/Glax147/kev-4b-ba-lora/tree/3d5932a57af805c36e3ce51256316997b8494e31) |
| [gyung/Qwev-9B-RLCD](https://huggingface.co/gyung/Qwev-9B-RLCD) | apache-2.0 | [2d2595d297a7](https://huggingface.co/gyung/Qwev-9B-RLCD/tree/2d2595d297a7258fce18dd4f492398fa20fe2e0e) |
| [Jacqkues/sev-0.8b](https://huggingface.co/Jacqkues/sev-0.8b) | other | [958c1835fef5](https://huggingface.co/Jacqkues/sev-0.8b/tree/958c1835fef51892ffe7283fec9bf6495a8f8fdc) |
| [Jakevin/taiwan-mainland-wording-kev-4b](https://huggingface.co/Jakevin/taiwan-mainland-wording-kev-4b) | apache-2.0 | [a442a4c7201c](https://huggingface.co/Jakevin/taiwan-mainland-wording-kev-4b/tree/a442a4c7201c2d338dfaf9df1a9f9e1c9ebe7e89) |
| [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | apache-2.0 | [bf75a6a8848e](https://huggingface.co/jaredpalmer/kev-0.8b/tree/bf75a6a8848ea6960ff2ed108d9ed44c2941174f) |
| [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | apache-2.0 | [6cfce5c2fa4b](https://huggingface.co/jaredpalmer/kev-4b/tree/6cfce5c2fa4b4bd64026336ab649c5ca78857d52) |
| [jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b) | apache-2.0 | [db029f08b290](https://huggingface.co/jaredpalmer/kev-9b/tree/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0) |
| [jchapuis/kev-4b-spine-scala](https://huggingface.co/jchapuis/kev-4b-spine-scala) | apache-2.0 | [e684fc9e16c9](https://huggingface.co/jchapuis/kev-4b-spine-scala/tree/e684fc9e16c96c993284aac0573cd6ff84708bef) |
| [joshnavardhan/kev-cert-r42-v3](https://huggingface.co/joshnavardhan/kev-cert-r42-v3) | not declared in Hub tags | [3be344d8203b](https://huggingface.co/joshnavardhan/kev-cert-r42-v3/tree/3be344d8203bb676ac2b8c9f4688b7c07f0368f8) |
| [jtatman/kev-0.8b-code-verify-v1](https://huggingface.co/jtatman/kev-0.8b-code-verify-v1) | apache-2.0 | [34ac4c100405](https://huggingface.co/jtatman/kev-0.8b-code-verify-v1/tree/34ac4c100405e6e07aaea4000f50dcef686f6de8) |
| [jtatman/kev-0.8b-code-verify-v2](https://huggingface.co/jtatman/kev-0.8b-code-verify-v2) | apache-2.0 | [a569b68288bf](https://huggingface.co/jtatman/kev-0.8b-code-verify-v2/tree/a569b68288bfd426bc662f872aee13f4e961db80) |
| [jtatman/kev-4b-code-verify-v1](https://huggingface.co/jtatman/kev-4b-code-verify-v1) | apache-2.0 | [37d06d265761](https://huggingface.co/jtatman/kev-4b-code-verify-v1/tree/37d06d2657619f9fefc16e5e7d1ca57f8693decd) |
| [jtatman/kev-4b-code-verify-v2](https://huggingface.co/jtatman/kev-4b-code-verify-v2) | apache-2.0 | [6ac368e31103](https://huggingface.co/jtatman/kev-4b-code-verify-v2/tree/6ac368e3110356f21c54e18733f8526533412e3d) |
| [junma/MedJev-Qwen3.5-0.8B](https://huggingface.co/junma/MedJev-Qwen3.5-0.8B) | apache-2.0 | [e2eb4054f440](https://huggingface.co/junma/MedJev-Qwen3.5-0.8B/tree/e2eb4054f4406c15aab3874333babdb736738d50) |
| [macmacmacmac/Sev-4B](https://huggingface.co/macmacmacmac/Sev-4B) | apache-2.0 | [7f9f3f53b9ba](https://huggingface.co/macmacmacmac/Sev-4B/tree/7f9f3f53b9baae041c64a71d4b3867c6700fd524) |
| [marcmagn1/kev-08b-typed-v1](https://huggingface.co/marcmagn1/kev-08b-typed-v1) | not declared in Hub tags | [ad7a66f76162](https://huggingface.co/marcmagn1/kev-08b-typed-v1/tree/ad7a66f76162558e84f860ffea158f5ccf7aefa0) |
| [marcmagn1/kev-08b-typed-v1-blend](https://huggingface.co/marcmagn1/kev-08b-typed-v1-blend) | not declared in Hub tags | [00cda0cb8cab](https://huggingface.co/marcmagn1/kev-08b-typed-v1-blend/tree/00cda0cb8cabe3344c931e91d1aefc1abb48a26d) |
| [marcmagn1/kev-4b-typed-v1](https://huggingface.co/marcmagn1/kev-4b-typed-v1) | not declared in Hub tags | [d4653225287e](https://huggingface.co/marcmagn1/kev-4b-typed-v1/tree/d4653225287e8600d8baeeef83579cf922fb54eb) |
| [per2021/kev-2b-qwen3.5](https://huggingface.co/per2021/kev-2b-qwen3.5) | apache-2.0 | [ac97885b5401](https://huggingface.co/per2021/kev-2b-qwen3.5/tree/ac97885b54011dd6002cf2c82fbcc0aa6033e8cf) |
| [Risethagain/rippy-kev-0.8b](https://huggingface.co/Risethagain/rippy-kev-0.8b) | apache-2.0 | [281cbcbe78c1](https://huggingface.co/Risethagain/rippy-kev-0.8b/tree/281cbcbe78c101936c478b25dab2f293b069cdae) |
| [Risethagain/rippy-kev-4b](https://huggingface.co/Risethagain/rippy-kev-4b) | apache-2.0 | [8a1a0bbd7805](https://huggingface.co/Risethagain/rippy-kev-4b/tree/8a1a0bbd7805e550d8bcf7ec876876632d96b6bd) |
| [SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA) | not declared in Hub tags | [4c7fd2539994](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA/tree/4c7fd2539994f37f5db3130704f528f011e448d1) |
| [SimpleJev/JevAny-Qwen3.5-4B-LoRA](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA) | not declared in Hub tags | [1c7aa9bab14a](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA/tree/1c7aa9bab14ac347aeb917c0bcd757838a8a78ce) |
| [SimpleJev/JevAny-Qwen3.8-27B-LoRA](https://huggingface.co/SimpleJev/JevAny-Qwen3.8-27B-LoRA) | not declared in Hub tags | [09c9e9102d5b](https://huggingface.co/SimpleJev/JevAny-Qwen3.8-27B-LoRA/tree/09c9e9102d5b8cc7d56558d25da1202a761b6c0d) |
| [YAlgoG/parakh-4b](https://huggingface.co/YAlgoG/parakh-4b) | cc-by-nc-4.0 | [e98052d583d8](https://huggingface.co/YAlgoG/parakh-4b/tree/e98052d583d889bfa0912cfdb042722302773210) |

### F3: Nimble-compatible checkpoints

| Repository | Declared license | Checked revision |
|---|---|---|
| [bespokelabs/Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) | apache-2.0 | [bd792f44ec8e](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B/tree/bd792f44ec8e265be861bfcdf4e05967ffe0e858) |
| [bespokelabs/Bespoke-Nimble-9B-v2](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) | apache-2.0 | [4b8c04d1ac2c](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2/tree/4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe) |
| [helmo/Bespoke-Nimble-9B](https://huggingface.co/helmo/Bespoke-Nimble-9B) | apache-2.0 | [5c653a567905](https://huggingface.co/helmo/Bespoke-Nimble-9B/tree/5c653a56790542acfbbe4e8a6cc42c6f21bfbc8c) |
| [helmo/Bespoke-Nimble-9B-v2](https://huggingface.co/helmo/Bespoke-Nimble-9B-v2) | apache-2.0 | [7481c58d235d](https://huggingface.co/helmo/Bespoke-Nimble-9B-v2/tree/7481c58d235d032b9588a08f72bca60f92d38319) |

### F5: Clef-compatible checkpoints

| Repository | Declared license | Checked revision |
|---|---|---|
| [ApolloRaines/Clef-Flash-Concise-m0.5](https://huggingface.co/ApolloRaines/Clef-Flash-Concise-m0.5) | apache-2.0 | [f056035c9d29](https://huggingface.co/ApolloRaines/Clef-Flash-Concise-m0.5/tree/f056035c9d2993e79995f840689f26c78c658f63) |
| [ApolloRaines/Clef-Flash-ReducedSycophancy-m0.5](https://huggingface.co/ApolloRaines/Clef-Flash-ReducedSycophancy-m0.5) | apache-2.0 | [dbb36481b267](https://huggingface.co/ApolloRaines/Clef-Flash-ReducedSycophancy-m0.5/tree/dbb36481b267ac254d0dd2e785e9cf97d0e194af) |
| [ApolloRaines/Clef-Flash-Truthful-m0.5](https://huggingface.co/ApolloRaines/Clef-Flash-Truthful-m0.5) | apache-2.0 | [7206762b094d](https://huggingface.co/ApolloRaines/Clef-Flash-Truthful-m0.5/tree/7206762b094dda606363a0b44645c6adf996e289) |
| [ArkhAngelLifeJiggy/clef](https://huggingface.co/ArkhAngelLifeJiggy/clef) | apache-2.0 | [8d8bae1b8ec2](https://huggingface.co/ArkhAngelLifeJiggy/clef/tree/8d8bae1b8ec261fe57ba9980f7416694291a3230) |
| [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) | apache-2.0 | [ed3eed331870](https://huggingface.co/Cloudflare/clef/tree/ed3eed331870db2eff4b0db01237128ede8a00ce) |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | apache-2.0 | [fde727a28700](https://huggingface.co/Cloudflare/clef-flash/tree/fde727a287004204b7518dcc983fe64379776712) |
| [developerjeremylive/clef-etheroi](https://huggingface.co/developerjeremylive/clef-etheroi) | apache-2.0 | [144957ba6851](https://huggingface.co/developerjeremylive/clef-etheroi/tree/144957ba685101e77a12aef926f477555630fae1) |
| [FluidInference/clef-vision-0.8b](https://huggingface.co/FluidInference/clef-vision-0.8b) | apache-2.0 | [740bf3db034d](https://huggingface.co/FluidInference/clef-vision-0.8b/tree/740bf3db034db3cda90fd4abbe4b91b121e129dc) |
| [longlenai-team/clef](https://huggingface.co/longlenai-team/clef) | apache-2.0 | [18343e8956e5](https://huggingface.co/longlenai-team/clef/tree/18343e8956e553eaf0b58dd9aaa7b1c8b1cf93a7) |
| [longlenai-team/clef-flash](https://huggingface.co/longlenai-team/clef-flash) | apache-2.0 | [03cb7e75369b](https://huggingface.co/longlenai-team/clef-flash/tree/03cb7e75369b64e858d95b386977c8a329755459) |
| [mlx-community/Qwen3.5-0.8B-decision](https://huggingface.co/mlx-community/Qwen3.5-0.8B-decision) | apache-2.0 | [0e2f011e1c97](https://huggingface.co/mlx-community/Qwen3.5-0.8B-decision/tree/0e2f011e1c975609cc4039e4c3b013faee2ec8e4) |
| [N8STORM/clef](https://huggingface.co/N8STORM/clef) | apache-2.0 | [592d8f496b38](https://huggingface.co/N8STORM/clef/tree/592d8f496b38a7f5a4288f6ad886355298c86e70) |
| [Nnnnnn737/clef](https://huggingface.co/Nnnnnn737/clef) | apache-2.0 | [2590f9da9457](https://huggingface.co/Nnnnnn737/clef/tree/2590f9da94577de350ded50cfad35fd9209eba10) |
| [sekkit/clef](https://huggingface.co/sekkit/clef) | apache-2.0 | [ec40125d5e08](https://huggingface.co/sekkit/clef/tree/ec40125d5e08b5506807f34baf027278b60d05c2) |
| [spmurrayzzz/clefsift](https://huggingface.co/spmurrayzzz/clefsift) | apache-2.0 | [179f4553fcb9](https://huggingface.co/spmurrayzzz/clefsift/tree/179f4553fcb905d16d0b2662885d5395d0a05aa7) |
| [suryatmodulus/clef](https://huggingface.co/suryatmodulus/clef) | apache-2.0 | [c23de02a8be0](https://huggingface.co/suryatmodulus/clef/tree/c23de02a8be04850b37bb3347d21852be0c83bb6) |
| [suryatmodulus/clef-flash](https://huggingface.co/suryatmodulus/clef-flash) | apache-2.0 | [4a0c98206305](https://huggingface.co/suryatmodulus/clef-flash/tree/4a0c98206305255040bb6832e9867683ad606dc3) |
| [vkshdev/clef-flash-9B](https://huggingface.co/vkshdev/clef-flash-9B) | apache-2.0 | [c4aaebdc7300](https://huggingface.co/vkshdev/clef-flash-9B/tree/c4aaebdc7300307bd8240dd7704a69adc7d0355d) |

## Important exclusions

- `jaredpalmer/kev-0.5b` uses Qwen2.5; `kev-0.6b` and `kev-8b` use Qwen3. The native Kev loader requires Qwen3.5.
- `jaredpalmer/kev-27b` is a full-weight release without the required standard LoRA adapter layout. Huncho does not support this Kev full-weight checkpoint contract.
- `bespokelabs/Bespoke-Nimble-9B-v3` lacks the required `serving_config.json` and bundled tokenizer. The current raw Hub loader cannot assemble it directly.
- MLX, Core ML, WebGPU/kevala, ONNX and GGUF conversions are not interchangeable with the native checkpoint layouts. Existing ONNX exports need the exact Huncho graph/prompt/head contract and a package manifest. The optional llama.cpp adapter supports specific Qwen3.5 F2/F3 package contracts and uniform F32/F16/Q8_0/Q4_0 artifacts; it does not make arbitrary Hub GGUFs, Q4_K_M files or Laya GGUFs loadable. See [llama.cpp](llamacpp.md) and [ONNX](backends.md#onnx-feature).

“Excluded” below means the checked release does not satisfy the current automatic native loader contract. A deliberately assembled package, conversion, model change or new implementation might make some of these usable.

| Repository | Family detected | Reason |
|---|---|---|
| [aac6fef/laya-coreml](https://huggingface.co/aac6fef/laya-coreml) | F1 | F1 auto-loader requires root model.safetensors |
| [aac6fef/laya-mlx](https://huggingface.co/aac6fef/laya-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [aac6fef/laya-multilingual-coreml](https://huggingface.co/aac6fef/laya-multilingual-coreml) | F1 | F1 auto-loader requires root model.safetensors |
| [aac6fef/laya-multilingual-coreml-ane](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) | F1 | F1 auto-loader requires root model.safetensors |
| [aac6fef/laya-multilingual-coreml-ane-w8](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane-w8) | F1 | F1 auto-loader requires root model.safetensors |
| [aac6fef/laya-multilingual-coreml-snake](https://huggingface.co/aac6fef/laya-multilingual-coreml-snake) | F1 | F1 auto-loader requires root model.safetensors |
| [aac6fef/laya-multilingual-mlx](https://huggingface.co/aac6fef/laya-multilingual-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [aac6fef/laya-typed-decisions-coreml](https://huggingface.co/aac6fef/laya-typed-decisions-coreml) | F1 | F1 auto-loader requires root model.safetensors |
| [aac6fef/laya-typed-decisions-mlx](https://huggingface.co/aac6fef/laya-typed-decisions-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [aaroncool9/kev-0.5b](https://huggingface.co/aaroncool9/kev-0.5b) | F2 | Native Kev loader rejects backbone qwen2 |
| [Aikimi/clef-flash-int8](https://huggingface.co/Aikimi/clef-flash-int8) | F5 | Backbone quantization is not supported by native Clef |
| [Aikimi/clef-nf4](https://huggingface.co/Aikimi/clef-nf4) | F5 | Backbone quantization is not supported by native Clef |
| [alpha-x-ai/clef-flash-NVFP4](https://huggingface.co/alpha-x-ai/clef-flash-NVFP4) | F5 | Backbone quantization is not supported by native Clef |
| [altslate/certo-unified-4b](https://huggingface.co/altslate/certo-unified-4b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [androidli/laya-multilingual-onnx-int4](https://huggingface.co/androidli/laya-multilingual-onnx-int4) | F1 | F1 auto-loader requires root model.safetensors |
| [anish-shrestha/granite-4.0-h-350m-decision](https://huggingface.co/anish-shrestha/granite-4.0-h-350m-decision) | F2 | Native Kev loader rejects backbone granitemoehybrid |
| [arbazsiddiqui/kev-0.6b-browser-use](https://huggingface.co/arbazsiddiqui/kev-0.6b-browser-use) | F2 | Native Kev loader rejects backbone qwen3 |
| [Arcapollo/laya-shell-judge](https://huggingface.co/Arcapollo/laya-shell-judge) | F1 | Missing encoder/config.json or config.json |
| [archevel/laya-web](https://huggingface.co/archevel/laya-web) | F1 | Missing encoder/config.json or config.json |
| [attentionai/laya-extended](https://huggingface.co/attentionai/laya-extended) | F1 | Missing encoder/config.json or config.json |
| [aufklarer/Clef-flash-9B-MLX-4bit](https://huggingface.co/aufklarer/Clef-flash-9B-MLX-4bit) | F5 | Backbone quantization is not supported by native Clef |
| [aungthuhein-dev/yway-system1-xlmr](https://huggingface.co/aungthuhein-dev/yway-system1-xlmr) | F1 | Native F1 synthesis rejects encoder xlm-roberta |
| [avartha/kev-0.5b-merged](https://huggingface.co/avartha/kev-0.5b-merged) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [avartha/kev-0.6b-merged](https://huggingface.co/avartha/kev-0.6b-merged) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [avartha/kev-0.8b-merged](https://huggingface.co/avartha/kev-0.8b-merged) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [avartha/kev-4b-merged](https://huggingface.co/avartha/kev-4b-merged) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [avartha/kev-8b-merged](https://huggingface.co/avartha/kev-8b-merged) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [avartha/kev-9b-merged](https://huggingface.co/avartha/kev-9b-merged) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [bespokelabs/Bespoke-Nimble-9B-v3](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3) | F3 | Missing required Nimble files: serving_config.json, tokenizer.json |
| [birgermoell/sven-0.6b](https://huggingface.co/birgermoell/sven-0.6b) | F2 | Native Kev loader rejects backbone qwen3 |
| [BrainboxAI/nitzotz](https://huggingface.co/BrainboxAI/nitzotz) | F1 | Native F1 synthesis rejects encoder roberta |
| [brishen/laya-burn-q8](https://huggingface.co/brishen/laya-burn-q8) | F1 | F1 auto-loader requires root model.safetensors |
| [bykof/peekaboolean-450m](https://huggingface.co/bykof/peekaboolean-450m) | F2 | Missing standard Kev adapter weights or tokenizer |
| [chaoliangUNSW/MacJev-322M-4K-Laya-GGUF](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya-GGUF) | F1 | Missing encoder/config.json or config.json |
| [charioteer/laya-mobile](https://huggingface.co/charioteer/laya-mobile) | F1 | Missing encoder/config.json or config.json |
| [Code4me2/clef-flash-NVFP4](https://huggingface.co/Code4me2/clef-flash-NVFP4) | F5 | Backbone quantization is not supported by native Clef |
| [cxj05h/laya-ledger-mlx](https://huggingface.co/cxj05h/laya-ledger-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [Dunde/laya-tf-judge](https://huggingface.co/Dunde/laya-tf-judge) | F1 | Missing encoder/config.json or config.json |
| [elbruno/laya-onnx](https://huggingface.co/elbruno/laya-onnx) | F1 | Native F1 synthesis rejects encoder laya |
| [EliovpAI/clef-MXFP4](https://huggingface.co/EliovpAI/clef-MXFP4) | F5 | Backbone quantization is not supported by native Clef |
| [Emerald7664/laya-multilingual-onnx](https://huggingface.co/Emerald7664/laya-multilingual-onnx) | F1 | Missing encoder/config.json or config.json |
| [enzinol/Fred](https://huggingface.co/enzinol/Fred) | F2 | Native Kev loader rejects backbone lfm2 |
| [fancyboi999/ockev](https://huggingface.co/fancyboi999/ockev) | F2 | Native Kev loader rejects backbone qwen2 |
| [fancyboi999/ockev-3b](https://huggingface.co/fancyboi999/ockev-3b) | F2 | Native Kev loader rejects backbone qwen2 |
| [FluidInference/laya-english-coreml](https://huggingface.co/FluidInference/laya-english-coreml) | F1 | F1 auto-loader requires root model.safetensors |
| [Foreverstorm/radiochat-laya-multilingual-onnx](https://huggingface.co/Foreverstorm/radiochat-laya-multilingual-onnx) | F1 | Missing encoder/config.json or config.json |
| [fr0stbit3/laya-gguf](https://huggingface.co/fr0stbit3/laya-gguf) | F1 | Missing encoder/config.json or config.json |
| [fr0stbit3/laya-multilingual-gguf](https://huggingface.co/fr0stbit3/laya-multilingual-gguf) | F1 | Missing encoder/config.json or config.json |
| [fr0stbit3/laya-typed-decisions-gguf](https://huggingface.co/fr0stbit3/laya-typed-decisions-gguf) | F1 | Missing encoder/config.json or config.json |
| [franckverrot/lev-350m](https://huggingface.co/franckverrot/lev-350m) | F2 | Native Kev loader rejects backbone lfm2 |
| [fukayatti0/jev-japanese-judgment](https://huggingface.co/fukayatti0/jev-japanese-judgment) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [fukayatti0/jev-japanese-judgment-v2](https://huggingface.co/fukayatti0/jev-japanese-judgment-v2) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [gabrielfior/decider-q2b-1.1b](https://huggingface.co/gabrielfior/decider-q2b-1.1b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [goktugoguz/laya-multilingual-arc-mlx](https://huggingface.co/goktugoguz/laya-multilingual-arc-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [goktugoguz/laya-multilingual-stellar-mlx](https://huggingface.co/goktugoguz/laya-multilingual-stellar-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [guoan1/jev-qwen35-08b](https://huggingface.co/guoan1/jev-qwen35-08b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [HanWangTJU/haidass1.5-143m-jev](https://huggingface.co/HanWangTJU/haidass1.5-143m-jev) | F2 | Native Kev loader rejects backbone qwen3 |
| [harshpreet931/cut-laya-onnx](https://huggingface.co/harshpreet931/cut-laya-onnx) | F1 | Missing encoder/config.json or config.json |
| [helmo/DecidaBERT-large](https://huggingface.co/helmo/DecidaBERT-large) | F1 | Safetensors missing native trained Laya head tensors |
| [hiteshluke/arbiter-4b](https://huggingface.co/hiteshluke/arbiter-4b) | F2 | Kev auto-detection requires FEATURE_EXTRACTION |
| [iFurySt/openjev](https://huggingface.co/iFurySt/openjev) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [inferenceprince/laya-onnx](https://huggingface.co/inferenceprince/laya-onnx) | F1 | Native F1 synthesis rejects encoder laya |
| [inferenceprince/laya-onnx-int8](https://huggingface.co/inferenceprince/laya-onnx-int8) | F1 | Native F1 synthesis rejects encoder laya |
| [itsdevice/lfm-2.5-350m-decs1](https://huggingface.co/itsdevice/lfm-2.5-350m-decs1) | F5 | Unsupported Clef backbone lfm2 |
| [JacobTDang/secondhand-laya](https://huggingface.co/JacobTDang/secondhand-laya) | F1 | Missing encoder/config.json or config.json |
| [Jakevin/clef-flash-mixed-GGUF](https://huggingface.co/Jakevin/clef-flash-mixed-GGUF) | F5 | No root backbone config |
| [Jakevin/clef-flash-mixed-vision-GGUF](https://huggingface.co/Jakevin/clef-flash-mixed-vision-GGUF) | F5 | No root backbone config |
| [Jakevin/clef-flash-ternary-mlx](https://huggingface.co/Jakevin/clef-flash-ternary-mlx) | F5 | Nonstandard packed/quantized backbone tensors: BF16,I32,U32,U8 |
| [Jakevin/clef-flash-ternary-vision-mlx](https://huggingface.co/Jakevin/clef-flash-ternary-vision-mlx) | F5 | Nonstandard packed/quantized backbone tensors: BF16,I32,U32,U8 |
| [Jakevin/kev-4b-ternary-mlx](https://huggingface.co/Jakevin/kev-4b-ternary-mlx) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [jaredpalmer/kev-0.5b](https://huggingface.co/jaredpalmer/kev-0.5b) | F2 | Native Kev loader rejects backbone qwen2 |
| [jaredpalmer/kev-0.6b](https://huggingface.co/jaredpalmer/kev-0.6b) | F2 | Native Kev loader rejects backbone qwen3 |
| [jaredpalmer/kev-27b](https://huggingface.co/jaredpalmer/kev-27b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [jaredpalmer/kev-8b](https://huggingface.co/jaredpalmer/kev-8b) | F2 | Native Kev loader rejects backbone qwen3 |
| [jaswanthsanjay88/jev-0.5b](https://huggingface.co/jaswanthsanjay88/jev-0.5b) | F2 | Native Kev loader rejects backbone qwen2 |
| [jaswanthsanjay88/rev-0.5b](https://huggingface.co/jaswanthsanjay88/rev-0.5b) | F2 | Native Kev loader rejects backbone qwen2 |
| [jaswanthsanjay88/rev-4b](https://huggingface.co/jaswanthsanjay88/rev-4b) | F2 | Native Kev loader rejects backbone qwen3 |
| [jbarney/circuit-1.7b](https://huggingface.co/jbarney/circuit-1.7b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [jbarney/circuit-8b](https://huggingface.co/jbarney/circuit-8b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [jbarney/circuit-audio-7b](https://huggingface.co/jbarney/circuit-audio-7b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [jbarney/circuit-vl-4b](https://huggingface.co/jbarney/circuit-vl-4b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [JohnP1/d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) | F2 | Adapter refers to a missing or local base path; cannot resolve it as a Hub model |
| [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) | F2 | Adapter refers to a missing or local base path; cannot resolve it as a Hub model |
| [JohnP1/kev-gemma4-e2b](https://huggingface.co/JohnP1/kev-gemma4-e2b) | F2 | Adapter refers to a missing or local base path; cannot resolve it as a Hub model |
| [jsaurabh/qwen3.5-9b-jev-data-mix-v2](https://huggingface.co/jsaurabh/qwen3.5-9b-jev-data-mix-v2) | F3 | Missing required Nimble files: serving_config.json |
| [Jundot/clef-flash-oQ4e](https://huggingface.co/Jundot/clef-flash-oQ4e) | F5 | Backbone quantization is not supported by native Clef |
| [Jundot/clef-oQ4e](https://huggingface.co/Jundot/clef-oQ4e) | F5 | Backbone quantization is not supported by native Clef |
| [kallebysantos/laya-onnx](https://huggingface.co/kallebysantos/laya-onnx) | F1 | Missing encoder/config.json or config.json |
| [knpatil/laya-browser-agent-base](https://huggingface.co/knpatil/laya-browser-agent-base) | F1 | Missing encoder/config.json or config.json |
| [kongyo/Ketsudan-27B](https://huggingface.co/kongyo/Ketsudan-27B) | F5 | No root backbone config |
| [lafalce/system-one-model](https://huggingface.co/lafalce/system-one-model) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [lewisdog/aoa-course-decision-0.6b](https://huggingface.co/lewisdog/aoa-course-decision-0.6b) | F2 | Native Kev loader rejects backbone qwen3 |
| [Livesport/clef-flash-GGUF](https://huggingface.co/Livesport/clef-flash-GGUF) | F5 | Missing native backbone weights |
| [luispoveda93/minicpm4-pcap-kev-v1](https://huggingface.co/luispoveda93/minicpm4-pcap-kev-v1) | F2 | Native Kev loader rejects backbone None |
| [luispoveda93/sharky-0.5B](https://huggingface.co/luispoveda93/sharky-0.5B) | F2 | Native Kev loader rejects backbone None |
| [marrviin/laya-en-fp16](https://huggingface.co/marrviin/laya-en-fp16) | F1 | Missing encoder/config.json or config.json |
| [meossistant/clef-flash-4bit](https://huggingface.co/meossistant/clef-flash-4bit) | F5 | Backbone quantization is not supported by native Clef |
| [meossistant/clef-flash-8bit](https://huggingface.co/meossistant/clef-flash-8bit) | F5 | Backbone quantization is not supported by native Clef |
| [meossistant/clef-flash-openvino](https://huggingface.co/meossistant/clef-flash-openvino) | F5 | No root backbone config |
| [millat/laya-mlx](https://huggingface.co/millat/laya-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [misukisu/kev-0.5b-multitask](https://huggingface.co/misukisu/kev-0.5b-multitask) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [misukisu/kev-0.5b-multitask-finnish-GGUF](https://huggingface.co/misukisu/kev-0.5b-multitask-finnish-GGUF) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [mizchi/laya-multilingual-onnx](https://huggingface.co/mizchi/laya-multilingual-onnx) | F1 | Missing encoder/config.json or config.json |
| [mlboydaisuke/clef-flash-CoreAI](https://huggingface.co/mlboydaisuke/clef-flash-CoreAI) | F5 | Missing native joint head or tokenizer |
| [mlboydaisuke/Laya-Multilingual-CoreAI](https://huggingface.co/mlboydaisuke/Laya-Multilingual-CoreAI) | F1 | F1 auto-loader requires root model.safetensors |
| [mlx-community/clef-4bit](https://huggingface.co/mlx-community/clef-4bit) | F5 | Backbone quantization is not supported by native Clef |
| [mlx-community/clef-8bit](https://huggingface.co/mlx-community/clef-8bit) | F5 | Backbone quantization is not supported by native Clef |
| [mlx-community/clef-flash-4bit](https://huggingface.co/mlx-community/clef-flash-4bit) | F5 | Backbone quantization is not supported by native Clef |
| [mlx-community/clef-flash-8bit](https://huggingface.co/mlx-community/clef-flash-8bit) | F5 | Backbone quantization is not supported by native Clef |
| [MohamedHamed001/laya-delegation-v3](https://huggingface.co/MohamedHamed001/laya-delegation-v3) | F1 | Missing encoder/config.json or config.json |
| [mudler/Bespoke-Nimble-9B-vllm-cpp](https://huggingface.co/mudler/Bespoke-Nimble-9B-vllm-cpp) | F3 | Missing required Nimble files: adapter_model.safetensors, serving_config.json |
| [mukti-sys/cortex-1-large](https://huggingface.co/mukti-sys/cortex-1-large) | F1 | Native F1 synthesis rejects encoder  |
| [myroslavtryhubets/clef-flash-NVFP4](https://huggingface.co/myroslavtryhubets/clef-flash-NVFP4) | F5 | Backbone quantization is not supported by native Clef |
| [myroslavtryhubets/clef-flash-NVFP4A16](https://huggingface.co/myroslavtryhubets/clef-flash-NVFP4A16) | F5 | Backbone quantization is not supported by native Clef |
| [myroslavtryhubets/clef-NVFP4](https://huggingface.co/myroslavtryhubets/clef-NVFP4) | F5 | Backbone quantization is not supported by native Clef |
| [nace-ai/drex-dlm](https://huggingface.co/nace-ai/drex-dlm) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [nampham1106/laya-flash](https://huggingface.co/nampham1106/laya-flash) | F1 | Native F1 synthesis rejects encoder lfm2 |
| [nativ-community/clef-flash-MLX-8bit](https://huggingface.co/nativ-community/clef-flash-MLX-8bit) | F5 | Backbone quantization is not supported by native Clef |
| [nativ-community/clef-flash-MLX-MXFP4](https://huggingface.co/nativ-community/clef-flash-MLX-MXFP4) | F5 | Backbone quantization is not supported by native Clef |
| [nativ-community/clef-flash-MLX-NVFP4](https://huggingface.co/nativ-community/clef-flash-MLX-NVFP4) | F5 | Backbone quantization is not supported by native Clef |
| [nativ-community/clef-MLX-8bit](https://huggingface.co/nativ-community/clef-MLX-8bit) | F5 | Backbone quantization is not supported by native Clef |
| [nativ-community/clef-MLX-MXFP4](https://huggingface.co/nativ-community/clef-MLX-MXFP4) | F5 | Backbone quantization is not supported by native Clef |
| [nativ-community/clef-MLX-NVFP4](https://huggingface.co/nativ-community/clef-MLX-NVFP4) | F5 | Backbone quantization is not supported by native Clef |
| [nigelleong0703/Taiji-2B](https://huggingface.co/nigelleong0703/Taiji-2B) | F2 | Kev auto-detection requires FEATURE_EXTRACTION |
| [Nipun/pocket-pilot-json-decisions-0.8b](https://huggingface.co/Nipun/pocket-pilot-json-decisions-0.8b) | F5 | No root backbone config |
| [nnlgsakib/laya-vision](https://huggingface.co/nnlgsakib/laya-vision) | F1 | Safetensors missing native trained Laya head tensors |
| [noscienthoon/ouro-2.6b-decision-lora](https://huggingface.co/noscienthoon/ouro-2.6b-decision-lora) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [olaverse/PurpleMIST-Flash-1.0](https://huggingface.co/olaverse/PurpleMIST-Flash-1.0) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [olaverse/PurpleMIST-Mini-1.0](https://huggingface.co/olaverse/PurpleMIST-Mini-1.0) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [p-rk/decision-engine-laya-onnx](https://huggingface.co/p-rk/decision-engine-laya-onnx) | F1 | F1 auto-loader requires root model.safetensors |
| [p-yan/laya-quanto](https://huggingface.co/p-yan/laya-quanto) | F1 | F1 auto-loader requires root model.safetensors |
| [pac-pac-pac/picky-9b-v10-r12](https://huggingface.co/pac-pac-pac/picky-9b-v10-r12) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [paideica/assunta-laya](https://huggingface.co/paideica/assunta-laya) | F1 | Missing encoder/config.json or config.json |
| [persly/persly-jev-9b](https://huggingface.co/persly/persly-jev-9b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [php-ai/zeiger-0.6b](https://huggingface.co/php-ai/zeiger-0.6b) | F1 | Native F1 synthesis rejects encoder qwen3 |
| [piffie/laya-onnx](https://huggingface.co/piffie/laya-onnx) | F1 | Missing encoder/config.json or config.json |
| [pleveneur/laya-multilingual-onnx](https://huggingface.co/pleveneur/laya-multilingual-onnx) | F1 | Missing encoder/config.json or config.json |
| [PSanni/mati-14b](https://huggingface.co/PSanni/mati-14b) | F2 | Native Kev loader rejects backbone qwen3 |
| [pythainlp/laya-multilingual-onnx](https://huggingface.co/pythainlp/laya-multilingual-onnx) | F1 | Missing encoder/config.json or config.json |
| [qnaug/embeddinggemma2-decision](https://huggingface.co/qnaug/embeddinggemma2-decision) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [Ramg77/laya-sentiment-multilingual-coreml](https://huggingface.co/Ramg77/laya-sentiment-multilingual-coreml) | F1 | F1 auto-loader requires root model.safetensors |
| [ramgpt/clef-EXL3](https://huggingface.co/ramgpt/clef-EXL3) | F5 | Backbone quantization is not supported by native Clef |
| [ramgpt/clef-flash-EXL3](https://huggingface.co/ramgpt/clef-flash-EXL3) | F5 | Backbone quantization is not supported by native Clef |
| [richardyoung/Bev-9B-inverted](https://huggingface.co/richardyoung/Bev-9B-inverted) | F3 | Missing required Nimble files: adapter_config.json, adapter_model.safetensors |
| [RoderickQiu/kev-4b-mlx-8bit](https://huggingface.co/RoderickQiu/kev-4b-mlx-8bit) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [RoeiG/laya-hebrew](https://huggingface.co/RoeiG/laya-hebrew) | F1 | Native F1 synthesis rejects encoder neobert |
| [RoeiG/Laya-Hebrew-V2](https://huggingface.co/RoeiG/Laya-Hebrew-V2) | F1 | Native F1 synthesis rejects encoder neobert |
| [rupeshs/laya-ov-int8](https://huggingface.co/rupeshs/laya-ov-int8) | F1 | Missing encoder/config.json or config.json |
| [RYasuda/kev-ja-emotion-kl-0.8b](https://huggingface.co/RYasuda/kev-ja-emotion-kl-0.8b) | F2 | Kev head.pt and adapter config disagree about base or LoRA rank |
| [salgara/laya-triz-typed-decisions](https://huggingface.co/salgara/laya-triz-typed-decisions) | F1 | Missing encoder/config.json or config.json |
| [Sharjeelbaig/laya-multilingual-onnx](https://huggingface.co/Sharjeelbaig/laya-multilingual-onnx) | F1 | Missing encoder/config.json or config.json |
| [Sharjeelbaig/laya-multilingual-onnx-int8](https://huggingface.co/Sharjeelbaig/laya-multilingual-onnx-int8) | F1 | Missing encoder/config.json or config.json |
| [sidman007/jev-vlm-4b](https://huggingface.co/sidman007/jev-vlm-4b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [Sigmo23/clef-flash-8bit](https://huggingface.co/Sigmo23/clef-flash-8bit) | F5 | Backbone quantization is not supported by native Clef |
| [simonlehmann/clef-NVFP4](https://huggingface.co/simonlehmann/clef-NVFP4) | F5 | Backbone quantization is not supported by native Clef |
| [SimpleJev/JevAny-Gemma-4B-LoRA](https://huggingface.co/SimpleJev/JevAny-Gemma-4B-LoRA) | F2 | Native Kev loader rejects backbone gemma4_text |
| [SimpleJev/JevAny-Muse-Glimmer-30B-LoRA](https://huggingface.co/SimpleJev/JevAny-Muse-Glimmer-30B-LoRA) | F2 | Native Kev loader rejects backbone muse_glimmer_text |
| [sjoerdbodbijl/laya-mlx](https://huggingface.co/sjoerdbodbijl/laya-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [smdesai/laya-multilingual-coreai](https://huggingface.co/smdesai/laya-multilingual-coreai) | F1 | F1 auto-loader requires root model.safetensors |
| [smha1012/jev-9b](https://huggingface.co/smha1012/jev-9b) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [Snugasabug/laya-finetuned-rlcd](https://huggingface.co/Snugasabug/laya-finetuned-rlcd) | F1 | Missing encoder/config.json or config.json |
| [soyelmismo/laya-multilingual-onnx](https://huggingface.co/soyelmismo/laya-multilingual-onnx) | F1 | Missing encoder/config.json or config.json |
| [Steven10429/laya-multilingual-webgpu](https://huggingface.co/Steven10429/laya-multilingual-webgpu) | F1 | Missing encoder/config.json or config.json |
| [Steven10429/laya-typed-decisions-webgpu-q4](https://huggingface.co/Steven10429/laya-typed-decisions-webgpu-q4) | F1 | Missing encoder/config.json or config.json |
| [sthanika-ai/Sieve-27B](https://huggingface.co/sthanika-ai/Sieve-27B) | F2 | Kev head.pt and adapter config disagree about base or LoRA rank |
| [sthanika-ai/Sieve-9B](https://huggingface.co/sthanika-ai/Sieve-9B) | F2 | Kev head.pt and adapter config disagree about base or LoRA rank |
| [sthanika-ai/Sieve-9B-Plus](https://huggingface.co/sthanika-ai/Sieve-9B-Plus) | F2 | Kev head.pt and adapter config disagree about base or LoRA rank |
| [Suarify/laya-pdpa-onnx-v2](https://huggingface.co/Suarify/laya-pdpa-onnx-v2) | F1 | Missing encoder/config.json or config.json |
| [Suarify/laya-pdpa-onnx-v2-32k](https://huggingface.co/Suarify/laya-pdpa-onnx-v2-32k) | F1 | Missing encoder/config.json or config.json |
| [Suarify/laya-pdpa-onnx-v3-32k](https://huggingface.co/Suarify/laya-pdpa-onnx-v3-32k) | F1 | Missing encoder/config.json or config.json |
| [Suarify/laya-pdpa-onnx-v4-32k](https://huggingface.co/Suarify/laya-pdpa-onnx-v4-32k) | F1 | Missing encoder/config.json or config.json |
| [sungwon1110/gemma4-e2b-gev](https://huggingface.co/sungwon1110/gemma4-e2b-gev) | F2 | Adapter refers to a missing or local base path; cannot resolve it as a Hub model |
| [SVPG-Labs/kev-27b-NVFP4](https://huggingface.co/SVPG-Labs/kev-27b-NVFP4) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [TanmaySah/InvariantOne-v1](https://huggingface.co/TanmaySah/InvariantOne-v1) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [terminusresearch/cloudflare-clef-int8-convrot](https://huggingface.co/terminusresearch/cloudflare-clef-int8-convrot) | F5 | Missing native backbone weights |
| [Terom/kev-0.5b](https://huggingface.co/Terom/kev-0.5b) | F2 | Native Kev loader rejects backbone qwen2 |
| [tozp/laya-onnx](https://huggingface.co/tozp/laya-onnx) | F1 | F1 auto-loader requires root model.safetensors |
| [trevest/Clef-LoRA-for-Qwen3.8-27B-GGUF](https://huggingface.co/trevest/Clef-LoRA-for-Qwen3.8-27B-GGUF) | F5 | No root backbone config |
| [TrevorJS/clef-flash-mlx-4bit](https://huggingface.co/TrevorJS/clef-flash-mlx-4bit) | F5 | Backbone quantization is not supported by native Clef |
| [TrevorJS/clef-flash-mlx-8bit](https://huggingface.co/TrevorJS/clef-flash-mlx-8bit) | F5 | Backbone quantization is not supported by native Clef |
| [uppppiiiii/laya_onnx](https://huggingface.co/uppppiiiii/laya_onnx) | F1 | Missing encoder/config.json or config.json |
| [uzuw/laya-cli-gate](https://huggingface.co/uzuw/laya-cli-gate) | F1 | Missing native tokenizer |
| [Vaibhavhome30/laya-mlx](https://huggingface.co/Vaibhavhome30/laya-mlx) | F1 | Safetensors missing native trained Laya head tensors |
| [VictorNanka/nev-0.8b](https://huggingface.co/VictorNanka/nev-0.8b) | F2 | Kev metadata missing positive head_dim/LoRA rank |
| [VishalMysore/laya-moe-web](https://huggingface.co/VishalMysore/laya-moe-web) | F1 | Missing encoder/config.json or config.json |
| [VishalMysore/layaForWeb](https://huggingface.co/VishalMysore/layaForWeb) | F1 | Missing encoder/config.json or config.json |
| [VishalMysore/layaForWebTrained](https://huggingface.co/VishalMysore/layaForWebTrained) | F1 | Missing encoder/config.json or config.json |
| [Vishva007/clef-flash-W4A16-AutoRound](https://huggingface.co/Vishva007/clef-flash-W4A16-AutoRound) | F5 | Backbone quantization is not supported by native Clef |
| [Vishva007/clef-flash-W4A16-AutoRound-GPTQ](https://huggingface.co/Vishva007/clef-flash-W4A16-AutoRound-GPTQ) | F5 | Backbone quantization is not supported by native Clef |
| [Vishva007/clef-W4A16-AutoRound](https://huggingface.co/Vishva007/clef-W4A16-AutoRound) | F5 | Backbone quantization is not supported by native Clef |
| [Vishva007/clef-W4A16-AutoRound-GPTQ](https://huggingface.co/Vishva007/clef-W4A16-AutoRound-GPTQ) | F5 | Backbone quantization is not supported by native Clef |
| [vrajnotviraj/laya-intent-router-150m-onnx](https://huggingface.co/vrajnotviraj/laya-intent-router-150m-onnx) | F1 | Missing encoder/config.json or config.json |
| [Weidows/laya-multilingual-FP8](https://huggingface.co/Weidows/laya-multilingual-FP8) | F1 | F1 auto-loader requires root model.safetensors |
| [Weidows/laya-multilingual-GGUF](https://huggingface.co/Weidows/laya-multilingual-GGUF) | F1 | Missing encoder/config.json or config.json |
| [wexare/laya-onnx](https://huggingface.co/wexare/laya-onnx) | F1 | Missing encoder/config.json or config.json |
| [wxsys/code-oracle-laya-421m](https://huggingface.co/wxsys/code-oracle-laya-421m) | F1 | Missing encoder/config.json or config.json |
| [xmahbub/laya-cyber-v1](https://huggingface.co/xmahbub/laya-cyber-v1) | F1 | F1 auto-loader requires root model.safetensors |
| [yah01/vjev-vision](https://huggingface.co/yah01/vjev-vision) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [yah01/vjev-vision-pilot](https://huggingface.co/yah01/vjev-vision-pilot) | F2 | Kev auto-loader requires LoRA adapter_config.json; full weights/other packaging unsupported |
| [ytbai/MiniCPM5-2B-Jev](https://huggingface.co/ytbai/MiniCPM5-2B-Jev) | F2 | Missing standard Kev adapter weights or tokenizer |
| [yyhlm/rubert-tiny-turbo-typed-decisions-ru](https://huggingface.co/yyhlm/rubert-tiny-turbo-typed-decisions-ru) | F1 | Native F1 synthesis rejects encoder bert |
| [ZhangPY/kev-snake-text-grpo](https://huggingface.co/ZhangPY/kev-snake-text-grpo) | F2 | Kev head.pt and adapter config disagree about base or LoRA rank |
| [ZhangPY/kev-snake-vision-grpo](https://huggingface.co/ZhangPY/kev-snake-vision-grpo) | F2 | Adapter refers to a missing or local base path; cannot resolve it as a Hub model |
| [ZichenAI/layazh-zh](https://huggingface.co/ZichenAI/layazh-zh) | F1 | Native F1 synthesis rejects encoder bert |
| [ZLHAOOO/laya-mlx-zh](https://huggingface.co/ZLHAOOO/laya-mlx-zh) | F1 | Safetensors missing native trained Laya head tensors |

## Unverified releases

| Repository | Reason |
|---|---|
| [IAMCB/laya-voice-router](https://huggingface.co/IAMCB/laya-voice-router) | HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/IAMCB/laya-voice-router/resolve/30b1cd74d1062e324df4919d2397191bc331f71d/encoder/config.json |
| [jlancaster/clef-27b-nf4](https://huggingface.co/jlancaster/clef-27b-nf4) | HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/jlancaster/clef-27b-nf4/resolve/ea93b6e346af80ce0bab76283bcb421b387d2a48/config.json |
| [ProKope-AI/ProKope-421M](https://huggingface.co/ProKope-AI/ProKope-421M) | HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/ProKope-AI/ProKope-421M/resolve/aa59adac0fac53d740d641f7902ac4ad017284dd/encoder/config.json |
| [ramu9703/radkev-27b](https://huggingface.co/ramu9703/radkev-27b) | Release files require authentication; adapter metadata could not be read |
| [ramu9703/radkev-27b-v2](https://huggingface.co/ramu9703/radkev-27b-v2) | Release files require authentication; adapter metadata could not be read |
| [ramu9703/radkev-9b](https://huggingface.co/ramu9703/radkev-9b) | Release files require authentication; adapter metadata could not be read |
| [Rckflr/kev-gemma-270m-experimental](https://huggingface.co/Rckflr/kev-gemma-270m-experimental) | HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/google/gemma-3-270m/resolve/main/config.json |
| [Rckflr/kev-gemma-270m-rule-classifier-es-v8-experimental](https://huggingface.co/Rckflr/kev-gemma-270m-rule-classifier-es-v8-experimental) | HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/google/gemma-3-270m/resolve/main/config.json |
| [wipen/Duan](https://huggingface.co/wipen/Duan) | HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/wipen/Duan/resolve/50e0615b576e4829cbad5c193d42edd0cadf2784/encoder/config.json |

## Search coverage

Repository listings were collected from the Hugging Face public models API, following pagination for the family-name searches and large tag lists. Discovery included authors `jaredpalmer`, `convaiinnovations`, `Cloudflare`; searches `laya`, `kev`, `clef`, `Nimble`, `Bespoke-Nimble`, `huncho`; and tags `decision-model`, `system-one`, `systemone`, `jev`, `calibrated-decisions`, `system-1`, `system1`, `decision-models`. Matching releases were pinned to their API snapshot commit before inspection.

This is a broad reproducible scan of public discoverable releases, not a claim that every repository on Hugging Face was inspected. Private releases, access-controlled files, untagged models with unrelated names, nonstandard subfolder packages and future updates may be absent. The other 8,036 discovered records did not expose one of the recognized decision-package marker files at the root and were not validated as decision checkpoints.

A machine-readable inventory with all 597 outcomes, full revision pins and compatibility summaries is in [huggingface-models.json](huggingface-models.json). Raw metadata, configs, tensor headers and the scan/refinement scripts remain under `/tmp/huncho-hf-*` and `/tmp/huncho_hf_*.py` for this session.
