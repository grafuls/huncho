#!/usr/bin/env python3
import hashlib, json, os, subprocess, time
from pathlib import Path
root = Path('/var/tmp/huncho-cpu-v3-20261008/q8-pilot')
root.mkdir(exist_ok=False)
source = Path('/var/tmp/huncho-optimization-20261007/labeled-data/fit.jsonl')
records = source.read_text().splitlines()[:8]
(root / 'fitting-inputs.jsonl').write_text('\n'.join(records)+'\n')
package = Path('/var/tmp/huncho-quant-cpu-20261008/q8_0-fp32')
binaries = {'portable':Path('/var/tmp/huncho-quant-cpu-20261008/source/target/release/huncho'), 'cpu_v3':Path('/var/tmp/huncho-cpu-v3-20261008/source/target/release/huncho')}
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''): digest.update(part)
    return digest.hexdigest()
env=os.environ.copy()
for key in ['CUDA_ROOT','CUDA_COMPUTE_CAP','LIBRARY_PATH','LD_LIBRARY_PATH','RUSTFLAGS','ORT_LIB_LOCATION','ORT_PREFER_DYNAMIC_LINK','HUNCHO_PROJECTION_CHUNK_ROWS','HUNCHO_ATTENTION_FP32','HUNCHO_CPU_CAUSAL_CONV','HUNCHO_PREFILL_CHUNK_TOKENS','HUNCHO_PROMPT_CACHE_BYTES','HUNCHO_TOKEN_CACHE_BYTES']: env.pop(key,None)
env.update(HUNCHO_DEVICE='cpu',HUNCHO_CPU_DELTA_RULE='true',RAYON_NUM_THREADS='16',CANDLE_NUM_THREADS='16')
audit={'qualified':False, 'samples':[], 'binary_sha256':{key:sha(path) for key,path in binaries.items()}, 'fitting_source_sha256':sha(source), 'pilot_inputs_sha256':sha(root/'fitting-inputs.jsonl'), 'manifest_sha256':sha(package/'huncho-model.json'), 'cpu_affinity':sorted(os.sched_getaffinity(0)), 'limits':['Eight fitting inputs only, no held-out labels or temperature refit.', 'Alternating end-to-end collection includes hashing, load and audit, not steady-state inference.', 'Other qualification/fitting jobs contend for the same CPUs and bandwidth; no isolated speed claim.', 'Same durable Q8 artifact; CPU build instructions change packed dot-product reductions.', 'Both packages and goldens stay unchanged.'], 'logits':{}}
for index, mode in enumerate(['cpu_v3','portable','portable','cpu_v3']):
    output=root/f'{index}-{mode}'
    start=time.monotonic()
    with (root/f'{index}-{mode}.json').open('w') as out, (root/f'{index}-{mode}.log').open('w') as err:
        result=subprocess.run([str(binaries[mode]),'capture-logits','--model',str(package),'--dtype','q8_0-fp32','--data',str(root/'fitting-inputs.jsonl'),'--output',str(output)],env=env,stdout=out,stderr=err)
    sample={'mode':mode,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start}
    if result.returncode==0:
        identity=json.loads((output/'identity.json').read_text())
        sample.update(questions=identity['questions'],work=identity['work'],identity_sha256=sha(output/'identity.json'))
        audit['logits'][str(index)]=json.loads((output/'fit.json').read_text())['rows']
    audit['samples'].append(sample)
    print(json.dumps(sample),flush=True)
    (root/'summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    if result.returncode!=0: raise SystemExit(result.returncode)
if sha(package/'huncho-model.json') != audit['manifest_sha256'] or any(sha(path)!=audit['binary_sha256'][key] for key,path in binaries.items()): raise RuntimeError('inputs changed')
