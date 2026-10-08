import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

repo = Path('/home/grafuls/Sources/huncho')
fixture = repo / 'crates/huncho-backend/tests/fixtures/tiny_kev'
binary = repo / 'target/release/huncho'
root = Path(tempfile.mkdtemp(prefix='huncho-prompt-cache-pilot-20261007-'))
package = root / 'package'
package.mkdir()
for filename in ['config.json', 'model.safetensors', 'adapter_config.json', 'adapter_model.safetensors', 'head.pt', 'tokenizer.json']:
    shutil.copyfile(fixture / filename, package / filename)
manifest = json.loads((repo / 'examples/mock-model/huncho-model.json').read_text())
manifest.update(name='tiny-kev', family='F2')
manifest['backbone'].update(source={'kind':'hf', 'repo':'fixture/qwen3.5', 'revision':'1111111111111111111111111111111111111111'},
    artifacts={'candle':[{'path':'adapter_model.safetensors','dtype':'fp32'}]}, hidden_size=16, max_context=512, tokenizer='tokenizer.json')
manifest['head'].update(kind='pointer', weights='head.pt', width=4)
manifest['prompt_contract'].update(template='kev-v1', state_budget=512, head_budget=512, max_len=512, head_max_len=192)
manifest['calibration']={'default':{'temperature':2.40605,'confidence':'peak','status':'fit'}, 'entries':{}}
manifest.pop('reference', None)
(package / 'huncho-model.json').write_text(json.dumps(manifest, indent=2) + '\n')
upstream = json.loads((fixture / 'golden.json').read_text())
cases = []
for index, case in enumerate(upstream['cases']):
    request = case['request']
    expected = {}
    for (qid, question), row in zip(request['questions'].items(), case['rows'], strict=True):
        if question['type'] == 'choice': labels = list(question['criteria'])
        elif question['type'] == 'score': labels = [str(i) for i in range(len(question['criteria']))]
        else: labels = ['no', 'yes']
        expected[qid] = dict(zip(labels, row['probabilities'], strict=True))
    cases.append({'id':str(index),'request':request,'expected':expected})
golden = root / 'golden.json'
golden.write_text(json.dumps({'schema_version':'1.0','family':'F2','cases':cases}, indent=2) + '\n')
env = os.environ.copy()
env.update(HUNCHO_DEVICE='cpu', HUNCHO_PROJECTION_CHUNK_ROWS='0', HUNCHO_ATTENTION_FP32='false',
    HUNCHO_TOKEN_CACHE_BYTES='0', RAYON_NUM_THREADS='4', CANDLE_NUM_THREADS='4')
for variable in ['HUNCHO_BACKEND','HUNCHO_DTYPE','HUNCHO_RESULT_CACHE_BYTES']:
    env.pop(variable, None)
records = []
for cache_bytes in [0, 1048576]:
    env['HUNCHO_PROMPT_CACHE_BYTES'] = str(cache_bytes)
    with (root / f'conform-{cache_bytes}.json').open('w') as output, (root / f'conform-{cache_bytes}.log').open('w') as log:
        subprocess.run([str(binary), 'conform', '--model', str(package), '--dtype', 'fp32', '--golden', str(golden), '--json'], env=env, stdout=output, stderr=log, check=True)
    gate = json.loads((root / f'conform-{cache_bytes}.json').read_text())
    assert gate['passed'] and gate['work']['prompt_cache_hits'] == 0
    assert gate['device'] == 'CPU' and gate['dtype'] == 'fp32'
# Alternate cache order in each repetition to reduce a fixed run-order bias.
for repetition in range(3):
    for repeat in [False, True]:
        for cache_bytes in ([0, 1048576] if repetition % 2 == 0 else [1048576, 0]):
            env['HUNCHO_PROMPT_CACHE_BYTES'] = str(cache_bytes)
            stem = f'bench-{repetition}-{int(repeat)}-{cache_bytes}'
            command = [str(binary),'bench','--model',str(package),'--dtype','fp32','--questions','20','--workload','mixed','--iterations','20','--json']
            if repeat: command.append('--repeat-inputs')
            start = time.monotonic()
            with (root / f'{stem}.json').open('w') as output, (root / f'{stem}.log').open('w') as log:
                subprocess.run(command, env=env, stdout=output, stderr=log, check=True)
            report = json.loads((root / f'{stem}.json').read_text())
            assert report['work']['forward_calls'] == 400 and report['work']['result_cache_hits'] == 0
            assert report['work']['prompt_cache_hits'] == (400 if repeat and cache_bytes else 0)
            assert report['device'] == 'CPU' and report['dtype'] == 'fp32'
            records.append({'repetition':repetition,'prompt_cache_bytes':cache_bytes,'repeat_inputs':repeat,'load_inclusive_elapsed_seconds':time.monotonic()-start,'report':report})
            print(json.dumps({k:v for k,v in records[-1].items() if k != 'report'} | {'mean_ms':report['mean_ms']}), flush=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
source_paths = subprocess.check_output(['rg','--files','crates','vendor','examples'], cwd=repo, text=True).splitlines() + ['Cargo.toml','Cargo.lock']
source_digest = hashlib.sha256()
for relative in sorted(source_paths):
    path = repo / relative
    source_digest.update(relative.encode() + b'\0' + path.read_bytes() + b'\0')
summary = {'schema_version':'1.0','fixture_pilot':True,'binary_sha256':sha(binary),'script_sha256':sha(Path(__file__)),
    'source_tree_digest_sha256':source_digest.hexdigest(), 'source_tree_digest_format':'sorted rg --files crates vendor examples plus Cargo.toml/Cargo.lock; relative UTF-8 path NUL file bytes NUL',
    'fixture_file_sha256':{p.name:sha(p) for p in sorted(fixture.iterdir()) if p.is_file()},
    'generated_manifest':manifest,'converted_golden_sha256':sha(golden),'cpu':next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
    'kernel':platform.release(),'build_profile':'release','rayon_num_threads':4,'candle_num_threads':4,
    'temperature':2.40605,'results':records,
    'limits':['tiny pinned two-layer Kev fixture, not released Kev-4B or T4 performance',
      '20 warm closed-loop timed requests per run, one client, 20 mixed questions; loading excluded from mean_ms',
      'three repetitions with alternating cache order; no CPU affinity/frequency isolation; not a production throughput/p99 estimate',
      'conformance vectors converted from unchanged upstream fixture probabilities, no observed outcomes/statistical calibration claim',
      'prepared-prompt hits retain every model forward; result, token-encoding, prefix and batch caches disabled']}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(root / 'golden.json').unlink()
print(root, flush=True)
