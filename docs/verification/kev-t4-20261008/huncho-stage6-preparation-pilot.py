"""Pinned Kev HTTP preparation pilot; no artifacts/temperatures are modified."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

root = Path('/var/tmp/huncho-optimization-20261007')
out = root / 'stage6-preparation-pilot.json'
assert not out.exists(), 'preserve previous evidence'
binary = root / 'huncho-stage6'
package = Path('/root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52')
golden = root / 'reference-cuda-fp16/golden.json'
manifest = json.loads((package / 'huncho-model.json').read_text())
suite = json.loads(golden.read_text())
base = next(c['request'] for c in suite['cases'] if c['id'] == 'structured')
requests = []
for index in range(8):
    request = copy.deepcopy(base)
    request['model'] = manifest['name']
    request['state'] = {'source': request['state'], 'pilot_request': index}
    requests.append(json.dumps(request, separators=(',', ':'), ensure_ascii=False).encode())
assert len(set(requests)) == 8
questions = sum(len(json.loads(r)['questions']) for r in requests)
port = 18268
with socket.socket() as probe:
    probe.bind(('127.0.0.1', port))
url = f'http://127.0.0.1:{port}'
env = os.environ.copy()
env.pop('HUNCHO_AUTH_TOKEN', None)
env.update(HUNCHO_DEVICE='cuda', HUNCHO_PROJECTION_CHUNK_ROWS='0', HUNCHO_ATTENTION_FP32='false',
    HUNCHO_RESULT_CACHE_BYTES='0', HUNCHO_PROMPT_CACHE_BYTES='0', HUNCHO_TOKEN_CACHE_BYTES='0',
    HUNCHO_COALESCE_BYTES='0', HUNCHO_PREFIX_CACHE='false', RAYON_NUM_THREADS='8', CANDLE_NUM_THREADS='8')
# Avoid ambient batch settings; this pilot measures the existing independent path.
env.pop('HUNCHO_MAX_BATCH_TOKENS', None)
started = datetime.now(timezone.utc).isoformat()
references = None
results = []

def get(path):
    with urllib.request.urlopen(url + path, timeout=120) as response:
        return response.read()

def post(body):
    req = urllib.request.Request(url + '/v1/systemone', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        assert response.status == 200
        return response.read()

def metrics():
    wanted = {'huncho_tokens_prefilled', 'huncho_questions_prepared', 'huncho_prompt_cache_hits',
        'huncho_result_cache_hits', 'huncho_requests_coalesced', 'huncho_batch_count', 'huncho_fork_count'}
    return {line.split()[0]: float(line.split()[1]) for line in get('/metrics').decode().splitlines()
        if line and line.split()[0] in wanted}

for repetition, slots in enumerate([0, 1, 1, 0]):
    with (root / f'stage6-preparation-http-{repetition}-{slots}.log').open('w') as log:
        command = [str(binary), 'serve', '--model', str(package), '--backend', 'candle', '--dtype', 'fp16',
            '--bind', f'127.0.0.1:{port}', '--extensions', '--max-queued-per-model', '8',
            '--max-prepared-per-model', str(slots), '--qualification-golden', f"{manifest['name']}={golden}"]
        server = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 240
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'server exited with {server.returncode}; see retained log')
                try:
                    if json.loads(get('/health'))['models'] == 1:
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('qualified server did not become ready')
                time.sleep(.2)
            post(requests[0])  # warmup is excluded from all deltas/timings
            before = metrics()
            barrier = threading.Barrier(len(requests))
            def send(index):
                barrier.wait(timeout=10)
                start = time.monotonic()
                response = post(requests[index])
                return index, response, time.monotonic() - start
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=len(requests)) as pool:
                rows = list(pool.map(send, range(len(requests))))
            elapsed = time.monotonic() - start
            after = metrics()
            responses = [r[1] for r in rows]
            if references is None:
                references = responses
            assert responses == references, 'answers/logits/usage changed across preparation modes'
            durations = [r[2] for r in rows]
            work = {key: after[key] - before[key] for key in before}
            assert work['huncho_tokens_prefilled'] == sum(json.loads(r)['usage']['input_tokens'] for r in responses)
            assert work['huncho_questions_prepared'] == (questions if slots else 0)
            assert work['huncho_result_cache_hits'] == work['huncho_prompt_cache_hits'] == work['huncho_requests_coalesced'] == 0
            record = {'repetition': repetition, 'preparation_slots': slots, 'requests': len(requests),
                'questions': questions, 'burst_seconds': elapsed, 'mean_request_seconds': statistics.mean(durations),
                'request_seconds': durations, 'byte_identical_to_disabled': True, 'physical_work': work,
                'response_sha256': [hashlib.sha256(r).hexdigest() for r in responses]}
            results.append(record)
            print(json.dumps(record), flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
report = {'schema_version': '1.0', 'diagnostic_pilot': True, 'started_utc': started,
    'finished_utc': datetime.now(timezone.utc).isoformat(), 'model': manifest['name'], 'device': 'Tesla T4',
    'dtype': 'fp16', 'temperature': 2.40605, 'affinity': sorted(os.sched_getaffinity(0)), 'threads': 8,
    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'source_archive_sha256': hashlib.sha256((root / 'huncho-stage6-20261008.tar.gz').read_bytes()).hexdigest(),
    'golden_sha256': hashlib.sha256(golden.read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'request_sha256': [hashlib.sha256(r).hexdigest() for r in requests], 'results': results,
    'limits': ['Two eight-client bursts per mode, alternating order; not production throughput or p99.',
        'Distinct states with the pinned short structured mixed-question template; no caches/coalescing/batching/prefix.',
        'Unchanged numerical startup gate on every server; held-out labeled preparation gate runs separately.',
        'CPU thread pilot may run concurrently on other NUMA cores; no frequency or system-wide isolation.']}
out.write_text(json.dumps(report, indent=2) + '\n')
