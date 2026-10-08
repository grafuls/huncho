import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

root = Path('/var/tmp/huncho-optimization-20261007')
binary = root / 'huncho-stage2'
package = Path('/root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52')
golden = root / 'reference-cuda-fp16/golden.json'
env = os.environ.copy()
env['HUNCHO_DEVICE'] = 'cuda'
env['RAYON_NUM_THREADS'] = '16'

def get(url):
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()

def metrics(base):
    wanted = {'huncho_tokens_prefilled', 'huncho_requests_coalesced', 'huncho_result_cache_hits'}
    result = {}
    for line in get(base + '/metrics').decode().splitlines():
        if line and not line.startswith('#'):
            parts = line.split()
            if parts[0] in wanted:
                result[parts[0]] = int(float(parts[1]))
    return result

def post(base, data, barrier=None):
    if barrier:
        barrier.wait(timeout=10)
    start = time.monotonic()
    request = urllib.request.Request(base + '/v1/systemone', data=data,
        headers={'Content-Type': 'application/json', 'X-Huncho-Extensions': '1'})
    with urllib.request.urlopen(request, timeout=180) as response:
        body = response.read()
    return (time.monotonic() - start) * 1000, body

def run_mode(name, budget, requests):
    base = 'http://127.0.0.1:18419'
    log = open(root / f'stage2-http-{name}.log', 'w')
    process = subprocess.Popen([str(binary), 'serve', '--model', str(package), '--dtype', 'fp16',
        '--bind', '127.0.0.1:18419', '--coalesce-bytes', str(budget),
        '--qualification-golden', f'kev-4b={golden}'], env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'{name} server exited: {process.returncode}')
            try:
                json.loads(get(base + '/health'))
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('server startup timed out')
                time.sleep(.2)
        post(base, requests[0])
        before = metrics(base)
        bursts = []
        bodies = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as workers:
            for data in requests:
                barrier = threading.Barrier(8)
                start = time.monotonic()
                futures = [workers.submit(post, base, data, barrier) for _ in range(8)]
                replies = [future.result() for future in futures]
                burst_ms = (time.monotonic() - start) * 1000
                assert len({body for _, body in replies}) == 1, 'same input must retain exact HTTP body'
                response = json.loads(replies[0][1])
                assert response['usage']['output_tokens'] == 0
                bodies.append(replies[0][1])
                bursts.append({'wall_ms': burst_ms, 'request_mean_ms': statistics.mean(ms for ms, _ in replies),
                    'request_latencies_ms': [ms for ms, _ in replies],
                    'logical_tokens_per_request': response['usage']['input_tokens'],
                    'body_sha256': hashlib.sha256(replies[0][1]).hexdigest(),
                    'input_sha256': hashlib.sha256(data).hexdigest()})
        after = metrics(base)
        return {'coalesce_bytes': budget, 'concurrency': 8, 'bursts': bursts,
            'work': {key: after[key] - before[key] for key in before},
            'mean_burst_wall_ms': statistics.mean(burst['wall_ms'] for burst in bursts)}, bodies
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()

def main():
    if (root / 'stage2-build.exit').read_text().strip() != '0':
        raise RuntimeError('stage2 build failed')
    with open(root / 'stage2-cuda-independent.json', 'w') as output, open(root / 'stage2-cuda-independent.log', 'w') as log:
        subprocess.run([str(binary), 'conform', '--model', str(package), '--dtype', 'fp16', '--golden', str(golden), '--json'],
            env=env, stdout=output, stderr=log, check=True)
    suite = json.loads(golden.read_text())
    requests = []
    for index in range(3):
        request = json.loads(json.dumps(suite['cases'][0]['request']))
        state = request['state']
        request['state'] = f'{state}\nHTTP burst {index}' if isinstance(state, str) else {'original': state, 'http_burst': index}
        requests.append(json.dumps(request, separators=(',', ':')).encode())
    baseline, old = run_mode('independent', 0, requests)
    coalesced, new = run_mode('coalesced', 1048576, requests)
    assert old == new, 'coalescing must retain bit-identical independent HTTP bodies'
    logical_tokens = sum(burst['logical_tokens_per_request'] for burst in baseline['bursts'])
    assert baseline['work']['huncho_tokens_prefilled'] == 8 * logical_tokens
    assert coalesced['work']['huncho_tokens_prefilled'] == logical_tokens
    assert coalesced['work']['huncho_requests_coalesced'] == 21
    assert coalesced['work']['huncho_result_cache_hits'] == 0
    report = {'schema_version': '1.0', 'model': 'kev-4b', 'dtype': 'fp16', 'device': 'CUDA Tesla T4',
        'temperature': 2.40605, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'source_snapshot_sha256': hashlib.sha256((root / 'huncho-stage2-20261007.tar.gz').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'kernel': os.uname().release, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).decode().strip(),
        'baseline': baseline, 'coalesced': coalesced, 'exact_body_agreement': True,
        'limits': ['three loopback eight-caller bursts after warmup; not open-loop saturation or production p99',
            'same-input duplicate workload; no completed-response retention, prefix reuse or tensor batching',
            'CUDA FP16 independent startup qualification uses unchanged goldens and corrected tie order']}
    (root / 'stage2-http-coalescing.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'independent_mean_burst_ms': baseline['mean_burst_wall_ms'],
        'coalesced_mean_burst_ms': coalesced['mean_burst_wall_ms'], 'exact_body_agreement': True}), flush=True)

try:
    main()
except BaseException:
    (root / 'stage2-qualification.exit').write_text('1\n')
    raise
else:
    (root / 'stage2-qualification.exit').write_text('0\n')
