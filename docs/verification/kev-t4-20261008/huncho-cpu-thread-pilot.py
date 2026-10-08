import hashlib
import json
import os
from pathlib import Path
import platform
from datetime import datetime, timezone
import subprocess

root = Path('/var/tmp/huncho-optimization-20261007')
started_utc = datetime.now(timezone.utc).isoformat()
assert not (root / 'stage5-cpu-thread-pilot.json').exists(), 'preserve previous evidence'
binary = root / 'huncho-stage5'
package = Path('/root/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52')
golden = root / 'reference-cpu-fp32/golden.json'
available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
assert available >= 40 * 1024**3, 'CPU qualification requires sufficient free system memory'
allowed = os.sched_getaffinity(0)
node_cpus = set()
for item in Path('/sys/devices/system/node/node1/cpulist').read_text().strip().split(','):
    bounds = item.split('-')
    node_cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
physical = {}
for cpu in sorted(node_cpus & allowed):
    core = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id').read_text().strip()
    physical.setdefault(core, cpu)
assert len(physical) >= 16
affinity = sorted(physical.values())[:16]
os.sched_setaffinity(0, affinity)
env = os.environ.copy()
env.update(HUNCHO_DEVICE='cpu', HUNCHO_PROJECTION_CHUNK_ROWS='0', HUNCHO_ATTENTION_FP32='false')
results = []
for threads in [4, 16]:
    env['RAYON_NUM_THREADS'] = str(threads)
    env['CANDLE_NUM_THREADS'] = str(threads)
    stem = f'stage5-cpu-threads{threads}'
    with open(root / f'{stem}-conform.json', 'w') as output, open(root / f'{stem}-conform.log', 'w') as log:
        process = subprocess.run([str(binary), 'conform', '--model', str(package), '--dtype', 'fp32', '--golden', str(golden), '--json'], env=env, stdout=output, stderr=log)
    gate = json.loads((root / f'{stem}-conform.json').read_text())
    assert gate['device'] == 'CPU' and gate['dtype'] == 'fp32'
    assert (process.returncode == 0) == gate['passed']
    result = {'threads': threads, 'numerical_gate_passed': gate['passed'], 'max_prob_delta': gate['max_prob_delta'], 'argmax_agreement': gate['argmax_agreement'], 'ece_drift': gate['ece']}
    if gate['passed']:
        with open(root / f'{stem}-bench.json', 'w') as output, open(root / f'{stem}-bench.log', 'w') as log:
            subprocess.run([str(binary), 'bench', '--model', str(package), '--dtype', 'fp32', '--questions', '5', '--workload', 'mixed', '--iterations', '5', '--json'], env=env, stdout=output, stderr=log, check=True)
        result['benchmark'] = json.loads((root / f'{stem}-bench.json').read_text())
    results.append(result)
    print(json.dumps(result), flush=True)
report = {'schema_version': '1.0', 'diagnostic_pilot': True, 'model': 'kev-4b', 'dtype': 'fp32', 'device': 'CPU', 'temperature': 2.40605,
    'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'cpu': next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
    'affinity': affinity, 'numa_node': 1, 'kernel': platform.release(), 'results': results,
    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'source_snapshot_sha256': hashlib.sha256((root / 'huncho-stage5-20261007.tar.gz').read_bytes()).hexdigest(),
    'golden_sha256': hashlib.sha256(golden.read_bytes()).hexdigest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'limits': ['five warm distinct mixed-type requests at each thread setting, one client; not production throughput/p99',
        'same 16 physical cores on NUMA node1; no CPU frequency or system-wide isolation; later concurrent jobs can affect timing',
        'unchanged six-case numerical gate, not held-out statistical calibration or universal hardware tuning',
        'RAYON_NUM_THREADS controls Candle GEMM; CANDLE_NUM_THREADS matches it for the barrier pool']}
(root / 'stage5-cpu-thread-pilot.json').write_text(json.dumps(report, indent=2) + '\n')
