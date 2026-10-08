// Assemble a new self-contained CPU browser bundle; never mark it qualified.
import { createHash } from 'node:crypto';
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const inputs = {};
for (let i = 0; i < args.length; i += 2) {
  if (!['--manifest', '--tokenizer', '--model', '--golden', '--out'].includes(args[i]) ||
      !args[i + 1] || inputs[args[i]]) throw new Error('expected unique --manifest --tokenizer --model --golden --out paths');
  inputs[args[i]] = resolve(args[i + 1]);
}
if (Object.keys(inputs).length !== 5) throw new Error('all five paths are required');
const files = {
  'huncho-model.json': inputs['--manifest'], 'tokenizer.json': inputs['--tokenizer'],
  'model.onnx': inputs['--model'], 'golden.json': inputs['--golden'],
  'huncho_browser_core.js': resolve(root, 'pkg/huncho_browser_core.js'),
  'huncho_browser_core_bg.wasm': resolve(root, 'pkg/huncho_browser_core_bg.wasm'),
  'ort.wasm.min.mjs': resolve(root, 'node_modules/onnxruntime-web/dist/ort.wasm.min.mjs'),
  'ort-wasm-simd-threaded.mjs': resolve(root, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs'),
  'ort-wasm-simd-threaded.wasm': resolve(root, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm'),
  'index.mjs': resolve(root, 'src/index.mjs'),
};
const budgets = { 'huncho-model.json': 1, 'tokenizer.json': 16, 'model.onnx': 64,
  'golden.json': 32, 'huncho_browser_core_bg.wasm': 32, 'ort-wasm-simd-threaded.wasm': 64 };
const bytes = Object.fromEntries(await Promise.all(Object.entries(files).map(async ([name, path]) => {
  const size = (budgets[name] ?? 2) * 1024 * 1024;
  const info = await stat(path);
  if (!info.isFile() || info.size > size) throw new Error(`${name} exceeds its browser asset budget`);
  const data = await readFile(path);
  if (data.length > size) throw new Error(`${name} grew beyond its browser asset budget`);
  return [name, data];
})));
const manifest = JSON.parse(bytes['huncho-model.json']);
if (manifest.family !== 'F1' || manifest.head?.kind !== 'option-marker' || manifest.head.width !== 1 ||
    !manifest.calibration?.entries?.['onnx:fp32'] || manifest.calibration.entries['onnx:fp32'].status === 'pending') {
  throw new Error('browser bundle requires an explicit fitted F1 onnx:fp32 scalar head package');
}
const spec = name => ({ url: `./${name}`, sha256: createHash('sha256').update(bytes[name]).digest('hex') });
const config = { provider: 'wasm', head: 'graph-integrated-f1-v1', qualified: false,
  manifest: spec('huncho-model.json'), tokenizer: spec('tokenizer.json'), model: spec('model.onnx'), golden: spec('golden.json'),
  core: { module: spec('huncho_browser_core.js'), wasm: spec('huncho_browser_core_bg.wasm') },
  runtime: { module: spec('ort.wasm.min.mjs'), loader: spec('ort-wasm-simd-threaded.mjs'), wasm: spec('ort-wasm-simd-threaded.wasm') },
  sdk: spec('index.mjs'), note: 'No serving acceptance is inherited. loadPackage performs fresh complete labeled CPU WASM conformance.' };
const descriptor = Buffer.from(JSON.stringify(config, null, 2) + '\n');
const output = inputs['--out'];
await mkdir(output);
try {
  for (const [name, data] of Object.entries(bytes)) await writeFile(resolve(output, name), data, { flag: 'wx' });
  await writeFile(resolve(output, 'config.json'), descriptor, { flag: 'wx' });
} catch (error) {
  await rm(output, { recursive: true, force: true });
  throw error;
}
process.stdout.write(JSON.stringify({ path: output, qualified: false,
  config_sha256: createHash('sha256').update(descriptor).digest('hex') }) + '\n');
