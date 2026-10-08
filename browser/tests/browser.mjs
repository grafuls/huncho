// Actual browser WASM execution, with every GPU API disabled and no GPU probe.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { dirname, resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const generated = resolve(root, 'tests/generated');
const readJson = async name => JSON.parse(await readFile(resolve(generated, name), 'utf8'));
const reference = await readJson('reference.json');
const requests = await readJson('requests.json');
const golden = await readJson('golden.json');
const unlabeled = structuredClone(golden);
unlabeled.cases.forEach(c => { c.targets = {}; });
const partial = structuredClone(golden);
delete partial.cases[1].targets.a_noul;
const drift = structuredClone(golden);
drift.cases[0].expected.z_choice = { billing: 0.999, returns: 0.0005, shipping: 0.0005 };
const pending = await readJson('huncho-model.json');
pending.calibration.default.status = 'pending';
pending.calibration.entries['onnx:fp32'].status = 'pending';
for (const [name, value] of Object.entries({ unlabeled, partial, drift, pending })) {
  await writeFile(resolve(generated, `${name}.json`), JSON.stringify(value));
}
const server = createServer(async (request, response) => {
  try {
    const path = resolve(root, `.${decodeURIComponent(new URL(request.url, 'http://local').pathname)}`);
    if (!path.startsWith(root + sep)) throw new Error('outside test assets');
    const bytes = path === resolve(root, 'index.html')
      ? Buffer.from('<!doctype html><title>Huncho CPU WASM tests</title>') : await readFile(path);
    response.writeHead(200, { 'Content-Type': ({ '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm',
      '.json': 'application/json', '.html': 'text/html' })[extname(path)] ?? 'application/octet-stream',
      'Cache-Control': 'no-store' });
    response.end(bytes);
  } catch {
    response.writeHead(404);
    response.end();
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}`;
async function asset(path) {
  return { url: `${base}/${path}`, sha256: createHash('sha256').update(await readFile(resolve(root, path))).digest('hex') };
}
const config = {
  provider: 'wasm', head: 'graph-integrated-f1-v1',
  manifest: await asset('tests/generated/huncho-model.json'),
  tokenizer: await asset('tests/generated/tokenizer.json'), model: await asset('tests/generated/model.onnx'),
  golden: await asset('tests/generated/golden.json'),
  core: { module: await asset('pkg/huncho_browser_core.js'), wasm: await asset('pkg/huncho_browser_core_bg.wasm') },
  runtime: { module: await asset('node_modules/onnxruntime-web/dist/ort.wasm.min.mjs'),
    loader: await asset('node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs'),
    wasm: await asset('node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm') },
};
const bundle = `tests/generated/bundle-${Date.now()}`;
const packageArgs = Object.entries({ manifest: 'huncho-model.json', tokenizer: 'tokenizer.json',
  model: 'model.onnx', golden: 'golden.json' }).flatMap(([flag, name]) => [`--${flag}`, resolve(generated, name)]);
const packaged = JSON.parse((await promisify(execFile)(process.execPath,
  [resolve(root, 'scripts/package.mjs'), ...packageArgs, '--out', resolve(root, bundle)])).stdout);
let browser;
const checks = [];
function close(actual, expected, path = '') {
  if (typeof expected === 'number') {
    assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-5, `${path}: ${actual} vs ${expected}`);
  } else if (expected && typeof expected === 'object') {
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), path);
    for (const key of Object.keys(expected)) close(actual[key], expected[key], `${path}.${key}`);
  } else assert.equal(actual, expected, path);
}
try {
  browser = await puppeteer.launch({ executablePath: process.env.HUNCHO_TEST_BROWSER ?? '/usr/bin/google-chrome',
    headless: true, args: ['--disable-gpu', '--disable-software-rasterizer', '--disable-webgl'] });
  const page = await browser.newPage();
  page.on('pageerror', error => process.stderr.write(`browser error: ${error}\n`));
  await page.goto(`${base}/index.html`);
  await page.evaluate(async config => {
    const { HunchoBrowser } = await import('/src/index.mjs');
    window.HunchoBrowser = HunchoBrowser;
    window.engine = await HunchoBrowser.load(config);
  }, config);
  const report = await page.evaluate(() => engine.qualification);
  assert.equal(report.passed, true);
  assert.equal(report.device, 'WASM');
  assert.equal(report.backend, 'onnx');
  assert.equal(report.dtype, 'fp32');
  assert.equal(report.outcome_calibration.questions, 12);
  assert.equal(report.work.forward_calls, 12);
  assert.equal(report.work.result_cache_hits, 0);
  assert.equal(report.execution_metadata.onnx_execution_provider, 'wasm');
  assert.equal(report.execution_metadata.model_sha256, config.model.sha256);
  assert.ok(report.max_prob_delta <= 1e-5);
  checks.push('fresh actual CPU WASM inference for every labeled fixture question');
  for (let i = 0; i < requests.length; i++) {
    const result = await page.evaluate(request => engine.evalWithStats(request, { extensions: true }), requests[i]);
    close(result.response, reference.responses[i], `case ${i}`);
    assert.equal(result.work.forward_calls, 3);
    assert.equal(result.work.processed_tokens, result.response.usage.input_tokens);
    assert.equal(result.response.usage.output_tokens, 0);
  }
  checks.push('all typed answers, per-type temperatures, raw logits, confidence, legend and logical usage');
  const plain = await page.evaluate(request => engine.eval(request), requests[0]);
  assert.equal('extensions' in plain, false);
  close(plain.answers, reference.responses[0].answers);
  checks.push('extensions opt-in and complete native wire parity');

  // Independently load the shared WASM core to check exact tokenizer inputs,
  // immutable/single-use plan handles and shape/nonfinite cleanup.
  const coreChecks = await page.evaluate(async ({ requests, reference, config }) => {
    const module = await import('/pkg/huncho_browser_core.js');
    await module.default({ module_or_path: new Uint8Array(await (await fetch(config.core.wasm.url)).arrayBuffer()) });
    const core = new module.MarkerEngine(await (await fetch(config.manifest.url)).text(),
      await (await fetch(config.tokenizer.url)).text(), JSON.stringify({ onnx_execution_provider: 'wasm', onnx_web_version: '1.30.0' }));
    const plans = requests.map(request => JSON.parse(core.prepare(JSON.stringify(request), true)));
    const inputs = plans.map(plan => plan.readouts);
    const finished = core.finish(plans[0].handle, JSON.stringify(Object.keys(requests[0].questions)
      .map(id => reference.responses[0].extensions.raw_logits[id])));
    let reuse = false;
    try { core.finish(plans[0].handle, '[]'); } catch { reuse = true; }
    let nonfinite = false;
    try { core.finish(plans[1].handle, '[[null]]'); } catch { nonfinite = true; }
    const consumed = !core.cancel(plans[1].handle);
    for (const plan of plans) core.cancel(plan.handle);
    core.free();
    return { inputs, finished: JSON.parse(finished), reuse, nonfinite, consumed };
  }, { requests, reference, config });
  assert.deepEqual(coreChecks.inputs, reference.readouts);
  assert.deepEqual(coreChecks.finished.response, reference.responses[0]);
  assert.equal(coreChecks.reuse && coreChecks.nonfinite && coreChecks.consumed, true);
  checks.push('exact native/WASM tokenizer and prompt parity; consumed and invalid plan cleanup');

  async function rejectedLoad(variant, pattern) {
    const error = await page.evaluate(async config => {
      try { const loaded = await HunchoBrowser.load(config); await loaded.dispose(); return ''; }
      catch (error) { return String(error); }
    }, variant);
    assert.match(error, pattern);
  }
  for (const name of ['unlabeled', 'partial']) {
    await rejectedLoad({ ...config, golden: await asset(`tests/generated/${name}.json`) }, /complete observed-label/);
  }
  await rejectedLoad({ ...config, golden: await asset('tests/generated/drift.json') }, /conformance failed/);
  await rejectedLoad({ ...config, manifest: await asset('tests/generated/pending.json') }, /fitted F1/);
  await rejectedLoad({ ...config, model: await asset('tests/generated/nonfinite.onnx') }, /nonfinite/);
  await rejectedLoad({ ...config, model: await asset('tests/generated/wrong-output.onnx') }, /raw FP32 scores/);
  await rejectedLoad({ ...config, model: { ...config.model, sha256: '0'.repeat(64) } }, /SHA-256 mismatch/);
  await rejectedLoad({ ...config, provider: 'webgpu' }, /CPU WASM/);
  checks.push('unlabeled, incomplete, drifted, pending, nonfinite, substituted and GPU-provider loads refuse serving');

  const packageResult = await page.evaluate(async ({ url, sha256, request }) => {
    const { HunchoBrowser } = await import(new URL('index.mjs', url).href);
    const loaded = await HunchoBrowser.loadPackage({ url, sha256 });
    const response = await loaded.eval(request);
    const report = loaded.qualification;
    await loaded.dispose();
    let bad = '';
    try { await HunchoBrowser.loadPackage({ url, sha256: '0'.repeat(64) }); }
    catch (error) { bad = String(error); }
    return { response, report, bad };
  }, { url: `${base}/${bundle}/config.json`, sha256: packaged.config_sha256, request: requests[0] });
  close(packageResult.response.answers, reference.responses[0].answers);
  assert.equal(packageResult.report.execution_metadata.package_sha256, packaged.config_sha256);
  assert.equal(packageResult.report.work.forward_calls, 12);
  assert.match(packageResult.bad, /SHA-256 mismatch/);
  const overwrite = await promisify(execFile)(process.execPath,
    [resolve(root, 'scripts/package.mjs'), ...packageArgs, '--out', resolve(root, bundle)])
    .then(() => false, error => error.code !== 0 && /EEXIST/.test(error.stderr));
  assert.equal(overwrite, true);
  checks.push('self-contained bundle, pinned descriptor/SDK, fresh session qualification and overwrite refusal');

  const errors = await page.evaluate(async request => {
    const invalids = [ { ...request, model: 'alias' }, { ...request, questions: {} },
      { ...request, images: [] }, { ...request, questions: { bad: { type: 'choice', criteria: {} } } },
      { ...request, questions: { bad: { type: 'choice', criteria: Object.fromEntries(Array.from({length:9}, (_,i)=>[`option${i}`, null])) } } },
      { ...request, state: 'x'.repeat(1024 * 1024) },
      { ...request, questions: Object.fromEntries(Array.from({length:65}, (_,i)=>[`question${i}`, {type:'noul'}])) } ];
    const errors = [];
    for (const invalid of invalids) {
      try { await engine.eval(invalid); errors.push(''); } catch (error) { errors.push(String(error)); }
    }
    for (const options of [{prefix_cache:true}, {extensions:'yes'}]) {
      try { await engine.eval(request, options); errors.push(''); } catch (error) { errors.push(String(error)); }
    }
    const recovered = await engine.eval(request);
    return { errors, recovered };
  }, requests[0]);
  assert.ok(errors.errors.every(Boolean));
  close(errors.recovered.answers, reference.responses[0].answers);
  checks.push('invalid model, empty requests/candidates, multimodal input and option budgets fail; next evaluation succeeds');
  const queued = await page.evaluate(async request => {
    const promises = Array.from({length:8}, () => engine.eval(request));
    let overflow = '';
    try { await engine.eval(request); } catch (error) { overflow = String(error); }
    // Serialized input is frozen when submitted, before another caller mutates it.
    request.state = 'mutated after submission';
    const responses = await Promise.all(promises);
    const mutatedReport = engine.qualification;
    mutatedReport.passed = false;
    const immutable = engine.qualification.passed;
    const final = engine.eval(responses.length ? { ...request, state: 'customer wants a refund' } : request);
    const first = engine.dispose();
    const second = engine.dispose();
    await final;
    await first;
    let disposed = '';
    try { await engine.eval(request); } catch (error) { disposed = String(error); }
    return { responses, overflow, immutable, same: first === second, disposed };
  }, structuredClone(requests[0]));
  queued.responses.forEach(response => close(response.answers, reference.responses[0].answers));
  assert.match(queued.overflow, /queue is full/);
  assert.equal(queued.immutable, true);
  assert.equal(queued.same, true);
  assert.match(queued.disposed, /disposed/);
  checks.push('bounded queue, input snapshots, immutable qualification and drain-before-dispose');
  const summary = { qualified: false, synthetic_fixture_only: true, actual_runtime: 'ONNX Runtime Web 1.30.0 CPU WASM',
    browser: await browser.version(), gpu_checks: false, checks, report, package_report: packageResult.report,
    note: 'Synthetic labels test gate plumbing only; no released-model calibration or speed acceptance.' };
  if (process.env.HUNCHO_BROWSER_TEST_REPORT) {
    await writeFile(process.env.HUNCHO_BROWSER_TEST_REPORT, JSON.stringify(summary, null, 2) + '\n');
  }
  process.stdout.write(JSON.stringify(summary, null, 2) + '\n');
} finally {
  try { if (browser) await browser.close(); }
  finally { await new Promise(resolve => server.close(resolve)); }
}
