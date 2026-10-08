// An optional browser runtime. Prompts, calibration, answers and conformance
// remain in huncho-core; ORT executes the graph's actual integrated scalar head.
const VERSION = '1.30.0';
const MiB = 1024 * 1024;
const constructorKey = Symbol('qualified Huncho browser instance');
const workerConstructorKey = Symbol('qualified Huncho worker instance');
let runtime;
let coreModule;

function snapshot(value) {
  return JSON.parse(JSON.stringify(value));
}

async function asset(spec, limit, label) {
  if (!spec || typeof spec.url !== 'string' || !/^[a-f0-9]{64}$/.test(spec.sha256)) {
    throw new Error(`${label} needs a URL and lowercase SHA-256`);
  }
  const response = await fetch(spec.url, { cache: 'no-store', signal: AbortSignal.timeout(120_000) });
  if (!response.ok || !response.body) throw new Error(`${label} fetch failed: ${response.status}`);
  const reader = response.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.length;
      if (size > limit) throw new Error(`${label} exceeds the browser asset budget`);
      chunks.push(value);
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    x => x.toString(16).padStart(2, '0')).join('');
  if (digest !== spec.sha256) throw new Error(`${label} SHA-256 mismatch`);
  return bytes;
}

async function importBytes(bytes) {
  const url = URL.createObjectURL(new Blob([bytes], { type: 'text/javascript' }));
  try {
    return await import(/* webpackIgnore: true */ url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function packageConfig(spec) {
  spec = snapshot(spec);
  const bytes = await asset(spec, 64 * 1024, 'package descriptor');
  const config = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  const base = new URL(spec.url, globalThis.location.href);
  const assets = [config.manifest, config.tokenizer, config.model, config.golden,
    config.core?.module, config.core?.wasm, config.runtime?.module, config.runtime?.loader, config.runtime?.wasm];
  if (config.worker) assets.push(config.worker.module);
  for (const entry of assets) {
    if (!entry || typeof entry.url !== 'string') throw new Error('invalid browser package descriptor');
    entry.url = new URL(entry.url, base).href;
  }
  if (!config.sdk || !/^[a-f0-9]{64}$/.test(config.sdk.sha256)) {
    throw new Error('browser package must pin its SDK module');
  }
  const sdkBytes = await asset({ url: import.meta.url, sha256: config.sdk.sha256 }, 2 * MiB, 'executing SDK module');
  config.sdk_sha256 = config.sdk.sha256;
  config.package_sha256 = spec.sha256;
  return { config, sdkBytes };
}

function evaluationInput(request, options) {
  if (!options || Object.keys(options).some(key => key !== 'extensions')) {
    throw new Error('browser evaluations support only the extensions option');
  }
  const { extensions = false } = options;
  if (typeof extensions !== 'boolean') throw new Error('extensions must be boolean');
  const json = JSON.stringify(request);
  if (typeof json !== 'string' || new TextEncoder().encode(json).length > MiB) {
    throw new Error('browser request exceeds the memory budget or is not serializable');
  }
  return { json, extensions };
}

function initializeCore(config, moduleBytes, wasmBytes) {
  const identity = `${config.module.sha256}:${config.wasm.sha256}`;
  if (coreModule && coreModule.identity !== identity) {
    throw new Error('one page must use a single pinned huncho-core build');
  }
  if (!coreModule) {
    coreModule = { identity, promise: (async () => {
      const module = await importBytes(moduleBytes);
      await module.default({ module_or_path: wasmBytes });
      return module;
    })() };
  }
  return coreModule.promise;
}

function initializeRuntime(config, moduleBytes, loaderBytes, wasmBytes) {
  const identity = `${config.module.sha256}:${config.loader.sha256}:${config.wasm.sha256}`;
  if (runtime && runtime.identity !== identity) {
    throw new Error('one page must use a single pinned ORT WASM build');
  }
  if (!runtime) {
    runtime = { identity, promise: (async () => {
      const ort = await importBytes(moduleBytes);
      if (ort.env.versions.web !== VERSION) throw new Error('unsupported ONNX Runtime Web version');
      const loaderUrl = URL.createObjectURL(new Blob([loaderBytes], { type: 'text/javascript' }));
      // Single-thread fixed SIMD CPU only. No WebGPU/Metal import, probe or EP.
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.simd = 'fixed';
      ort.env.wasm.wasmBinary = wasmBytes;
      ort.env.wasm.wasmPaths = { mjs: loaderUrl };
      ort.env.logLevel = 'warning';
      return { ort, loaderUrl };
    })() };
  }
  // ORT retains a process-global WASM runtime. Its verified loader URL is kept
  // for the page lifetime; release() below only destroys model sessions.
  return runtime.promise;
}

function validateGraph(session, batched) {
  const inputs = new Map(session.inputMetadata.map(x => [x.name, x]));
  const outputs = session.outputMetadata;
  const dynamic = x => typeof x === 'string' && x.length > 0;
  function input(name, shape) {
    const meta = inputs.get(name);
    if (!meta?.isTensor || meta.type !== 'int64' || meta.shape.length !== shape.length ||
        meta.shape.some((x, i) => shape[i] === '*' ? !dynamic(x) : x !== shape[i])) {
      throw new Error(`integrated marker graph has invalid ${name} metadata`);
    }
  }
  input('tokens', [batched ? '*' : 1, '*']);
  input('positions', batched ? ['*', 2] : ['*']);
  input('qtype', [batched ? '*' : 1]);
  if (inputs.has('attention_mask')) input('attention_mask', [batched ? '*' : 1, '*']);
  if (inputs.size !== (inputs.has('attention_mask') ? 4 : 3) ||
      outputs.length !== 1 || outputs[0].name !== 'scores' ||
      !outputs[0].isTensor || outputs[0].type !== 'float32' ||
      outputs[0].shape.length !== 2 || !dynamic(outputs[0].shape[0]) || outputs[0].shape[1] !== 1) {
    throw new Error('expected only raw FP32 scores [markers,1] from the integrated F1 head');
  }
  return { mask: inputs.has('attention_mask'), batched };
}

export class HunchoBrowser {
  #core;
  #session;
  #ort;
  #mask;
  #batched;
  #queue = Promise.resolve();
  #pending = 0;
  #closed = false;
  #disposal;
  #report;

  constructor(key, core, session, ort, graph) {
    if (key !== constructorKey) throw new Error('use HunchoBrowser.load() for fresh labeled qualification');
    this.#core = core;
    this.#session = session;
    this.#ort = ort;
    this.#mask = graph.mask;
    this.#batched = graph.batched;
  }

  static async loadPackage(spec) {
    const { config } = await packageConfig(spec);
    return HunchoBrowser.load(config);
  }

  static async load(config) {
    config = snapshot(config);
    const batched = config.head === 'graph-integrated-f1-batch-v1';
    if (config.provider !== 'wasm' || (!batched && config.head !== 'graph-integrated-f1-v1')) {
      throw new Error('this increment supports CPU WASM with an integrated F1 scalar or row/marker head only');
    }
    if (batched ? !config.batch || Object.keys(config.batch).some(key => !['max_tokens', 'padding_percent'].includes(key)) ||
        !Number.isInteger(config.batch.max_tokens) || config.batch.max_tokens < 1 || config.batch.max_tokens > 65536 ||
        !Number.isInteger(config.batch.padding_percent) || config.batch.padding_percent < 0 || config.batch.padding_percent > 100
      : config.batch !== undefined) {
      throw new Error('immutable browser batches require 1..65536 tokens and padding percent 0..100');
    }
    const specs = [config.manifest, config.tokenizer, config.model, config.golden,
      config.core?.module, config.core?.wasm, config.runtime?.module, config.runtime?.loader, config.runtime?.wasm];
    const limits = [1, 16, 64, 32, 2, 32, 2, 2, 64];
    const labels = ['manifest', 'tokenizer', 'model', 'golden', 'core module', 'core WASM',
      'ORT module', 'ORT loader', 'ORT WASM'];
    const bytes = await Promise.all(specs.map((spec, i) => asset(spec, limits[i] * MiB, labels[i])));
    const decode = data => new TextDecoder('utf-8', { fatal: true }).decode(data);
    const [module, { ort }] = await Promise.all([
      initializeCore(config.core, bytes[4], bytes[5]),
      initializeRuntime(config.runtime, bytes[6], bytes[7], bytes[8]),
    ]);
    const identity = { onnx_execution_provider: 'wasm', onnx_web_version: VERSION,
      native_execution: 'onnxruntime-web-integrated-f1-v1', browser_user_agent: navigator.userAgent,
      wasm_threads: '1', wasm_simd: 'fixed', graph_optimization: 'all' };
    if (batched) Object.assign(identity, {
      native_execution: 'onnxruntime-web-integrated-f1-batch-v1',
      external_marker_scores: config.head, onnx_native_batch: 'wasm-f1-row-marker-v1',
      browser_batch_tokens: String(config.batch.max_tokens),
      browser_batch_padding_percent: String(config.batch.padding_percent),
    });
    identity.browser_execution = typeof document === 'undefined' &&
      typeof DedicatedWorkerGlobalScope !== 'undefined' && globalThis instanceof DedicatedWorkerGlobalScope
      ? 'dedicated-worker-v1' : 'calling-thread-v1';
    if (config.worker?.module) identity.worker_module_sha256 = config.worker.module.sha256;
    if (config.package_sha256) identity.package_sha256 = config.package_sha256;
    if (config.sdk_sha256) identity.sdk_sha256 = config.sdk_sha256;
    labels.forEach((label, i) => { identity[`${label.replaceAll(' ', '_')}_sha256`] = specs[i].sha256; });
    const core = new module.MarkerEngine(decode(bytes[0]), decode(bytes[1]), JSON.stringify(identity));
    let session;
    let instance;
    try {
      const golden = decode(bytes[3]);
      // Reject incomplete/unlabeled suites before allocating a model session.
      const requests = JSON.parse(core.requests_for_qualification(golden));
      session = await ort.InferenceSession.create(bytes[2], {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all', executionMode: 'sequential',
      });
      const graph = validateGraph(session, batched);
      if (batched && config.batch.padding_percent > 0 && !graph.mask) {
        throw new Error('padded browser batches require an explicit original-length attention mask');
      }
      instance = new HunchoBrowser(constructorKey, core, session, ort, graph);
      const responses = [];
      const work = {};
      const independent = [];
      const independentWork = {};
      for (const request of requests) {
        const result = await instance.#execute(JSON.stringify(request), false);
        responses.push(result.response);
        for (const [key, value] of Object.entries(result.work)) work[key] = (work[key] ?? 0) + value;
        if (batched) {
          const scalar = await instance.#execute(JSON.stringify(request), false, true);
          independent.push(scalar.response);
          for (const [key, value] of Object.entries(scalar.work)) independentWork[key] = (independentWork[key] ?? 0) + value;
        }
      }
      const report = JSON.parse(batched
        ? core.qualify_batched(golden, JSON.stringify(responses), JSON.stringify(work),
          JSON.stringify(independent), JSON.stringify(independentWork))
        : core.qualify(golden, JSON.stringify(responses), JSON.stringify(work)));
      if (!report.passed || !report.outcome_calibration) {
        throw new Error(`browser labeled conformance failed: delta=${report.max_prob_delta}, argmax=${report.argmax_agreement}, ECE drift=${report.ece}, paired delta=${report.optimization_parity?.max_prob_delta}`);
      }
      instance.#report = report;
      return instance;
    } catch (error) {
      if (instance) await instance.dispose();
      else {
        try { if (session) await session.release(); }
        finally { core.free(); }
      }
      throw error;
    }
  }

  get qualification() { return snapshot(this.#report); }

  async #execute(requestJson, extensions, independent = false) {
    const plan = JSON.parse(independent ? this.#core.prepare_independent(requestJson, extensions)
      : this.#core.prepare(requestJson, extensions));
    try {
      if (this.#batched) return await this.#executeBatches(plan);
      const scores = [];
      for (const input of plan.readouts) {
        const tensors = [];
        const tensor = (values, dims) => {
          const value = new this.#ort.Tensor('int64', BigInt64Array.from(values, BigInt), dims);
          tensors.push(value);
          return value;
        };
        const feeds = {
          tokens: tensor(input.tokens, [1, input.tokens.length]),
          positions: tensor(input.positions, [input.positions.length]),
          qtype: tensor([input.qtype], [1]),
        };
        if (this.#mask) feeds.attention_mask = tensor(new Array(input.tokens.length).fill(1), [1, input.tokens.length]);
        let outputs;
        try {
          outputs = await this.#session.run(feeds, ['scores']);
          const output = outputs.scores;
          if (output.type !== 'float32' || output.dims.length !== 2 ||
              output.dims[0] !== input.positions.length || output.dims[1] !== 1) {
            throw new Error('integrated head returned the wrong marker-score shape');
          }
          const data = await output.getData();
          if (!(data instanceof Float32Array) || data.some(x => !Number.isFinite(x))) {
            throw new Error('integrated head returned nonfinite or non-FP32 scores');
          }
          scores.push(Array.from(data));
        } finally {
          if (outputs) Object.values(outputs).forEach(value => value.dispose());
          tensors.forEach(value => value.dispose());
        }
      }
      return JSON.parse(this.#core.finish(plan.handle, JSON.stringify(scores)));
    } finally {
      this.#core.cancel(plan.handle);
    }
  }

  async #executeBatches(plan) {
    const scores = new Array(plan.readouts.length);
    const groups = plan.groups ?? plan.readouts.map((input, i) => ({ readouts: [i], sequence: input.tokens.length }));
    for (const group of groups) {
      const tokens = new Array(group.sequence * group.readouts.length).fill(0);
      const mask = this.#mask ? new Array(tokens.length).fill(0) : undefined;
      const positions = [];
      const types = [];
      group.readouts.forEach((index, row) => {
        const input = plan.readouts[index];
        input.tokens.forEach((token, i) => {
          tokens[row * group.sequence + i] = token;
          if (mask) mask[row * group.sequence + i] = 1;
        });
        input.positions.forEach(position => positions.push(row, position));
        types.push(input.qtype);
      });
      const markers = positions.length / 2;
      const tensors = [];
      const tensor = (values, dims) => {
        const value = new this.#ort.Tensor('int64', BigInt64Array.from(values, BigInt), dims);
        tensors.push(value);
        return value;
      };
      let outputs;
      try {
        const feeds = {
          tokens: tensor(tokens, [group.readouts.length, group.sequence]),
          positions: tensor(positions, [markers, 2]),
          qtype: tensor(types, [group.readouts.length]),
        };
        if (mask) feeds.attention_mask = tensor(mask, [group.readouts.length, group.sequence]);
        outputs = await this.#session.run(feeds, ['scores']);
        const output = outputs.scores;
        if (output.type !== 'float32' || output.dims.length !== 2 ||
            output.dims[0] !== markers || output.dims[1] !== 1) {
          throw new Error('integrated batch head returned the wrong marker-score shape');
        }
        const data = await output.getData();
        if (!(data instanceof Float32Array) || data.some(x => !Number.isFinite(x))) {
          throw new Error('integrated batch head returned nonfinite or non-FP32 scores');
        }
        let offset = 0;
        for (const index of group.readouts) {
          const count = plan.readouts[index].positions.length;
          scores[index] = Array.from(data.subarray(offset, offset + count));
          offset += count;
        }
      } finally {
        if (outputs) Object.values(outputs).forEach(value => value.dispose());
        tensors.forEach(value => value.dispose());
      }
    }
    return JSON.parse(this.#core.finish(plan.handle, JSON.stringify(scores)));
  }

  evalWithStats(request, options = {}) {
    if (this.#closed) return Promise.reject(new Error('browser engine is disposed'));
    if (this.#pending >= 8) return Promise.reject(new Error('browser evaluation queue is full'));
    let input;
    try { input = evaluationInput(request, options); }
    catch (error) { return Promise.reject(error); }
    this.#pending++;
    const result = this.#queue.then(() => this.#execute(input.json, input.extensions));
    this.#queue = result.then(() => {}, () => {}).finally(() => { this.#pending--; });
    return result;
  }

  async eval(request, options) { return (await this.evalWithStats(request, options)).response; }

  dispose() {
    if (!this.#disposal) {
      this.#closed = true;
      this.#disposal = this.#queue.then(async () => {
        try { await this.#session.release(); }
        finally { this.#core.free(); }
      });
    }
    return this.#disposal;
  }
}

// One dedicated single-thread CPU runtime per instance. Only frozen JSON and
// copied wire results cross the boundary; no unqualified engine is exposed.
export class HunchoBrowserWorker {
  #worker;
  #urls;
  #requests = new Map();
  #next = 1;
  #closed = false;
  #disposal;
  #report;

  constructor(key, worker, urls) {
    if (key !== workerConstructorKey) throw new Error('use HunchoBrowserWorker.loadPackage()');
    this.#worker = worker;
    this.#urls = urls;
    worker.onerror = event => {
      event.preventDefault();
      this.#fail(new Error(event.message || 'browser worker failed'));
    };
    worker.onmessageerror = () => this.#fail(new Error('browser worker message could not be decoded'));
    worker.onmessage = ({ data }) => {
      const pending = data && this.#requests.get(data.id);
      if (!pending || typeof data.ok !== 'boolean') {
        this.#fail(new Error('browser worker returned an invalid protocol response'));
        return;
      }
      this.#requests.delete(data.id);
      if (data.ok) pending.resolve(data.value);
      else pending.reject(new Error(typeof data.error === 'string' ? data.error : 'browser worker request failed'));
    };
  }

  static async loadPackage(spec) {
    const { config, sdkBytes } = await packageConfig(spec);
    const workerBytes = await asset(config.worker?.module, 2 * MiB, 'worker module');
    const urls = [workerBytes, sdkBytes].map(bytes =>
      URL.createObjectURL(new Blob([bytes], { type: 'text/javascript' })));
    let instance;
    let timer;
    try {
      instance = new HunchoBrowserWorker(workerConstructorKey,
        new Worker(urls[0], { type: 'module', name: 'Huncho CPU inference' }), urls);
      const report = await Promise.race([
        instance.#send('init', { config, sdkUrl: urls[1] }),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('browser worker initialization timed out')), 120_000); }),
      ]);
      if (!report?.passed || !report.outcome_calibration ||
          report.execution_metadata?.browser_execution !== 'dedicated-worker-v1' ||
          report.execution_metadata?.worker_module_sha256 !== config.worker.module.sha256 ||
          report.execution_metadata?.package_sha256 !== config.package_sha256) {
        throw new Error('browser worker lacks fresh matching labeled qualification');
      }
      instance.#report = snapshot(report);
      return instance;
    } catch (error) {
      if (instance) instance.#fail(error);
      else urls.forEach(url => URL.revokeObjectURL(url));
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  #send(kind, payload = {}) {
    if (!Number.isSafeInteger(this.#next)) return Promise.reject(new Error('browser worker request ids exhausted'));
    const id = this.#next++;
    return new Promise((resolve, reject) => {
      this.#requests.set(id, { resolve, reject });
      try { this.#worker.postMessage({ id, kind, ...payload }); }
      catch (error) { this.#fail(error); }
    });
  }

  #fail(error) {
    this.#closed = true;
    this.#worker.terminate();
    this.#urls.forEach(url => URL.revokeObjectURL(url));
    this.#urls = [];
    for (const pending of this.#requests.values()) pending.reject(error);
    this.#requests.clear();
  }

  get qualification() { return snapshot(this.#report); }

  evalWithStats(request, options = {}) {
    if (this.#closed) return Promise.reject(new Error('browser worker is disposed'));
    if (this.#requests.size >= 8) return Promise.reject(new Error('browser evaluation queue is full'));
    let input;
    try { input = evaluationInput(request, options); }
    catch (error) { return Promise.reject(error); }
    return this.#send('eval', input);
  }

  async eval(request, options) { return (await this.evalWithStats(request, options)).response; }

  dispose() {
    if (!this.#disposal) {
      if (this.#closed) this.#disposal = Promise.resolve();
      else {
        this.#closed = true;
        this.#disposal = this.#send('dispose').finally(() => {
          this.#worker.terminate();
          this.#urls.forEach(url => URL.revokeObjectURL(url));
          this.#urls = [];
        });
      }
    }
    return this.#disposal;
  }

  // Immediately abandon queued/in-flight work and release the worker runtime.
  terminate() { this.#fail(new Error('browser worker terminated')); }
}
