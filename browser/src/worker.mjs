// This verified module runs only inside a dedicated CPU inference Worker.
// The owning SDK supplies a verified Blob SDK URL and hashed absolute assets.
let engine;
let initialized = false;
let closing = false;

self.onmessage = async ({ data }) => {
  const { id, kind } = data ?? {};
  if (!Number.isSafeInteger(id) || id <= 0) throw new Error('invalid worker request id');
  try {
    let value;
    if (kind === 'init' && !initialized) {
      initialized = true;
      if (typeof data.sdkUrl !== 'string' || !data.sdkUrl.startsWith('blob:')) {
        throw new Error('worker initialization requires a verified SDK snapshot');
      }
      const { HunchoBrowser } = await import(data.sdkUrl);
      // Fresh real graph execution and the shared fixed labeled gate happen
      // here. No serialized report from the page can authorize this session.
      engine = await HunchoBrowser.load(data.config);
      value = engine.qualification;
    } else if (!engine || closing) {
      throw new Error('browser worker is uninitialized or disposed');
    } else if (kind === 'eval') {
      if (typeof data.json !== 'string' || typeof data.extensions !== 'boolean') {
        throw new Error('invalid frozen worker evaluation');
      }
      value = await engine.evalWithStats(JSON.parse(data.json), { extensions: data.extensions });
    } else if (kind === 'dispose') {
      closing = true;
      await engine.dispose();
      value = null;
    } else {
      throw new Error('invalid worker operation');
    }
    self.postMessage({ id, ok: true, value });
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error) });
  }
};
