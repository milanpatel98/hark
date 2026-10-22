// The streaming Zipformer transducer, run with ONNX Runtime (in the browser: WebAssembly).
export class Model {
  // load(name) -> Uint8Array of the model file; config: model/config.json written by `hark export`
  static async create(ort, load, config, { threads = 1 } = {}) {
    const m = new Model();
    m.ort = ort;
    const opts = { executionProviders: ["wasm"], intraOpNumThreads: threads, interOpNumThreads: 1, graphOptimizationLevel: "all" };
    [m.enc, m.dec, m.join] = await Promise.all(["encoder", "decoder", "joiner"].map(async (n) => ort.InferenceSession.create(await load(n), opts)));
    Object.assign(m, { T: config.T, shift: config.shift, layers: config.layers, encDims: config.encDims, attDims: config.attDims,
      left: config.left, kernels: config.kernels, context: config.context, tokens: config.tokens, blank: 0 });
    m.unk = m.tokens.indexOf("<unk>");
    m.stateNames = m.enc.inputNames.slice(1);
    return m;
  }

  initState() {
    const T = this.ort.Tensor, s = {};
    this.layers.forEach((n, i) => {
      const d = this.encDims[i], a = this.attDims[i], l = this.left[i], k = this.kernels[i];
      s[`cached_len_${i}`] = new T("int64", new BigInt64Array(n), [n, 1]);
      s[`cached_avg_${i}`] = new T("float32", new Float32Array(n * d), [n, 1, d]);
      s[`cached_key_${i}`] = new T("float32", new Float32Array(n * l * a), [n, l, 1, a]);
      s[`cached_val_${i}`] = new T("float32", new Float32Array((n * l * a) / 2), [n, l, 1, a / 2]);
      s[`cached_val2_${i}`] = new T("float32", new Float32Array((n * l * a) / 2), [n, l, 1, a / 2]);
      s[`cached_conv1_${i}`] = new T("float32", new Float32Array(n * d * (k - 1)), [n, 1, d, k - 1]);
      s[`cached_conv2_${i}`] = new T("float32", new Float32Array(n * d * (k - 1)), [n, 1, d, k - 1]);
    });
    return this.stateNames.map((n) => s[n]);
  }

  // chunk: Float32Array [T*80] -> { frames: Float32Array[t*512], count: t, state }
  async encode(chunk, state) {
    const feeds = { x: new this.ort.Tensor("float32", chunk, [1, this.T, 80]) };
    this.stateNames.forEach((n, i) => (feeds[n] = state[i]));
    const out = await this.enc.run(feeds);
    const names = this.enc.outputNames;
    const e = out[names[0]];
    return { frames: e.data, count: e.dims[1], state: names.slice(1).map((n) => out[n]) };
  }

  // contexts: array of [a, b] -> Float32Array [N*512]
  async decode(contexts) {
    const flat = BigInt64Array.from(contexts.flat().map((v) => BigInt(v)));
    const out = await this.dec.run({ y: new this.ort.Tensor("int64", flat, [contexts.length, this.context]) });
    return out[this.dec.outputNames[0]].data;
  }

  // enc, dec: Float32Array [N*512] -> Float32Array [N*V]
  async joint(enc, dec, n) {
    const out = await this.join.run({
      encoder_out: new this.ort.Tensor("float32", enc, [n, 512]),
      decoder_out: new this.ort.Tensor("float32", dec, [n, 512]),
    });
    return out[this.join.outputNames[0]];
  }

  text(ids) {
    return ids.map((i) => this.tokens[i]).join("").replaceAll("▁", " ").trim();
  }
}

