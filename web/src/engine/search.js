// Turning encoder frames into words: greedy and modified beam search with phrase biasing (same as hark/search.py).

export class Greedy {
  static async create(m) {
    const g = new Greedy();
    g.m = m; g.ids = []; g.times = [];
    g.ctx = [...Array(m.context - 1).fill(-1), m.blank];
    g.dec = await m.decode([g.ctx]);
    return g;
  }
  async step(frames, count, t0) {
    for (let t = 0; t < count; t++) {
      const logits = (await this.m.joint(frames.subarray(t * 512, (t + 1) * 512), this.dec, 1)).data;
      let y = 0;
      for (let k = 1; k < logits.length; k++) if (logits[k] > logits[y]) y = k;
      if (y !== this.m.blank && y !== this.m.unk) {
        this.ids.push(y); this.times.push(t0 + t);
        this.ctx = [...this.ctx.slice(1), y];
        this.dec = await this.m.decode([this.ctx]);
      }
    }
  }
  text() { return this.m.text(this.ids); }
}

// a phrase list as a trie with fallback links; matching pieces earn a bonus, dead ends give it back
export class Bias {
  constructor(phrases, bonus) {
    this.bonus = bonus;
    this.child = [new Map()]; this.depth = [0]; this.end = [false]; this.fail = [0];
    for (const p of phrases) {
      let n = 0;
      for (const t of p) {
        if (!this.child[n].has(t)) {
          this.child.push(new Map()); this.depth.push(this.depth[n] + bonus); this.end.push(false); this.fail.push(0);
          this.child[n].set(t, this.child.length - 1);
        }
        n = this.child[n].get(t);
      }
      this.end[n] = true;
    }
    const queue = [...this.child[0].values()];
    while (queue.length) {
      const n = queue.shift();
      for (const [t, c] of this.child[n]) {
        let f = this.fail[n];
        while (f && !this.child[f].has(t)) f = this.fail[f];
        this.fail[c] = this.child[f].has(t) && this.child[f].get(t) !== c ? this.child[f].get(t) : 0;
        queue.push(c);
      }
    }
  }
  step(state, token) {
    let n = state;
    while (n && !this.child[n].has(token)) n = this.fail[n];
    const nw = this.child[n].get(token) ?? 0;
    const score = this.depth[nw] - this.depth[state];
    return this.end[nw] ? [score, -1 - nw] : [score, nw];
  }
  finish(state) { return state > 0 ? -this.depth[state] : 0; }
}

const norm = (s) => (s < 0 ? 0 : s);

function logSoftmaxRows(data, rows, V) {
  const out = new Float64Array(rows * V);
  for (let r = 0; r < rows; r++) {
    let mx = -Infinity;
    for (let k = 0; k < V; k++) mx = Math.max(mx, data[r * V + k]);
    let s = 0;
    for (let k = 0; k < V; k++) s += Math.exp(data[r * V + k] - mx);
    const lz = mx + Math.log(s);
    for (let k = 0; k < V; k++) out[r * V + k] = data[r * V + k] - lz;
  }
  return out;
}

const logaddexp = (a, b) => (a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b)));

export class Beam {
  static async create(m, { beam = 4, bias = null } = {}) {
    const b = new Beam();
    b.m = m; b.k = beam; b.bias = bias;
    const start = [...Array(m.context - 1).fill(-1), m.blank];
    b.hyps = new Map([[start.join(","), { toks: start, lp: 0, bs: 0, st: 0, times: [] }]]);
    b.cache = new Map();
    return b;
  }

  async decs(hyps) {
    const ctxs = hyps.map((h) => h.toks.slice(-this.m.context));
    const missing = [...new Set(ctxs.map((c) => c.join(",")))].filter((k) => !this.cache.has(k)).sort();
    if (missing.length) {
      const out = await this.m.decode(missing.map((k) => k.split(",").map(Number)));
      missing.forEach((k, i) => this.cache.set(k, out.slice(i * 512, (i + 1) * 512)));
    }
    const res = new Float32Array(hyps.length * 512);
    ctxs.forEach((c, i) => res.set(this.cache.get(c.join(",")), i * 512));
    return res;
  }

  async step(frames, count, t0) {
    for (let t = 0; t < count; t++) {
      const hyps = [...this.hyps.values()];
      const n = hyps.length;
      const enc = new Float32Array(n * 512);
      for (let i = 0; i < n; i++) enc.set(frames.subarray(t * 512, (t + 1) * 512), i * 512);
      const logits = await this.m.joint(enc, await this.decs(hyps), n);
      const V = logits.dims[1];
      const lp = logSoftmaxRows(logits.data, n, V);
      // top k over all (hypothesis, token) pairs
      const cand = [];
      for (let h = 0; h < n; h++) {
        const base = hyps[h].lp + hyps[h].bs;
        for (let k = 0; k < V; k++) {
          const s = lp[h * V + k] + base;
          if (cand.length < this.k) { cand.push([s, h, k]); cand.sort((a, b) => b[0] - a[0]); }
          else if (s > cand[this.k - 1][0]) { cand[this.k - 1] = [s, h, k]; cand.sort((a, b) => b[0] - a[0]); }
        }
      }
      const next = new Map();
      for (const [, h, tok] of cand) {
        const p = hyps[h];
        const lprob = p.lp + lp[h * V + tok];
        let nh;
        if (tok === this.m.blank || tok === this.m.unk) nh = { toks: p.toks, lp: lprob, bs: p.bs, st: p.st, times: p.times };
        else {
          let bs = p.bs, st = p.st;
          if (this.bias) { const [d, s2] = this.bias.step(norm(p.st), tok); bs += d; st = s2; }
          nh = { toks: [...p.toks, tok], lp: lprob, bs, st, times: [...p.times, t0 + t] };
        }
        const key = nh.toks.join(",");
        const o = next.get(key);
        if (o) o.lp = logaddexp(o.lp, nh.lp);
        else next.set(key, nh);
      }
      this.hyps = next;
    }
  }

  best() {
    let best = null, bestScore = -Infinity;
    for (const h of this.hyps.values()) {
      const s = h.lp + h.bs + (this.bias ? this.bias.finish(norm(h.st)) : 0);
      if (s > bestScore) { bestScore = s; best = h; }
    }
    return best;
  }
  get ids() { return this.best().toks.filter((t) => t >= 0 && t !== this.m.blank); }
  text() { return this.m.text(this.ids); }
}
