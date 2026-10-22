// Learning a speaker's words from their corrections (same as hark/adapt.py).
export const STOP = new Set("A AN THE OF TO AND IN ON AT IS IT I YOU HE SHE WE THEY BE WAS ARE OR FOR WITH AS BY THAT THIS".split(" "));

export const words = (text) => text.toUpperCase().replaceAll("-", " ").replace(/[^A-Z' ]+/g, " ").split(/\s+/).filter(Boolean);

export function align(hyp, ref) {
  const n = hyp.length, m = ref.length;
  const d = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: m + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)));
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (hyp[i - 1] !== ref[j - 1] ? 1 : 0));
  const ops = [];
  let i = n, j = m;
  while (i || j) {
    if (i && j && hyp[i - 1] === ref[j - 1] && d[i][j] === d[i - 1][j - 1]) { ops.push(["ok", i - 1, i, j - 1, j]); i--; j--; }
    else if (i && j && d[i][j] === d[i - 1][j - 1] + 1) { ops.push(["sub", i - 1, i, j - 1, j]); i--; j--; }
    else if (j && d[i][j] === d[i][j - 1] + 1) { ops.push(["del", i, i, j - 1, j]); j--; }
    else { ops.push(["ins", i - 1, i, j, j]); i--; }
  }
  ops.reverse();
  const merged = [];
  for (const op of ops) {
    const last = merged[merged.length - 1];
    if (op[0] !== "ok" && last && last[0] !== "ok") merged[merged.length - 1] = ["err", last[1], op[2], last[3], op[4]];
    else merged.push(op[0] === "ok" ? [...op] : ["err", ...op.slice(1)]);
  }
  return merged;
}

export function werCounts(hyp, ref) {
  let prev = Array.from({ length: ref.length + 1 }, (_, j) => j);
  for (let i = 1; i <= hyp.length; i++) {
    const cur = [i];
    for (let j = 1; j <= ref.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (hyp[i - 1] !== ref[j - 1] ? 1 : 0));
    prev = cur;
  }
  return [prev[ref.length], ref.length];
}

const inc = (map, k, n = 1) => map.set(k, (map.get(k) || 0) + n);
// most_common order: count descending, then first-inserted first (as Python's Counter does)
const mostCommon = (map) => [...map.entries()].map((e, i) => [...e, i]).sort((a, b) => b[1] - a[1] || a[2] - b[2]);

export class Memory {
  // common: { WORD: zipf } for everyday words (data/common.json); only words rarer than maxZipf are listed
  constructor({ minRuleCount = 2, maxSpan = 3, minPhraseCount = 1, maxZipf = 99, common = {} } = {}) {
    this.missed = new Map(); this.heard = new Map(); this.rules = new Map(); this.right = new Map();
    this.minRuleCount = minRuleCount; this.maxSpan = maxSpan; this.minPhraseCount = minPhraseCount; this.corrections = 0;
    this.maxZipf = maxZipf; this.common = common;
  }

  learn(heardText, saidText) {
    const h = words(heardText), r = words(saidText);
    this.corrections++;
    for (const [op, hs, he, rs, re] of align(h, r)) {
      if (op === "ok") {
        inc(this.heard, r[rs]);
        for (let n = 1; n <= this.maxSpan; n++) if (hs - n + 1 >= 0) inc(this.right, h.slice(hs - n + 1, he).join(" "));
      } else {
        for (const w of r.slice(rs, re)) if (w.length > 1 && !STOP.has(w)) inc(this.missed, w);
        if (he - hs > 0 && he - hs <= this.maxSpan && re - rs > 0 && re - rs <= this.maxSpan)
          inc(this.rules, JSON.stringify([h.slice(hs, he).join(" "), r.slice(rs, re).join(" ")]));
      }
    }
  }

  rare(w) { return (this.common[w] || 0) < this.maxZipf; }

  phrases(limit = 300) { return mostCommon(this.missed).filter((e) => e[1] >= this.minPhraseCount && this.rare(e[0])).slice(0, limit).map((e) => e[0]); }

  // words missed once: listed after one more correction, so a single slip never changes what Hark listens for
  pending() { return mostCommon(this.missed).filter((e) => e[1] < this.minPhraseCount && this.rare(e[0])).map((e) => e[0]); }

  activeRules() {
    const out = new Map();
    for (const [k, n] of mostCommon(this.rules)) {
      const [src, dst] = JSON.parse(k);
      if (n >= this.minRuleCount && n > (this.right.get(src) || 0) && !out.has(src)) out.set(src, dst);
    }
    return out;
  }

  rewrite(text) {
    const rules = this.activeRules(), w = words(text);
    if (!rules.size) return w.join(" ");
    const out = [];
    let i = 0;
    while (i < w.length) {
      let hit = false;
      for (let n = this.maxSpan; n > 0; n--) {
        const part = w.slice(i, i + n);
        if (part.length === n && rules.has(part.join(" "))) { out.push(rules.get(part.join(" "))); i += n; hit = true; break; }
      }
      if (!hit) { out.push(w[i]); i++; }
    }
    return out.join(" ");
  }

  toJSON() {
    return { missed: Object.fromEntries(this.missed), rules: [...this.rules].map(([k, n]) => [...JSON.parse(k), n]),
      right: Object.fromEntries(this.right), corrections: this.corrections };
  }

  static fromJSON(o, opts) {
    const m = new Memory(opts);
    m.missed = new Map(Object.entries(o.missed || {}));
    m.rules = new Map((o.rules || []).map(([s, d, n]) => [JSON.stringify([s, d]), n]));
    m.right = new Map(Object.entries(o.right || {}));
    m.corrections = o.corrections || 0;
    return m;
  }
}
