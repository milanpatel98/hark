// Spelling words in the model's word pieces (same as hark/tokens.py).
export class Pieces {
  constructor(tokens) { this.index = new Map(tokens.map((t, i) => [t, i])); }

  word(w) {
    w = w.toUpperCase();
    const memo = new Map();
    const best = (i, first) => {
      const key = `${i}:${first}`;
      if (memo.has(key)) return memo.get(key);
      let out = null;
      if (i === w.length) out = [];
      else {
        for (let j = w.length; j > i; j--) {
          const piece = (first ? "▁" : "") + w.slice(i, j);
          if (this.index.has(piece)) {
            const rest = best(j, false);
            if (rest !== null && (out === null || 1 + rest.length < out.length)) out = [this.index.get(piece), ...rest];
          }
        }
        if (first && out === null && this.index.has("▁")) {
          const rest = best(i, false);
          if (rest !== null) out = [this.index.get("▁"), ...rest];
        }
      }
      memo.set(key, out);
      return out;
    };
    const r = best(0, true);
    return r && r.length ? r : null;
  }

  phrase(text) {
    const out = [];
    for (const w of text.split(/\s+/).filter(Boolean)) {
      const p = this.word(w);
      if (!p) return null;
      out.push(...p);
    }
    return out;
  }
}
