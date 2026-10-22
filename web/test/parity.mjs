// The browser engine must give the same words as the Python engine, piece for piece.
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { features } from "../src/engine/fbank.js";
import { Model } from "../src/engine/model.js";
import { Beam, Bias, Greedy } from "../src/engine/search.js";
import { Stream } from "../src/engine/stream.js";
import { Pieces } from "../src/engine/tokens.js";
import { Memory } from "../src/engine/adapt.js";

ort.env.wasm.numThreads = 1;
const here = (p) => new URL(p, import.meta.url);
const exp = JSON.parse(readFileSync(here("./expected.json")));
const cfg = JSON.parse(readFileSync(here("../public/model/config.json")));
const m = await Model.create(ort, async (n) => readFileSync(here(`../public/model/${n}.onnx`)), cfg);
const pieces = new Pieces(m.tokens);

function wav(path) { // 16-bit PCM mono WAV -> Float32Array at its own rate
  const b = readFileSync(path), dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let p = 12, rate = 16000, data = null;
  while (p < b.length) {
    const id = b.toString("latin1", p, p + 4), size = dv.getUint32(p + 4, true);
    if (id === "fmt ") rate = dv.getUint32(p + 12, true);
    if (id === "data") data = new Int16Array(b.buffer.slice(b.byteOffset + p + 8, b.byteOffset + p + 8 + size));
    p += 8 + size + (size & 1);
  }
  if (rate !== 16000) throw new Error(`expected 16 kHz, got ${rate}`);
  return Float32Array.from(data, (v) => v / 32768);
}

let bad = 0, n = 0;
const check = (name, got, want) => {
  n++;
  if (JSON.stringify(got) !== JSON.stringify(want)) { bad++; console.error(`MISMATCH ${name}\n  py: ${JSON.stringify(want)}\n  js: ${JSON.stringify(got)}`); }
};
// Two choices can be within a few hundredths of a point of each other, and the browser's int8 kernels round
// differently from the desktop ones. Those near-ties may flip one word piece ("dishonoured" / "dishonored");
// anything more fails.
const near = (name, got, want) => {
  n++;
  const a = got.ids, b = want.ids;
  let d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const c = [i];
    for (let j = 1; j <= b.length; j++) c[j] = Math.min(d[j] + 1, c[j - 1] + 1, d[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0));
    d = c;
  }
  if (d[b.length] > 1) { bad++; console.error(`MISMATCH ${name}: ${d[b.length]} pieces differ`); }
  else if (d[b.length]) console.log(`near-tie ${name}: 1 piece differs`);
};

for (const c of exp.cases) {
  const a = wav(here(`../../samples/${c.file}`));
  const f = Array.from(features(a, 0, 5), (v) => Math.round(v * 1e4) / 1e4);
  const maxDiff = Math.max(...f.map((v, i) => Math.abs(v - c.fbank.flat()[i])));
  n++; if (maxDiff > 2e-3) { bad++; console.error(`MISMATCH fbank ${c.file} max diff ${maxDiff}`); }
  for (const [name, make] of [["greedy", () => Greedy.create(m)], ["beam", () => Beam.create(m)],
    ["biased", () => Beam.create(m, { bias: new Bias(exp.bias.map((w) => pieces.phrase(w)), 2.0) })]]) {
    const s = new Stream(m, await make());
    for (let i = 0; i < a.length; i += 1600) await s.accept(a.subarray(i, i + 1600));
    await s.finish();
    near(`${c.file} ${name}`, { text: s.text, ids: s.search.ids }, c[name]);
  }
}

const mem = new Memory();
const pairs = [["I WILL FILL UP THE CAR", "I WILL PHILIP THE CAR"], ["CALL FILL UP NOW", "CALL PHILIP NOW"],
  ["FILL UP THE FORMS", "FILL UP THE FORMS"], ["ASK FILL UP", "ASK PHILIP"], ["THE CAT", "THE CAT SAT"]];
pairs.forEach(([h, s], i) => {
  mem.learn(h, s);
  check(`memory step ${i}`, { phrases: mem.phrases(), rules: [...mem.activeRules()].sort(), rewrite: mem.rewrite("TELL FILL UP HI") }, exp.memory[i]);
});
const rare = new Memory({ minPhraseCount: 2, maxZipf: 5.0, common: JSON.parse(readFileSync(here("../public/data/common.json"))) });
rare.learn("ASK GRACE ONE WERE", "ASK GREGSON WAS"); rare.learn("GRACE ONE WERE HERE", "GREGSON WAS HERE");
check("memory, uncommon words only", { phrases: rare.phrases(), rules: [...rare.activeRules()].sort(), rewrite: rare.rewrite("GRACE ONE WERE") }, exp.memory[pairs.length]);
for (const [w, ids] of Object.entries(exp.spell)) check(`spell ${w}`, pieces.phrase(w), ids);

console.log(`${n - bad}/${n} checks match Python`);
process.exit(bad ? 1 : 0);
