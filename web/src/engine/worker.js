// Runs the speech engine off the main thread. Messages in: start, audio, bias, stop. Messages out: ready, text, line, done.
import * as ort from "onnxruntime-web/wasm";
import { Model } from "./model.js";
import { Beam, Bias } from "./search.js";
import { Stream } from "./stream.js";
import { Pieces } from "./tokens.js";

ort.env.wasm.numThreads = 1;

let model, pieces, bias = null, stream = null, line = 0, lineStart = 0, started = 0, chunks = 0, busyMs = 0;
const PAUSE_FRAMES = 30;   // 1.2 s with no new word ends a caption line

async function load(base) {
  ort.env.wasm.wasmPaths = `${base}ort/`;
  const cfg = await (await fetch(`${base}model/config.json`)).json();
  model = await Model.create(ort, async (n) => new Uint8Array(await (await fetch(`${base}model/${n}.onnx`)).arrayBuffer()), cfg);
  pieces = new Pieces(model.tokens);
  postMessage({ type: "ready" });
}

async function newLine() {
  const search = await Beam.create(model, { bias });
  if (!stream) stream = new Stream(model, search);
  else stream.search = search;
  lineStart = stream.decoded;
}

onmessage = async ({ data }) => {
  if (data.type === "load") return load(data.base);
  if (data.type === "bias") {
    const phrases = data.words.map((w) => pieces.phrase(w)).filter(Boolean);
    bias = phrases.length ? new Bias(phrases, data.bonus) : null;
    return;
  }
  if (data.type === "start") {
    stream = null; line = 0; started = performance.now(); chunks = 0; busyMs = 0;
    await newLine();
    return;
  }
  if (!stream) return;
  if (data.type === "audio") {
    const t = performance.now();
    await stream.accept(data.samples);
    busyMs += performance.now() - t; chunks++;
    const best = stream.search.best();
    const text = stream.text;
    const last = best.times.length ? best.times[best.times.length - 1] : lineStart;
    if (text && stream.decoded - Math.max(last, lineStart) > PAUSE_FRAMES) {
      postMessage({ type: "line", n: line++, text, start: lineStart * 0.04, end: stream.decoded * 0.04 });
      await newLine();
    } else {
      postMessage({ type: "text", n: line, text, audioSec: stream.samples / 16000, busyMs });
    }
  }
  if (data.type === "stop") {
    await stream.finish();
    if (stream.text) postMessage({ type: "line", n: line++, text: stream.text, start: lineStart * 0.04, end: stream.decoded * 0.04 });
    postMessage({ type: "done", audioSec: stream.samples / 16000, busyMs, wallMs: performance.now() - started });
    stream = null;
  }
};
