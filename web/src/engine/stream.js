// A live recognition stream: push 16 kHz audio in, read words out (same as hark/stream.py).
import { BINS, SHIFT, SR, WIN, features, numFrames } from "./fbank.js";

export class Stream {
  constructor(model, search) {
    this.m = model; this.search = search;
    this.wave = new Float32Array(0); this.waveOffset = 0;      // samples kept, and how many were dropped before them
    this.feats = new Float32Array(0); this.nFeats = 0; this.featOffset = 0;
    this.done = 0; this.state = model.initState(); this.finished = false; this.decoded = 0;
    this.busy = Promise.resolve();
  }

  accept(samples) {
    this.busy = this.busy.then(() => this._accept(samples));
    return this.busy;
  }

  get samples() { return this.waveOffset + this.wave.length; }

  async _accept(samples) {
    const w = new Float32Array(this.wave.length + samples.length);
    w.set(this.wave); w.set(samples, this.wave.length);
    this.wave = w;
    const total = this.samples;
    const ready = this.finished ? numFrames(total) : Math.max(0, Math.floor((total - WIN / 2 - SHIFT / 2) / SHIFT));
    if (ready > this.nFeats) {
      const f = features(this.wave, this.nFeats, ready, this.waveOffset);
      const keep = this.nFeats - this.featOffset;
      const all = new Float32Array((keep + ready - this.nFeats) * BINS);
      all.set(this.feats.subarray(0, keep * BINS)); all.set(f, keep * BINS);
      this.feats = all; this.nFeats = ready;
    }
    while (this.nFeats - this.done >= this.m.T) {
      const s0 = (this.done - this.featOffset) * BINS;
      const chunk = this.feats.slice(s0, s0 + this.m.T * BINS);
      const { frames, count, state } = await this.m.encode(chunk, this.state);
      this.state = state;
      await this.search.step(frames, count, this.decoded);
      this.decoded += count; this.done += this.m.shift;
    }
    // drop what no future frame or chunk can need, so an hour of listening stays small
    const dropFeats = this.done - this.featOffset;
    if (dropFeats > 0) { this.feats = this.feats.slice(dropFeats * BINS); this.featOffset = this.done; }
    const firstNeeded = this.nFeats * SHIFT + SHIFT / 2 - WIN / 2;
    const dropWave = firstNeeded - this.waveOffset;
    if (dropWave > 0 && !this.finished) { this.wave = this.wave.slice(dropWave); this.waveOffset += dropWave; }
  }

  finish(tail = 0.66) {
    this.busy = this.busy.then(() => { this.finished = true; return this._accept(new Float32Array(Math.round(tail * SR))); });
    return this.busy;
  }

  get text() { return this.search.text(); }
}
