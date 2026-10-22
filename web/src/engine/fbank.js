// Log-mel filterbank features, the same numbers as hark/fbank.py (Kaldi style: 25 ms frames every 10 ms, 80 mel bands).
export const SR = 16000, SHIFT = 160, WIN = 400, NFFT = 512, BINS = 80;
const LOW = 20, HIGH = SR / 2 - 400, PREEMPH = 0.97, EPS = 1.1920928955078125e-7;
const mel = (f) => 1127 * Math.log(1 + f / 700);

const WINDOW = Float64Array.from({ length: WIN }, (_, n) => Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (WIN - 1)), 0.85));
const BANKS = (() => {
  const lo = mel(LOW), hi = mel(HIGH), step = (hi - lo) / (BINS + 1);
  const out = [];
  for (let b = 0; b < BINS; b++) {
    const left = lo + b * step, centre = lo + (b + 1) * step, right = lo + (b + 2) * step;
    const w = new Float64Array(NFFT / 2);
    let first = -1, last = -1;
    for (let k = 0; k < NFFT / 2; k++) {
      const m = mel((k * SR) / NFFT);
      if (m > left && m < right) {
        w[k] = Math.min((m - left) / (centre - left), (right - m) / (right - centre));
        if (first < 0) first = k;
        last = k;
      }
    }
    out.push({ w, first, last });
  }
  return out;
})();

// in-place radix-2 FFT on (re, im)
const REV = (() => {
  const r = new Uint16Array(NFFT), bits = Math.log2(NFFT);
  for (let i = 0; i < NFFT; i++) {
    let x = i, y = 0;
    for (let b = 0; b < bits; b++) { y = (y << 1) | (x & 1); x >>= 1; }
    r[i] = y;
  }
  return r;
})();
const COS = Float64Array.from({ length: NFFT / 2 }, (_, i) => Math.cos((-2 * Math.PI * i) / NFFT));
const SIN = Float64Array.from({ length: NFFT / 2 }, (_, i) => Math.sin((-2 * Math.PI * i) / NFFT));
function fft(re, im) {
  for (let i = 0; i < NFFT; i++) {
    const j = REV[i];
    if (j > i) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let size = 2; size <= NFFT; size <<= 1) {
    const half = size >> 1, step = NFFT / size;
    for (let s = 0; s < NFFT; s += size) {
      for (let k = 0; k < half; k++) {
        const c = COS[k * step], si = SIN[k * step];
        const a = s + k, b = a + half;
        const tr = re[b] * c - im[b] * si, ti = re[b] * si + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}

export const numFrames = (n) => Math.floor((n + SHIFT / 2) / SHIFT);

// features for frames [start, end) of a mono 16 kHz Float32Array in [-1, 1].
// `offset` is how many samples have been dropped from the front of `wave` (a live stream keeps only recent audio).
export function features(wave, start = 0, end = numFrames(wave.length), offset = 0) {
  const n = wave.length + offset, out = new Float32Array(Math.max(0, end - start) * BINS);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT), x = new Float64Array(WIN);
  for (let f = start; f < end; f++) {
    const s0 = f * SHIFT + SHIFT / 2 - WIN / 2;
    let mean = 0;
    for (let k = 0; k < WIN; k++) {
      let i = s0 + k;
      if (i < 0) i = -i - 1;
      if (i >= n) i = 2 * n - 1 - i;
      x[k] = wave[i - offset];
      mean += x[k];
    }
    mean /= WIN;
    for (let k = 0; k < WIN; k++) x[k] -= mean;
    re.fill(0); im.fill(0);
    re[0] = (x[0] - PREEMPH * x[0]) * WINDOW[0];
    for (let k = 1; k < WIN; k++) re[k] = (x[k] - PREEMPH * x[k - 1]) * WINDOW[k];
    fft(re, im);
    const row = (f - start) * BINS;
    for (let b = 0; b < BINS; b++) {
      const { w, first, last } = BANKS[b];
      let e = 0;
      for (let k = first; k <= last; k++) e += w[k] * (re[k] * re[k] + im[k] * im[k]);
      out[row + b] = Math.log(Math.max(e, EPS));
    }
  }
  return out;
}
