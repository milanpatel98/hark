"""Log-mel filterbank features, computed the way Kaldi does (the features the speech model was trained on).

25 ms frames every 10 ms, 80 mel bands from 20 Hz to 7600 Hz, Povey window, no dither, frames centred on
10 ms boundaries (Kaldi's snip_edges=false). Written out in full so the browser can compute exactly the same numbers.
"""

from __future__ import annotations

import numpy as np

SR = 16000
SHIFT = 160          # 10 ms
WIN = 400            # 25 ms
NFFT = 512
BINS = 80
LOW, HIGH = 20.0, SR / 2 - 400.0
PREEMPH = 0.97
EPS = np.finfo(np.float32).eps


def _mel(f):
    return 1127.0 * np.log(1.0 + f / 700.0)


def mel_banks() -> np.ndarray:
    """[BINS, NFFT/2] triangular filters, spaced evenly on the mel scale."""
    lo, hi = _mel(LOW), _mel(HIGH)
    step = (hi - lo) / (BINS + 1)
    freqs = np.arange(NFFT // 2) * (SR / NFFT)
    m = _mel(freqs)
    banks = np.zeros((BINS, NFFT // 2), np.float32)
    for b in range(BINS):
        left, centre, right = lo + b * step, lo + (b + 1) * step, lo + (b + 2) * step
        up = (m - left) / (centre - left)
        down = (right - m) / (right - centre)
        w = np.minimum(up, down)
        banks[b] = np.where((m > left) & (m < right), w, 0.0)
    return banks


WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / (WIN - 1))) ** 0.85
BANKS = mel_banks()


def num_frames(n_samples: int) -> int:
    return (n_samples + SHIFT // 2) // SHIFT


def frame(wave: np.ndarray, i: int) -> np.ndarray:
    """Samples of frame i, mirrored at the signal edges as Kaldi does."""
    start = i * SHIFT + SHIFT // 2 - WIN // 2
    idx = np.arange(start, start + WIN)
    n = len(wave)
    idx = np.where(idx < 0, -idx - 1, idx)
    idx = np.where(idx >= n, 2 * n - 1 - idx, idx)
    return wave[idx]


def features(wave: np.ndarray, start_frame: int = 0, end_frame: int | None = None) -> np.ndarray:
    """[frames, 80] log-mel features for a 16 kHz float waveform in [-1, 1]."""
    w = wave.astype(np.float64)  # samples stay in [-1, 1], as the model was trained
    end = num_frames(len(w)) if end_frame is None else end_frame
    out = np.zeros((max(0, end - start_frame), BINS), np.float32)
    for k, i in enumerate(range(start_frame, end)):
        x = frame(w, i)
        x = x - x.mean()
        x = np.concatenate([[x[0] - PREEMPH * x[0]], x[1:] - PREEMPH * x[:-1]])
        x = x * WINDOW
        spec = np.abs(np.fft.rfft(x, NFFT)[: NFFT // 2]) ** 2
        out[k] = np.log(np.maximum(BANKS @ spec, EPS))
    return out
