"""A live recognition stream: audio in, words out, as it is spoken."""

from __future__ import annotations

import numpy as np

from . import fbank
from .search import Greedy


class Stream:
    def __init__(self, model, search=None):
        self.m = model
        self.search = search or Greedy(model)
        self.wave = np.zeros(0, np.float32)
        self.feats = np.zeros((0, fbank.BINS), np.float32)
        self.done_frames = 0           # feature frames already handed to the encoder
        self.state = model.init_state()
        self.finished = False
        self.frames_decoded = 0        # encoder output frames, 40 ms each

    def accept(self, samples: np.ndarray) -> None:
        self.wave = np.concatenate([self.wave, samples.astype(np.float32)])
        # a frame is final once the samples it covers (and Kaldi's mirrored edge) have arrived
        ready = fbank.num_frames(len(self.wave)) if self.finished else max(0, (len(self.wave) - fbank.WIN // 2 - fbank.SHIFT // 2) // fbank.SHIFT)
        have = len(self.feats)
        if ready > have:
            self.feats = np.concatenate([self.feats, fbank.features(self.wave, have, ready)])
        self._run()

    def finish(self, tail_seconds: float = 0.66) -> None:
        self.finished = True
        self.accept(np.zeros(int(tail_seconds * fbank.SR), np.float32))

    def _run(self) -> None:
        while len(self.feats) - self.done_frames >= self.m.T:
            chunk = self.feats[self.done_frames: self.done_frames + self.m.T]
            enc, self.state = self.m.encode(chunk, self.state)
            self.search.step(enc, self.frames_decoded)
            self.frames_decoded += len(enc)
            self.done_frames += self.m.shift

    @property
    def text(self) -> str:
        return self.search.text()
