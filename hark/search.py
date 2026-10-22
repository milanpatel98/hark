"""Turning encoder frames into words."""

from __future__ import annotations

import numpy as np


class Greedy:
    """Pick the most likely token at every frame; the reference the beam search is checked against."""

    def __init__(self, model):
        self.m = model
        self.ids: list[int] = []
        self.times: list[int] = []
        self.ctx = [-1] * (model.context - 1) + [model.blank]  # -1 pads the start, as in training
        self.dec = model.decode(np.array([self.ctx]))

    def step(self, enc: np.ndarray, t0: int) -> None:
        for t, frame in enumerate(enc):
            y = int(np.argmax(self.m.joint(frame[None], self.dec)[0]))
            if y not in (self.m.blank, self.m.unk):
                self.ids.append(y)
                self.times.append(t0 + t)
                self.ctx = self.ctx[1:] + [y]
                self.dec = self.m.decode(np.array([self.ctx]))

    def text(self) -> str:
        return self.m.text(self.ids)


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


class Bias:
    """A phrase list compiled for decoding: a trie over word pieces with fallback links (Aho-Corasick).

    While the beam follows a listed phrase, each matching piece earns a bonus. A partial match that goes nowhere
    gives its bonus back, so the list nudges the model toward words it knows the user says, without forcing them.
    """

    def __init__(self, phrases: list[list[int]], bonus: float):
        self.bonus = bonus
        self.child: list[dict[int, int]] = [{}]
        self.depth_score: list[float] = [0.0]
        self.end: list[bool] = [False]
        self.fail: list[int] = [0]
        for p in phrases:
            n = 0
            for t in p:
                if t not in self.child[n]:
                    self.child.append({}); self.depth_score.append(self.depth_score[n] + bonus); self.end.append(False); self.fail.append(0)
                    self.child[n][t] = len(self.child) - 1
                n = self.child[n][t]
            self.end[n] = True
        queue = list(self.child[0].values())
        while queue:  # breadth first: fallback of a node = longest proper suffix that is also a prefix
            n = queue.pop(0)
            for t, c in self.child[n].items():
                f = self.fail[n]
                while f and t not in self.child[f]:
                    f = self.fail[f]
                self.fail[c] = self.child[f][t] if t in self.child[f] and self.child[f][t] != c else 0
                queue.append(c)

    def step(self, state: int, token: int) -> tuple[float, int]:
        """(score change, new state) after emitting token from state."""
        n = state
        while n and token not in self.child[n]:
            n = self.fail[n]
        new = self.child[n].get(token, 0)
        score = self.depth_score[new] - self.depth_score[state]
        if self.end[new]:
            return score, -1 - new  # a finished phrase keeps its bonus: negative state = committed at root
        return score, new

    def finish(self, state: int) -> float:
        return -self.depth_score[state] if state > 0 else 0.0


def _norm(state: int) -> int:
    return 0 if state < 0 else state


class Beam:
    """Modified beam search: at most one word piece per frame, the best few hypotheses kept, equal ones merged."""

    def __init__(self, model, beam: int = 4, bias: Bias | None = None):
        self.m, self.k, self.bias = model, beam, bias
        start = tuple([-1] * (model.context - 1) + [model.blank])
        # hypothesis: tokens -> [log prob, bias score, bias state, times]
        self.hyps: dict[tuple, list] = {start: [0.0, 0.0, 0, []]}
        self._dec_cache: dict[tuple, np.ndarray] = {}

    def _dec(self, keys: list[tuple]) -> np.ndarray:
        missing = [k for k in keys if k[-self.m.context:] not in self._dec_cache]
        if missing:
            ctx = sorted({k[-self.m.context:] for k in missing})
            out = self.m.decode(np.array(ctx))
            for c, o in zip(ctx, out):
                self._dec_cache[c] = o
        return np.stack([self._dec_cache[k[-self.m.context:]] for k in keys])

    def step(self, enc: np.ndarray, t0: int) -> None:
        for t, frame in enumerate(enc):
            keys = list(self.hyps)
            dec = self._dec(keys)
            lp = log_softmax(self.m.joint(np.repeat(frame[None], len(keys), 0), dec).astype(np.float64))
            base = np.array([self.hyps[k][0] + self.hyps[k][1] for k in keys])[:, None]
            V = lp.shape[1]
            flat = (lp + base).ravel()
            top = np.argpartition(-flat, min(self.k, flat.size - 1))[: self.k]
            new: dict[tuple, list] = {}
            for idx in top:
                h, tok = divmod(int(idx), V)
                key = keys[h]
                lprob, bscore, bstate, times = self.hyps[key]
                lprob = lprob + float(lp[h, tok])
                if tok in (self.m.blank, self.m.unk):
                    nk, nb, ns, nt = key, bscore, bstate, times
                else:
                    nk, nt = key + (tok,), times + [t0 + t]
                    if self.bias:
                        d, ns = self.bias.step(_norm(bstate), tok)
                        nb = bscore + d
                    else:
                        nb, ns = bscore, bstate
                if nk in new:  # same words reached two ways: add the probabilities
                    o = new[nk]
                    o[0] = float(np.logaddexp(o[0], lprob))
                else:
                    new[nk] = [lprob, nb, ns, nt]
            self.hyps = new

    def best(self) -> tuple[tuple, list]:
        def total(item):
            lprob, bscore, bstate, _ = item[1]
            return lprob + bscore + (self.bias.finish(_norm(bstate)) if self.bias else 0.0)
        return max(self.hyps.items(), key=total)

    @property
    def ids(self) -> list[int]:
        return [t for t in self.best()[0] if t >= 0 and t != self.m.blank]

    @property
    def times(self) -> list[int]:
        return self.best()[1][3]

    def text(self) -> str:
        return self.m.text(self.ids)
