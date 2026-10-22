"""Spelling a word in the model's word pieces, so a phrase list can be matched during decoding."""

from __future__ import annotations

from functools import lru_cache


class Pieces:
    def __init__(self, tokens: list[str]):
        self.index = {t: i for i, t in enumerate(tokens)}

    def word(self, w: str) -> list[int] | None:
        """Fewest-piece spelling of an upper-case word, starting with a word-initial piece ('▁...')."""
        w = w.upper()

        @lru_cache(None)
        def best(i: int, first: bool) -> tuple[int, ...] | None:
            if i == len(w):
                return ()
            out = None
            for j in range(len(w), i, -1):
                piece = ("▁" if first else "") + w[i:j]
                if piece in self.index:
                    rest = best(j, False)
                    if rest is not None and (out is None or 1 + len(rest) < len(out)):
                        out = (self.index[piece],) + rest
            if first and out is None and "▁" in self.index:  # a lone word-start marker, then the pieces
                rest = best(i, False)
                if rest is not None:
                    out = (self.index["▁"],) + rest
            return out

        r = best(0, True)
        return list(r) if r else None

    def phrase(self, text: str) -> list[int] | None:
        out: list[int] = []
        for w in text.split():
            p = self.word(w)
            if p is None:
                return None
            out += p
        return out
