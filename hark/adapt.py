"""Learning a speaker's words from their corrections, on the device, with no retraining.

When the user fixes a caption, Hark lines up what it heard against what was said and keeps two things:
  * the words it missed, which become a phrase list that biases the beam search toward them (see search.Bias);
  * the exact mistakes it made ("fill up" -> "Philip"), which become rewrite rules, used only once the same
    mistake has been corrected more often than the same words were confirmed right.
Everything is plain counts, so it fits in a few kilobytes and can be inspected, exported or wiped.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# How common each everyday English word is (wordfreq Zipf scale: 6 = once per thousand words, 4 = once per hundred
# thousand). Listening harder for a word everyone says all the time makes Hark insert it where it was not said, so
# only words rarer than `max_zipf` go on the phrase list: names, places, jargon.
COMMON: dict[str, float] = json.loads((Path(__file__).parent / "common_words.json").read_text())

STOP = set("A AN THE OF TO AND IN ON AT IS IT I YOU HE SHE WE THEY BE WAS ARE OR FOR WITH AS BY THAT THIS".split())


def words(text: str) -> list[str]:
    return re.sub(r"[^A-Z' ]+", " ", text.upper().replace("-", " ")).split()


def align(hyp: list[str], ref: list[str]) -> list[tuple[str, int, int, int, int]]:
    """Word alignment as edit operations: ('ok'|'sub'|'del'|'ins', hyp start, hyp end, ref start, ref end).

    Neighbouring errors are merged into one span, so 'fill up' -> 'Philip' is a single substitution."""
    n, m = len(hyp), len(ref)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]))
    ops = []
    i, j = n, m
    while i or j:
        if i and j and hyp[i - 1] == ref[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(("ok", i - 1, i, j - 1, j)); i, j = i - 1, j - 1
        elif i and j and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, i, j - 1, j)); i, j = i - 1, j - 1
        elif j and d[i][j] == d[i][j - 1] + 1:
            ops.append(("del", i, i, j - 1, j)); j -= 1
        else:
            ops.append(("ins", i - 1, i, j, j)); i -= 1
    ops.reverse()
    merged: list[list] = []
    for op in ops:
        if op[0] != "ok" and merged and merged[-1][0] != "ok":
            merged[-1] = ["err", merged[-1][1], op[2], merged[-1][3], op[4]]
        else:
            merged.append(list(op) if op[0] == "ok" else ["err", *op[1:]])
    return [tuple(x) for x in merged]


def wer_counts(hyp: list[str], ref: list[str]) -> tuple[int, int]:
    """(word errors, reference words)."""
    n, m = len(hyp), len(ref)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (hyp[i - 1] != ref[j - 1]))
        prev = cur
    return prev[m], m


class Memory:
    def __init__(self, min_rule_count: int = 2, max_span: int = 3, min_phrase_count: int = 1, max_zipf: float = 99.0):
        self.missed: Counter[str] = Counter()      # words said but not captioned
        self.heard: Counter[str] = Counter()       # words said and captioned right
        self.rules: Counter[tuple[str, str]] = Counter()   # (heard span, said span)
        self.right: Counter[str] = Counter()       # spans that were captioned right when seen
        self.min_rule_count, self.max_span, self.min_phrase_count = min_rule_count, max_span, min_phrase_count
        self.max_zipf = max_zipf
        self.corrections = 0

    def learn(self, heard: str, said: str) -> None:
        h, r = words(heard), words(said)
        self.corrections += 1
        for op, hs, he, rs, re_ in align(h, r):
            if op == "ok":
                self.heard[r[rs]] += 1
                for n in range(1, self.max_span + 1):  # every correct span, so a rule can be weighed against it
                    if hs - n + 1 >= 0:
                        self.right[" ".join(h[hs - n + 1: he])] += 1
            else:
                for w in r[rs:re_]:
                    if len(w) > 1 and w not in STOP:
                        self.missed[w] += 1
                if 0 < he - hs <= self.max_span and 0 < re_ - rs <= self.max_span:
                    self.rules[(" ".join(h[hs:he]), " ".join(r[rs:re_]))] += 1

    def phrases(self, limit: int = 300) -> list[str]:
        """Words to bias toward: the ones this speaker says that the model keeps missing."""
        return [w for w, n in self.missed.most_common() if n >= self.min_phrase_count and COMMON.get(w, 0.0) < self.max_zipf][:limit]

    def pending(self) -> list[str]:
        """Words missed fewer times than needed to be listed."""
        return [w for w, n in self.missed.most_common() if n < self.min_phrase_count and COMMON.get(w, 0.0) < self.max_zipf]

    def active_rules(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for (src, dst), n in self.rules.most_common():
            if n >= self.min_rule_count and n > self.right[src] and src not in out:
                out[src] = dst
        return out

    def rewrite(self, text: str) -> str:
        rules = self.active_rules()
        if not rules:
            return " ".join(words(text))
        w, out, i = words(text), [], 0
        while i < len(w):
            for n in range(self.max_span, 0, -1):
                span = " ".join(w[i: i + n])
                if len(w[i: i + n]) == n and span in rules:
                    out.append(rules[span]); i += n
                    break
            else:
                out.append(w[i]); i += 1
        return " ".join(out)

    def to_json(self) -> dict:
        return {"missed": dict(self.missed), "rules": [[s, d, n] for (s, d), n in self.rules.items()], "right": dict(self.right),
                "corrections": self.corrections}
