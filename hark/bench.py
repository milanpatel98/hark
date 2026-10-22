"""The experiment: how much does Hark improve for a speaker after they correct some of its captions?

For every speaker, the first 200 sentences are the ones they 'correct' (we use the true transcript as the
correction), in order, with Hark learning as it goes. The last 300 sentences are never corrected and are only
used to measure. Word error rate on those 300 is reported after 0, 25, 50, 100 and 200 corrections.

Settings (bias strength, how often a mistake must repeat before a rewrite rule is used) are chosen on two
development speakers only, and every other speaker is reported as a held-out test.
"""

from __future__ import annotations

import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import corpus, fbank
from .adapt import Memory, align, wer_counts, words
from .model import Model
from .search import Beam, Bias
from .tokens import Pieces

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "speech" / "cache"
OUT = ROOT / "results"
DEV = ["ABA", "PNV"]
CHECKPOINTS = [0, 25, 50, 100, 200]
N_ADAPT, N_TEST = 200, 300
TAIL = 0.66
# ARCTIC recordings start speaking at the very first sample; a live microphone never does. The streaming model
# drops a word it hears before it has any context, so each recording gets the same 0.66 s of silence in front
# that it gets behind (the tail every stream is flushed with).
LEAD = 0.66

_model: Model | None = None


def model() -> Model:
    global _model
    if _model is None:
        _model = Model(threads=1)
    return _model


def encode(wave: np.ndarray) -> np.ndarray:
    """All encoder frames for an utterance, exactly as the live stream would produce them."""
    m = model()
    silence = lambda sec: np.zeros(int(sec * fbank.SR), np.float32)
    feats = fbank.features(np.concatenate([silence(LEAD), wave, silence(TAIL)]))
    state, out, p = m.init_state(), [], 0
    while len(feats) - p >= m.T:
        enc, state = m.encode(feats[p: p + m.T], state)
        out.append(enc)
        p += m.shift
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def cached(u: corpus.Utt) -> np.ndarray:
    f = CACHE / u.speaker / f"{u.id}.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    enc = encode(corpus.load_wave(u.path))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, enc.astype(np.float16))
    return enc.astype(np.float16).astype(np.float32)


def decode(enc: np.ndarray, bias: Bias | None) -> str:
    b = Beam(model(), bias=bias)
    b.step(enc, 0)
    return b.text()


def split(speaker: str):
    utts = corpus.speaker_utts(speaker)
    return utts[:N_ADAPT], utts[-N_TEST:]


def make_bias(mem: Memory, bonus: float) -> Bias | None:
    pieces = Pieces(model().tokens)
    phrases = [p for p in (pieces.phrase(w) for w in mem.phrases()) if p]
    return Bias(phrases, bonus) if phrases and bonus > 0 else None


def hyps_for(test_enc, mem: Memory | None, bonus: float, use_rules: bool) -> list[str]:
    bias = make_bias(mem, bonus) if mem else None
    out = []
    for enc in test_enc:
        h = decode(enc, bias)
        out.append(mem.rewrite(h) if mem and use_rules else " ".join(words(h)))
    return out


def word_hits(hyp: str, ref: str) -> list[tuple[str, bool]]:
    """Every reference word, and whether the caption got it right."""
    out = []
    for op, hs, he, rs, re_ in align(words(hyp), words(ref)):
        out += [(w, op == "ok") for w in words(ref)[rs:re_]]
    return out


def measure(test, base_hyps: list[str], hyps: list[str], learned: set[str]) -> dict:
    """Overall word errors, and accuracy on the words the speaker had corrected before (same occurrences, before vs after)."""
    e = n = 0
    seen = before = after = 0
    for u, b, h in zip(test, base_hyps, hyps):
        de, dn = wer_counts(words(h), words(u.text))
        e, n = e + de, n + dn
        for (w, ok_b), (_, ok_a) in zip(word_hits(b, u.text), word_hits(h, u.text)):
            if w in learned:
                seen += 1; before += ok_b; after += ok_a
    return {"errors": e, "words": n, "learned_seen": seen, "learned_right_before": before, "learned_right_after": after}


def run_speaker(args) -> dict:
    speaker, bonus, min_rule, methods, min_phrase, max_zipf = (*args, 1, 99.0)[:6] if len(args) == 4 else (*args, 99.0)[:6] if len(args) == 5 else args
    adapt, test = split(speaker)
    t0 = time.time()
    test_enc = [cached(u) for u in test]
    adapt_enc = [cached(u) for u in adapt]
    base = hyps_for(test_enc, None, 0, False)
    res = {"speaker": speaker, "bonus": bonus, "min_rule": min_rule, "min_phrase": min_phrase, "max_zipf": max_zipf, "curve": {}}
    for meth in methods:
        bonus_m = 0 if meth == "rules" else bonus
        rules_m = meth != "bias"
        mem = Memory(min_rule_count=min_rule, min_phrase_count=min_phrase, max_zipf=max_zipf)
        curve = {0: measure(test, base, base, set())}
        for i, (u, enc) in enumerate(zip(adapt, adapt_enc), 1):
            shown = decode(enc, make_bias(mem, bonus_m))   # what the user saw, before their correction
            mem.learn(shown, u.text)
            if i in CHECKPOINTS:
                curve[i] = measure(test, base, hyps_for(test_enc, mem, bonus_m, rules_m), set(mem.phrases()))
        res["curve"][meth] = curve
        res.setdefault("memory", {})[meth] = {"phrases": len(mem.phrases()), "rules": len(mem.active_rules()),
                                             "examples": list(mem.active_rules().items())[:8], "top_words": mem.phrases(12)}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


# second round (after the first test run showed common words being inserted): only uncommon words are listed
GRID = [(b, p, z) for b in (1.5, 2.0, 3.0) for p in (1, 2) for z in (4.0, 5.0)]


def tune(workers: int = 2) -> dict:
    """Pick (bias strength, how often a word must be missed before it is listed) on the development speakers only.

    Rule: the setting with the most learned-word gain at 200 fixes whose overall error count is no higher than the
    unadapted baseline; Hark should never make captions worse for someone who corrects it."""
    jobs = [(s, b, 2, ["both"], p, z) for s in DEV for b, p, z in GRID]
    with Pool(workers) as pool:
        rows = pool.map(run_speaker, jobs)
    table = []
    for b, p, z in GRID:
        rs = [r for r in rows if r["bonus"] == b and r["min_phrase"] == p and r["max_zipf"] == z]
        end = [r["curve"]["both"][N_ADAPT] for r in rs]
        start = [r["curve"]["both"][0] for r in rs]
        table.append({"bonus": b, "min_phrase": p, "max_zipf": z, "errors_before": sum(x["errors"] for x in start), "errors_after": sum(x["errors"] for x in end),
                      "learned_seen": sum(x["learned_seen"] for x in end), "learned_gain": sum(x["learned_right_after"] - x["learned_right_before"] for x in end)})
    ok = [t for t in table if t["errors_after"] <= t["errors_before"]] or table
    best = max(ok, key=lambda t: (t["learned_gain"], -t["errors_after"]))
    OUT.mkdir(exist_ok=True)
    (OUT / "tuning.json").write_text(json.dumps({"grid": table, "chosen": {"bonus": best["bonus"], "min_phrase": best["min_phrase"], "max_zipf": best["max_zipf"], "min_rule": 2},
                                                  "dev_speakers": DEV}, indent=1))
    return best


def run(bonus: float, min_rule: int, min_phrase: int = 1, workers: int = 2, speakers: list[str] | None = None, max_zipf: float = 99.0) -> dict:
    speakers = speakers or corpus.available()
    with Pool(workers) as p:
        rows = p.map(run_speaker, [(s, bonus, min_rule, ["bias", "rules", "both"], min_phrase, max_zipf) for s in speakers])
    for r in rows:
        info = corpus.L2.get(r["speaker"]) or corpus.NATIVE[r["speaker"]]
        r["accent"], r["sex"], r["split"] = info[0], info[1], "dev" if r["speaker"] in DEV else "test"
    OUT.mkdir(exist_ok=True)
    out = {"settings": {"bonus": bonus, "min_rule": min_rule, "min_phrase": min_phrase, "max_zipf": max_zipf, "adapt": N_ADAPT, "test": N_TEST,
                        "checkpoints": CHECKPOINTS, "lead_seconds": LEAD},
           "speakers": rows}
    (OUT / "accents.json").write_text(json.dumps(out, indent=1))
    return out


def _cache_speaker(speaker: str) -> tuple[str, int]:
    adapt, test = split(speaker)
    for u in adapt + test:
        cached(u)
    return speaker, len(adapt) + len(test)


def cache_all(speakers: list[str], workers: int = 2) -> None:
    """Run the encoder over every sentence once; all later experiments reuse its output."""
    with Pool(workers) as p:
        for spk, n in p.imap_unordered(_cache_speaker, speakers):
            print(f"cached {spk}: {n} sentences", flush=True)
