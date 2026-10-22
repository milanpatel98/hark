"""Write what the website needs: the model files, their settings, and expected outputs for the JavaScript port."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import fbank
from .adapt import Memory
from .corpus import load_wave
from .model import MODEL, Model, model_file
from .search import Beam, Bias, Greedy
from .stream import Stream
from .tokens import Pieces

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SAMPLES = ROOT / "samples"  # short public-domain LibriSpeech clips shipped with the model (Apache-2.0)

BIAS_WORDS = ["PLAYING", "BROTHELS", "AFTER EARLY", "HESTER PRYNNE", "WHITTEMORE"]


def export() -> None:
    m = Model()
    out = WEB / "public" / "model"
    out.mkdir(parents=True, exist_ok=True)
    for n in ("encoder", "decoder", "joiner"):
        shutil.copy(model_file(MODEL, n), out / f"{n}.onnx")
    cfg = {"T": m.T, "shift": m.shift, "layers": m.layers, "encDims": m.enc_dims, "attDims": m.att_dims, "left": m.left,
           "kernels": m.kernels, "context": m.context, "tokens": m.tokens}
    (out / "config.json").write_text(json.dumps(cfg))
    data = WEB / "public" / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "hark" / "common_words.json", data / "common.json")

    pieces = Pieces(m.tokens)
    cases = []
    for wav in sorted(SAMPLES.glob("*.wav")):
        a = load_wave(wav)
        row = {"file": wav.name, "fbank": fbank.features(a, 0, 5).round(4).tolist()}
        for name, make in [("greedy", lambda: Greedy(m)), ("beam", lambda: Beam(m)),
                           ("biased", lambda: Beam(m, bias=Bias([pieces.phrase(w) for w in BIAS_WORDS], 2.0)))]:
            s = Stream(m, make())
            for i in range(0, len(a), 1600):
                s.accept(a[i:i + 1600])
            s.finish()
            row[name] = {"text": s.text, "ids": s.search.ids}
        cases.append(row)

    mem = Memory()
    log = []
    for heard, said in [("I WILL FILL UP THE CAR", "I WILL PHILIP THE CAR"), ("CALL FILL UP NOW", "CALL PHILIP NOW"),
                        ("FILL UP THE FORMS", "FILL UP THE FORMS"), ("ASK FILL UP", "ASK PHILIP"), ("THE CAT", "THE CAT SAT")]:
        mem.learn(heard, said)
        log.append({"phrases": mem.phrases(), "rules": sorted(mem.active_rules().items()), "rewrite": mem.rewrite("TELL FILL UP HI")})
    rare = Memory(min_phrase_count=2, max_zipf=5.0)
    for heard, said in [("ASK GRACE ONE WERE", "ASK GREGSON WAS"), ("GRACE ONE WERE HERE", "GREGSON WAS HERE")]:
        rare.learn(heard, said)
    log.append({"phrases": rare.phrases(), "rules": sorted(rare.active_rules().items()), "rewrite": rare.rewrite("GRACE ONE WERE")})
    spell = {w: pieces.phrase(w) for w in ["PHILIP", "BROTHELS", "STELLA", "HESTER PRYNNE", "QWXZ"]}
    (WEB / "test").mkdir(exist_ok=True)
    (WEB / "test" / "expected.json").write_text(json.dumps({"bias": BIAS_WORDS, "cases": cases, "memory": log, "spell": spell}))
    export_results()
    print(f"model and {len(cases)} expected cases -> {WEB}")


def _totals(path: Path, method: str = "both") -> dict:
    data = json.loads(path.read_text())
    last = str(max(int(k) for k in data["speakers"][0]["curve"][method]))
    t = {"errors_before": 0, "errors_after": 0, "words": 0, "learned_seen": 0, "learned_wrong_before": 0, "learned_wrong_after": 0}
    for s in data["speakers"]:
        if s["split"] != "test":
            continue
        b, a = s["curve"][method]["0"], s["curve"][method][last]
        t["errors_before"] += b["errors"]; t["errors_after"] += a["errors"]; t["words"] += b["words"]
        t["learned_seen"] += a["learned_seen"]
        t["learned_wrong_before"] += a["learned_seen"] - a["learned_right_before"]
        t["learned_wrong_after"] += a["learned_seen"] - a["learned_right_after"]
    t["fixes"] = int(last)
    t["speakers"] = sum(1 for s in data["speakers"] if s["split"] == "test")
    return t


def export_results() -> dict | None:
    """Summarise results/accents.json (and the first run, kept as it happened) for the Results screen."""
    src = ROOT / "results" / "accents.json"
    if not src.exists():
        return None
    data = json.loads(src.read_text())
    last = str(max(int(k) for k in data["speakers"][0]["curve"]["both"]))
    rows = []
    for s in data["speakers"]:
        b, a = s["curve"]["both"]["0"], s["curve"]["both"][last]
        rows.append({"speaker": s["speaker"], "accent": s["accent"], "sex": s["sex"], "split": s["split"],
                     "before": b["errors"] / b["words"], "after": a["errors"] / a["words"]})
    rows.sort(key=lambda r: (r["split"] != "test", -r["before"]))
    out = {"summary": _totals(src), "speakers": rows, "settings": data["settings"]}
    first = ROOT / "results" / "accents-first-run.json"
    if first.exists():
        out["first_run"] = _totals(first)
    (WEB / "public" / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "public" / "data" / "results.json").write_text(json.dumps(out))
    return out
