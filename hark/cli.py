"""hark: command line.

  hark get-model                 download the speech model (Apache-2.0) into models/
  hark caption <file>            caption an audio file, streaming, as the app does
  hark tune                      choose settings on the two development speakers
  hark bench [--bonus B --min-rule R]
                                 the accent experiment on every speaker in speech/
  hark export                    write the model and expected outputs for the website
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
import urllib.request

from .model import MODEL

MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21-mobile.tar.bz2"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="hark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("get-model")
    c = sub.add_parser("caption"); c.add_argument("file"); c.add_argument("--greedy", action="store_true")
    sub.add_parser("tune")
    b = sub.add_parser("bench"); b.add_argument("--bonus", type=float); b.add_argument("--min-rule", type=int); b.add_argument("--workers", type=int, default=2)
    sub.add_parser("export")
    a = ap.parse_args(argv)

    if a.cmd == "get-model":
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        path = MODEL.parent / "model.tar.bz2"
        urllib.request.urlretrieve(MODEL_URL, path)
        with tarfile.open(path) as t:
            t.extractall(MODEL.parent, filter="data")
        path.unlink()
        print(f"model -> {MODEL}")
    elif a.cmd == "caption":
        from .corpus import load_wave
        from .model import Model
        from .search import Beam, Greedy
        from .stream import Stream
        m = Model(threads=2)
        wave = load_wave(a.file)
        s = Stream(m, Greedy(m) if a.greedy else Beam(m))
        t = time.time()
        for i in range(0, len(wave), 1600):
            s.accept(wave[i:i + 1600])
        s.finish()
        print(s.text.lower())
        print(f"[{len(wave) / 16000:.1f} s of audio in {time.time() - t:.1f} s]")
    elif a.cmd == "tune":
        from .bench import tune
        print(tune())
    elif a.cmd == "bench":
        from .bench import OUT, run
        cfg = json.loads((OUT / "tuning.json").read_text())["chosen"] if a.bonus is None else {"bonus": a.bonus, "min_rule": a.min_rule or 2, "min_phrase": 2}
        run(cfg["bonus"], cfg["min_rule"], cfg.get("min_phrase", 2), a.workers, max_zipf=cfg.get("max_zipf", 99.0))
        print(f"-> {OUT / 'accents.json'}")
    elif a.cmd == "export":
        from .export import export
        export()


if __name__ == "__main__":
    main()
