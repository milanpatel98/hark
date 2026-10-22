"""Reading the speech recordings: L2-ARCTIC (non-native speakers) and CMU ARCTIC (native speakers).

Both read the same 1,132 ARCTIC sentences, so the only thing that differs between speakers is how they talk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "speech"  # not committed: see README for where the recordings come from

# L2-ARCTIC speaker -> (first language, sex)
L2 = {"ABA": ("Arabic", "M"), "SKA": ("Arabic", "F"), "BWC": ("Mandarin", "M"), "LXC": ("Mandarin", "F"),
      "ASI": ("Hindi", "M"), "SVBI": ("Hindi", "F"), "HKK": ("Korean", "M"), "HJK": ("Korean", "F"),
      "ERMS": ("Spanish", "M"), "NJS": ("Spanish", "F"), "TLV": ("Vietnamese", "M"), "PNV": ("Vietnamese", "F")}
NATIVE = {"bdl": ("English (US)", "M"), "slt": ("English (US)", "F")}


@dataclass
class Utt:
    speaker: str
    id: str
    path: Path
    text: str


def load_wave(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    return a


def speaker_utts(speaker: str) -> list[Utt]:
    if speaker in L2:
        base = next(p for p in [DATA / speaker / speaker, DATA / speaker] if (p / "wav").exists())
        out = [Utt(speaker, p.stem, p, (base / "transcript" / f"{p.stem}.txt").read_text(encoding="utf-8").strip())
               for p in sorted((base / "wav").glob("*.wav")) if (base / "transcript" / f"{p.stem}.txt").exists()]
    else:
        base = DATA / f"cmu_us_{speaker}_arctic"
        texts = dict(re.findall(r'\(\s*(\S+)\s+"(.*)"\s*\)', (base / "etc" / "txt.done.data").read_text(encoding="utf-8")))
        out = [Utt(speaker, i, base / "wav" / f"{i}.wav", t) for i, t in sorted(texts.items()) if (base / "wav" / f"{i}.wav").exists()]
    return out


def available() -> list[str]:
    return [s for s in [*L2, *NATIVE] if (DATA / s).exists() or (DATA / f"cmu_us_{s}_arctic").exists()]
