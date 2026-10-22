"""The streaming speech model: a Zipformer transducer (Apache-2.0, k2-fsa/icefall, trained on LibriSpeech and
GigaSpeech, 2023-06-21; the int8 "mobile" export from sherpa-onnx).

The encoder hears audio in chunks of 39 feature frames (0.39 s), stepping 32 frames (0.32 s) at a time, and carries
its memory of everything before in a set of cached tensors. The decoder and joiner turn that into word pieces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "sherpa-onnx-streaming-zipformer-en-2023-06-21-mobile"


def model_file(folder: Path, part: str, int8: bool = True) -> Path:
    """encoder/decoder/joiner file: the int8 export where there is one."""
    q, f = folder / f"{part}-epoch-99-avg-1.int8.onnx", folder / f"{part}-epoch-99-avg-1.onnx"
    return q if int8 and q.exists() else f


class Model:
    def __init__(self, folder: Path = MODEL, int8: bool = True, threads: int = 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        load = lambda n: ort.InferenceSession(str(model_file(folder, n, int8)), opts, providers=["CPUExecutionProvider"])
        self.enc, self.dec, self.join = load("encoder"), load("decoder"), load("joiner")
        meta = self.enc.get_modelmeta().custom_metadata_map
        ints = lambda k: [int(v) for v in meta[k].split(",")]
        self.T, self.shift = int(meta["T"]), int(meta["decode_chunk_len"])
        self.layers, self.enc_dims, self.att_dims = ints("num_encoder_layers"), ints("encoder_dims"), ints("attention_dims")
        self.left, self.kernels = ints("left_context_len"), ints("cnn_module_kernels")
        self.context = int(self.dec.get_modelmeta().custom_metadata_map["context_size"])
        self.tokens = [l.rsplit(" ", 1)[0] for l in (folder / "tokens.txt").read_text(encoding="utf-8").splitlines()]
        self.blank, self.unk = 0, self.tokens.index("<unk>")
        self.state_names = [i.name for i in self.enc.get_inputs()][1:]

    def init_state(self) -> list[np.ndarray]:
        s: dict[str, np.ndarray] = {}
        for i, (n, d, a, l, k) in enumerate(zip(self.layers, self.enc_dims, self.att_dims, self.left, self.kernels)):
            s[f"cached_len_{i}"] = np.zeros((n, 1), np.int64)
            s[f"cached_avg_{i}"] = np.zeros((n, 1, d), np.float32)
            s[f"cached_key_{i}"] = np.zeros((n, l, 1, a), np.float32)
            s[f"cached_val_{i}"] = np.zeros((n, l, 1, a // 2), np.float32)
            s[f"cached_val2_{i}"] = np.zeros((n, l, 1, a // 2), np.float32)
            s[f"cached_conv1_{i}"] = np.zeros((n, 1, d, k - 1), np.float32)
            s[f"cached_conv2_{i}"] = np.zeros((n, 1, d, k - 1), np.float32)
        return [s[n] for n in self.state_names]

    def encode(self, chunk: np.ndarray, state: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """chunk [T, 80] -> encoder frames [t, 512] and the new state."""
        out = self.enc.run(None, {"x": chunk[None].astype(np.float32), **dict(zip(self.state_names, state))})
        return out[0][0], out[1:]

    def decode(self, contexts: np.ndarray) -> np.ndarray:
        """[N, context] token ids -> [N, 512]."""
        return self.dec.run(None, {"y": contexts.astype(np.int64)})[0]

    def joint(self, enc: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """[N, 512] x [N, 512] -> [N, vocab] logits."""
        return self.join.run(None, {"encoder_out": enc.astype(np.float32), "decoder_out": dec.astype(np.float32)})[0]

    def text(self, ids: list[int]) -> str:
        return "".join(self.tokens[i] for i in ids).replace("▁", " ").strip()
