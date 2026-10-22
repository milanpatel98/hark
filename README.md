# Hark

**Live captions that run on your device and learn your voice from your corrections.**

Small on-device speech models make far more mistakes on accented English than on the voices they were trained on. Hark captions speech live in the browser, with nothing sent to a server. When you correct a caption, it learns the words you say that it keeps getting wrong, and listens harder for them next time. Nothing is retrained; it learns from plain counts that stay in your browser.

**[Try it](https://milanpatel98.github.io/hark/)**: works on a phone or a computer. "Try a sample" plays a real accented recording if you don't want to use the microphone.

---

## Results

__RESULTS__

## How it works

```
 microphone ─► 16 kHz audio ─► log-mel features ─► streaming encoder (0.32 s steps, keeps its memory)
                                                          │
             your corrections ─► words you say that it    │
                                 missed  ───────────────► beam search, nudged toward those words
                                                          │
                                                          ▼
                                                  live captions, line by line
```

**The speech model.** A streaming Zipformer transducer from k2-fsa/icefall, trained on LibriSpeech and GigaSpeech (Apache-2.0), in its int8 export (about 110 MB, downloaded once). A smaller 20M-parameter model was tried first. It got more than half the words wrong on accented speech, so it was dropped.

**My own engine, not a library.** Everything between the microphone and the words is written here, in Python and again in JavaScript for the browser:
- Kaldi-style filterbank features, with my own FFT.
- The streaming loop that feeds the encoder 0.39 s chunks and carries its cached state.
- Greedy and beam search over the transducer.
- The phrase biasing.

`tests/` checks the Python engine against sherpa-onnx (identical output) and against kaldi-native-fbank. `web/test/parity.mjs` checks the browser engine against Python, piece by piece. Greedy and beam search can differ by one word piece where two choices tie to a few hundredths of a point, because the browser's int8 kernels round differently. Anything more fails the test.

**Learning from corrections** (`hark/adapt.py`). A correction is lined up word by word against what Hark showed:
- **Missed words** become a phrase list. During beam search, a hypothesis that spells one of those words earns a small bonus per word piece. A trie with fallback links (Aho-Corasick) means a partial match that goes nowhere gives its bonus back, so the list nudges and never forces.
- **Exact mistakes** ("flip" → "Philip") become rewrite rules. A rule is only used once the same mistake has been corrected at least twice, and more often than the same words were confirmed right.

**Privacy.** Audio is processed in a Web Worker on the device. What Hark learns is a few kilobytes in `localStorage`, shown on the Your voice screen, with a button to erase it.

## The experiment

- **Speakers:** non-native speakers from [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) (Arabic, Mandarin, Hindi, Korean, Spanish and Vietnamese), each reading the same ARCTIC sentences.
- **Corrections:** for each speaker, the first 200 sentences are "corrected" in order, using the true transcript as the correction. Hark learns as it goes, from what it actually showed at the time.
- **Measurement:** the last 300 sentences are never corrected, and are only used to measure:
  - word error rate over all words;
  - accuracy on the exact occurrences of words the speaker had corrected, before learning and after.
- **Settings:** the bias strength, and how often a word must be missed before it is listed, are chosen on two development speakers (ABA, PNV) with one rule: the most gain on learned words that does not raise overall errors. Every other speaker is a held-out test.
- **Audio:** each recording gets 0.66 s of silence in front, because the recordings start speaking on the first sample and a live microphone never does. On the development speakers this lowered errors from 17.8% to 13.1% (ABA) and 14.8% to 14.5% (PNV).

## Run it

Python 3.10+, Node 20+.

```bash
git clone https://github.com/milanpatel98/hark.git && cd hark
pip install -e ".[dev]"
hark get-model                 # the speech model, from the sherpa-onnx releases
pytest -q

hark caption samples/l2arctic-ska-a0010.wav

hark export                    # model and expected answers for the browser engine
cd web && npm install && npm test && npm run dev
```

To rerun the experiment, download the L2-ARCTIC speakers listed in `hark/corpus.py` into `speech/`, then run `hark tune` and `hark bench`.

## Limitations

- **The gain is narrow.** Learning helps on the words you corrected and on little else. People repeat their own names and terms, so it matters in use, but read-aloud test sentences mostly don't repeat them.
- **No speaker labels.** Hark doesn't tell speakers apart. A line break means a pause, not a new person.
- **English only, one model.** The model is trained on read audiobooks and podcasts, so noisy rooms and fast conversation are harder.
- **First load is about 110 MB.** After that the browser caches it.
- **Speed varies.** Below real time on a laptop in single-threaded WebAssembly; slower phones may lag.

## Licenses

- **Code:** MIT.
- **Speech model:** Apache-2.0 (k2-fsa).
- **L2-ARCTIC:** CC BY-NC 4.0. The sample clip is used for non-commercial demonstration; see `samples/README.md`.
- **LibriSpeech:** CC BY 4.0.
