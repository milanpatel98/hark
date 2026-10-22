import numpy as np
import pytest

from hark import fbank
from hark.adapt import Memory, align, wer_counts, words
from hark.corpus import load_wave
from hark.export import SAMPLES
from hark.model import MODEL, Model, model_file
from hark.search import Beam, Bias, Greedy
from hark.stream import Stream
from hark.tokens import Pieces

WAV = SAMPLES / "0.wav"


@pytest.fixture(scope="module")
def m():
    return Model()


@pytest.fixture(scope="module")
def wave():
    return load_wave(WAV)


def caption(m, wave, search, step=1600):
    s = Stream(m, search)
    for i in range(0, len(wave), step):
        s.accept(wave[i:i + step])
    s.finish()
    return s


# ---- features ------------------------------------------------------------------------------------

def test_features_match_kaldi(wave):
    knf = pytest.importorskip("kaldi_native_fbank")
    o = knf.FbankOptions()
    o.frame_opts.dither = 0; o.frame_opts.snip_edges = False; o.frame_opts.samp_freq = 16000
    o.mel_opts.num_bins = 80; o.mel_opts.high_freq = -400; o.mel_opts.low_freq = 20
    f = knf.OnlineFbank(o); f.accept_waveform(16000, wave.tolist()); f.input_finished()
    ref = np.array([f.get_frame(i) for i in range(f.num_frames_ready)])
    mine = fbank.features(wave)
    assert mine.shape == ref.shape
    assert np.abs(mine - ref).max() < 1e-3


def test_streaming_features_equal_batch(wave):
    whole = fbank.features(wave)
    parts = np.concatenate([fbank.features(wave, i, min(i + 37, len(whole))) for i in range(0, len(whole), 37)])
    assert np.allclose(whole, parts)


# ---- recognition -------------------------------------------------------------------------------

def reference(method):
    sherpa = pytest.importorskip("sherpa_onnx")
    f = lambda n: str(model_file(MODEL, n))
    return sherpa.OnlineRecognizer.from_transducer(tokens=str(MODEL / "tokens.txt"), encoder=f("encoder"), decoder=f("decoder"),
                                                   joiner=f("joiner"), num_threads=1, decoding_method=method, max_active_paths=4)


def sherpa_text(rec, wave):
    st = rec.create_stream()
    st.accept_waveform(16000, wave)
    st.accept_waveform(16000, np.zeros(int(0.66 * 16000), np.float32))
    st.input_finished()
    while rec.is_ready(st):
        rec.decode_stream(st)
    return rec.get_result(st).strip()


def test_greedy_matches_sherpa_onnx(m, wave):
    assert caption(m, wave, Greedy(m)).text == sherpa_text(reference("greedy_search"), wave)


def test_beam_matches_sherpa_onnx(m, wave):
    assert caption(m, wave, Beam(m)).text == sherpa_text(reference("modified_beam_search"), wave)


def test_chunk_size_does_not_change_the_words(m, wave):
    assert caption(m, wave, Greedy(m), 1600).text == caption(m, wave, Greedy(m), 4000).text == caption(m, wave, Greedy(m), 333).text


def test_phrase_list_fixes_the_word_it_lists(m):
    accented = load_wave(SAMPLES / "l2arctic-ska-a0010.wav")   # "I'm playing a single hand in what looks like a losing game"
    pieces = Pieces(m.tokens)
    plain = caption(m, accented, Beam(m)).text
    biased = caption(m, accented, Beam(m, bias=Bias([pieces.phrase("PLAYING")], 2.0))).text
    assert "PLAYING" not in plain and "PLAYING" in biased
    assert biased.split()[-6:] == plain.split()[-6:]  # the rest of the sentence is untouched


def test_a_dead_end_partial_match_gives_its_bonus_back(m):
    pieces = Pieces(m.tokens)
    b = Bias([pieces.phrase("HESTER PRYNNE")], 1.5)
    state, total = 0, 0.0
    for t in pieces.phrase("HESTER")[:2]:
        d, state = b.step(state, t)
        total += d
    assert total > 0 and abs(total + b.finish(state)) < 1e-9


# ---- learning from corrections -------------------------------------------------------------------

def test_alignment_groups_a_mistake_into_one_span():
    ops = align(words("i will fill up the car"), words("I will Philip the car"))
    assert ("err", 2, 4, 2, 3) in ops


def test_rules_need_repeats_and_must_beat_correct_uses():
    mem = Memory(min_rule_count=2)
    mem.learn("call fill up now", "call Philip now")
    assert mem.rewrite("fill up") == "FILL UP"          # once is not enough
    mem.learn("ask fill up", "ask Philip")
    assert mem.rewrite("tell fill up") == "TELL PHILIP"
    for _ in range(3):
        mem.learn("fill up the tank", "fill up the tank")  # the same words were right more often
    assert mem.rewrite("tell fill up") == "TELL FILL UP"


def test_missed_words_become_the_phrase_list():
    mem = Memory()
    mem.learn("the squalid quarter of the braffls", "the squalid quarter of the brothels")
    assert mem.phrases() == ["BROTHELS"]


def test_word_error_counts():
    assert wer_counts(words("a b c"), words("a x c d")) == (2, 4)
