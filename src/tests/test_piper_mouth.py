"""Piper mouth tests — run from the app root with:
  _dev\\venv\\Scripts\\python.exe src\\tests\\test_piper_mouth.py

Real tests (actual synthesis + actual audio out the speakers):
  1. piper_direct     — piper.exe synthesizes a real non-silent wav and
                        sounddevice playback completes (blocking returns).
  2. forced_fallback  — a bad piper voice makes _speak_once fall back to
                        SAPI without raising; the fallback demonstrably fires.
  3. callbacks        — on_start fires before the first audio sample plays,
                        on_done fires after playback ends.
  4. queue_order      — 3 queued utterances all speak, in order;
                        is_speaking() is True throughout and False after.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd

import mouth as mouth_mod
from mouth import Mouth, available_voices


def test_piper_direct():
    m = Mouth.__new__(Mouth)  # no thread — call the engine method directly
    m.piper_voice = "auto"

    played = {}
    real_play = sd.play

    def spy_play(samples, samplerate, **kw):
        arr = np.asarray(samples)
        played["n"] = len(arr)
        played["peak"] = int(np.abs(arr).max())
        played["sr"] = samplerate
        played["t_play"] = time.time()
        return real_play(samples, samplerate, **kw)

    sd.play = spy_play
    t0 = time.time()
    try:
        m._speak_piper("Local voice online. All systems are running offline.")
    finally:
        sd.play = real_play
    elapsed = time.time() - t0
    assert played.get("n", 0) > 10_000, f"too few samples: {played}"
    assert played.get("peak", 0) > 500, f"silent PCM: {played}"
    voice = m._resolve_voice().stem
    # blocking playback: total time must at least cover the audio duration
    audio_s = played["n"] / played["sr"]
    assert elapsed >= audio_s * 0.9, (
        f"playback did not block to completion: {elapsed:.2f}s < {audio_s:.2f}s")
    print(f"PASS piper_direct — voice {voice}: {played['n']} samples @ "
          f"{played['sr']} Hz ({audio_s:.2f}s audio), peak {played['peak']}, "
          f"playback completed in {elapsed:.2f}s total")
    return voice


def test_forced_fallback():
    m = Mouth.__new__(Mouth)
    m.piper_voice = r"Z:\nope\does-not-exist.onnx"  # force piper failure
    m.engine = mouth_mod.ENGINE_PIPER
    m.rate = 285  # fast SAPI so the test is quick
    m._piper_warned = False

    fired = {}
    real_sapi = m._speak_sapi

    def spy_sapi(text):
        fired["text"] = text
        return real_sapi(text)

    m._speak_sapi = spy_sapi
    try:
        m._speak_once("Fallback voice check.")  # must not raise
    except Exception as e:
        raise AssertionError(f"_speak_once raised through fallback: {e}")
    assert fired.get("text") == "Fallback voice check.", \
        "SAPI fallback did not fire"
    assert m._piper_warned, "piper failure was not logged/flagged"
    print("PASS forced_fallback — bad voice path -> SAPI spoke, no exception, "
          "one warning flagged")


def test_callbacks():
    events = []
    real_play = sd.play

    def spy_play(samples, samplerate, **kw):
        events.append(("audio_start", time.time()))
        r = real_play(samples, samplerate, **kw)
        events.append(("audio_end", time.time()))
        return r

    sd.play = spy_play
    m = Mouth(engine=mouth_mod.ENGINE_PIPER,
              on_start=lambda: events.append(("on_start", time.time())),
              on_done=lambda: events.append(("on_done", time.time())))
    try:
        m.speak("Callback order test.")
        deadline = time.time() + 60
        while time.time() < deadline:
            if [e for e in events if e[0] == "on_done"]:
                break
            time.sleep(0.05)
        m.close()
        names = [e[0] for e in events]
        assert names == ["on_start", "audio_start", "audio_end", "on_done"], \
            f"bad event order: {names}"
    finally:
        sd.play = real_play
    print(f"PASS callbacks — order {', '.join(n for n, _ in events)} "
          f"(on_start {events[1][1]-events[0][1]:.3f}s before audio; "
          f"on_done {events[3][1]-events[2][1]:.3f}s after audio end)")


def test_queue_order():
    spoken = []
    m = Mouth(engine=mouth_mod.ENGINE_PIPER)
    real_piper = m._speak_piper

    def spy_piper(text):
        spoken.append(text)
        return real_piper(text)

    m._speak_piper = spy_piper
    texts = ["First utterance.", "Second utterance.", "Third one."]
    for t in texts:
        m.speak(t)
    m.speak("")        # empty text must be ignored
    m.speak("   ")     # whitespace-only too

    assert m.is_speaking(), "is_speaking() False right after queueing 3"
    sampled_true = 0
    deadline = time.time() + 120
    while m.is_speaking() and time.time() < deadline:
        sampled_true += 1
        time.sleep(0.1)
    assert len(spoken) == 3, f"expected 3 utterances, spoke {spoken}"
    assert spoken == texts, f"out of order: {spoken}"
    assert not m.is_speaking(), "is_speaking() still True after drain"
    assert sampled_true > 5, "is_speaking() was not observably True during speech"
    m.close()
    print(f"PASS queue_order — 3/3 spoken in order, is_speaking() True for "
          f"~{sampled_true * 0.1:.1f}s during, False after; empty/whitespace "
          f"inputs ignored")


if __name__ == "__main__":
    print(f"voices available: {available_voices()}")
    failures = 0
    for fn in (test_piper_direct, test_forced_fallback, test_callbacks,
               test_queue_order):
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f"FAIL {fn.__name__} — {type(e).__name__}: {e}")
    print(f"\n{'ALL 4 TESTS PASSED' if failures == 0 else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
