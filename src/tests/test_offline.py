"""Offline test suite for TBS Eyes — no API key, real hardware optional.

Run:  .venv\\Scripts\\python.exe tests\\test_offline.py

1. Vosk model loads; 2s of synthesized silence runs through the pipeline
   without exceptions.
2. TTS-generated speech ("buddy" + a command) fed through the Ears state
   machine: wake word fires, command is captured. (Proves STT end to end
   with no microphone.)
3. No-API-key path: the real command-worker code path produces the
   "I need an API key" reply and queues it to TTS.
4. Live mic: 5s of real stream produces no exceptions (skipped if no mic).
"""

import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from brain import Brain, NoAPIKeyError, NO_KEY_MSG
from ears import Ears, SAMPLE_RATE

PASS, FAIL = [], []


def report(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))


def make_ears(**cb):
    model = config.vosk_model_path()
    if model is None:
        # Offline AI layout: model lives at <app root>\models\vosk\<name>
        p = ROOT.parent / "models" / "vosk" / config.MODEL_DIR_NAME
        model = p if p.is_dir() else None
    assert model is not None, "vosk model folder missing"
    return Ears(model_path=str(model), wake_word="buddy",
                silence_timeout=0.5, command_max_seconds=15.0, **cb)


# ---- helpers: synthesize speech offline via pyttsx3, resample to 16k ----
def tts_to_pcm16k(text: str, tmp: Path) -> bytes:
    import audioop
    import pyttsx3
    wav_path = tmp / "utt.wav"
    engine = pyttsx3.init()
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()
    engine.stop()
    del engine
    with wave.open(str(wav_path), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        data = audioop.lin2lin(data, width, 2)
    if ch == 2:
        data = audioop.tomono(data, 2, 0.5, 0.5)
    if rate != SAMPLE_RATE:
        data, _ = audioop.ratecv(data, 2, 1, rate, SAMPLE_RATE, None)
    return data


def feed(ears: Ears, pcm: bytes, chunk=8000):
    for i in range(0, len(pcm), chunk):
        ears.process_chunk(pcm[i:i + chunk])


# ---- test 1: model loads + silence runs clean ----
def test_silence():
    try:
        ears = make_ears()
        silence = b"\x00" * (SAMPLE_RATE * 2 * 2)  # 2 s int16 silence
        feed(ears, silence)
        report("model loads + 2s silence pipeline", True)
    except Exception as e:
        report("model loads + 2s silence pipeline", False, repr(e))


# ---- test 2: wake word + command via synthesized speech ----
def test_wake_and_command(tmp: Path):
    events = {"wake": 0, "commands": [], "partials": []}
    try:
        ears = make_ears(
            on_wake=lambda: events.__setitem__("wake", events["wake"] + 1),
            on_command=lambda t: events["commands"].append(t),
            on_partial=lambda t: events["partials"].append(t),
        )
        gap = b"\x00" * int(SAMPLE_RATE * 0.8 * 2)
        feed(ears, gap + tts_to_pcm16k("buddy", tmp) + gap)
        # simulate the silence timer elapsing with no audio
        if ears.state == "capturing":
            time.sleep(0.6)
            ears.process_chunk(b"\x00" * 8000)
        woke = events["wake"] >= 1
        report("wake word detected from synthesized 'buddy'", woke,
               f"wake={events['wake']} partials={events['partials'][-3:]}")

        # full loop: wake word + command in one utterance
        events["commands"].clear()
        speech = tts_to_pcm16k("tbs what do you see", tmp)
        feed(ears, gap + speech + gap)
        deadline = time.time() + 3
        while not events["commands"] and time.time() < deadline:
            time.sleep(0.55)
            ears.process_chunk(b"\x00" * 8000)  # silence ends the command
        got = events["commands"][0] if events["commands"] else ""
        report("command captured after wake", bool(got), f"command={got!r}")
    except Exception as e:
        report("wake word + command capture", False, repr(e))


# ---- test 3: no-API-key handler path (real worker code, fake app) ----
def test_no_key_path():
    import queue

    class FakeMouth:
        def __init__(self):
            self.q = queue.Queue()
        def speak(self, text):
            self.q.put(text)
        def is_speaking(self):
            return False

    class FakeVar:
        def get(self):
            return True

    class FakeFeed:
        def get_frame(self):
            return None

    class FakeApp:
        # engine='claude' forces the API path (the facade's default 'auto'
        # would resolve to LOCAL with models on disk and never raise).
        brain = Brain(api_key="", model="claude-opus-5", engine="claude",
                      prewarm=False)
        mouth = FakeMouth()
        speak_var = FakeVar()
        feed = FakeFeed()
        _thinking = True
        said = []
        def log_threadsafe(self, text, tag="tbs", sender=None):
            pass
        def set_status_threadsafe(self, text):
            pass
        def _say(self, text):
            self.said.append(text)
            if self.speak_var.get():
                self.mouth.speak(text)

    try:
        try:
            FakeApp.brain.ask("hello", None)
            raise AssertionError("Brain.ask with no key should raise NoAPIKeyError")
        except NoAPIKeyError:
            pass
        from tbs_app import TBSApp
        app = FakeApp()
        TBSApp._command_worker(app, "what do you see")
        ok = app.said == [NO_KEY_MSG] and app.mouth.q.get_nowait() == NO_KEY_MSG \
            and app._thinking is False
        report("no-API-key command path speaks the key message", ok,
               f"said={app.said}")
    except Exception as e:
        report("no-API-key command path", False, repr(e))


# ---- test 4: live mic stream for 5 s ----
def test_live_mic():
    errors = []
    ears = make_ears(on_error=lambda m: errors.append(m))
    started = ears.start()
    if not started:
        report("live mic 5s", True, "SKIPPED — no microphone (graceful degrade verified: "
               + (errors[0] if errors else "no error msg"))
        return
    time.sleep(5)
    ears.stop()
    stt_errors = [e for e in errors if "STT error" in e]
    report("live mic 5s stream, no STT exceptions", not stt_errors,
           f"errors={errors}" if errors else "clean")


def main():
    tmp = ROOT / "tests" / "_tmp"
    tmp.mkdir(exist_ok=True)
    test_silence()
    test_wake_and_command(tmp)
    test_no_key_path()
    test_live_mic()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
