"""Mouth — offline-first TTS for TBS Offline.

Two engines behind one queue thread:

  piper (default) — local neural TTS. Runs <root>\\piper\\piper.exe with a
         voice model from <root>\\models\\piper\\*.onnx, text on stdin, wav to
         a temp file, decoded with wave+numpy and played through sounddevice
         (blocking inside the TTS thread, so the mic-mute window covers the
         whole playback). 100%% offline, 100%% portable — all paths resolved
         from the app root at runtime.
  sapi  (fallback) — pyttsx3 / Windows SAPI. A fresh engine per utterance
         avoids the SAPI5 'second runAndWait hangs' issue. Used when
         piper.exe or the voice models are missing, or a piper synth fails.

The speaker exposes is_speaking() and fires on_start/on_done callbacks so the
ears can be muted while TBS talks (otherwise it hears itself).
"""

import logging
import os
import queue
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

log = logging.getLogger("mouth")

ENGINE_PIPER = "piper"
ENGINE_SAPI = "sapi"

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PIPER_TIMEOUT = 120  # seconds; piper does ~1s of audio per ~0.1s CPU, so ample


# -- portable path resolution ----------------------------------------------
def _root_dirs():
    """(piper_dir, piper_models_dir) via portable.py, with an identical-logic
    fallback for when portable.py hasn't landed yet (concurrent build)."""
    try:
        import portable
        return Path(portable.PIPER), Path(portable.MODELS) / "piper"
    except Exception:
        import sys
        if getattr(sys, "frozen", False):
            root = Path(sys.executable).resolve().parents[2]
        else:
            root = Path(__file__).resolve().parents[1]
        return root / "piper", root / "models" / "piper"


def available_voices():
    """Stems of the .onnx voice models present in models\\piper (may be [])."""
    _, voices_dir = _root_dirs()
    try:
        return sorted(p.stem for p in voices_dir.glob("*.onnx"))
    except OSError:
        return []


def _tts_tmp_dir() -> Path:
    """Temp wavs live INSIDE the app root (<root>\\_tmp) so TTS never leaves
    a trace on the host PC — even if the process is killed mid-playback, the
    leaked file sits on the flash drive and is swept on the next run."""
    piper_dir, _ = _root_dirs()
    return piper_dir.parent / "_tmp"


def _sweep_stale_wavs() -> None:
    """Best-effort cleanup of wavs leaked by a killed/crashed earlier run."""
    try:
        for p in _tts_tmp_dir().glob("tbs_tts_*.wav"):
            try:
                p.unlink()
            except OSError:
                pass  # in use by a concurrent Mouth — its finally cleans up
    except OSError:
        pass


class PiperError(RuntimeError):
    pass


class Mouth:
    def __init__(self, rate: int = 180, engine: str = ENGINE_PIPER,
                 piper_voice: str = "auto",
                 on_start=None, on_done=None, **_legacy):
        """on_start()/on_done() are called from the TTS thread around each
        drained batch of utterances (for muting the mic).

        rate        — SAPI (pyttsx3) speed, words per minute.
        engine      — "piper" (local neural, default) or "sapi".
        piper_voice — "auto" (prefer en_GB), a model stem
                      ("en_GB-alan-medium"), a file name, or a full path.
        _legacy     — swallows retired kwargs (neural_voice, neural_rate)
                      from older callers; ignored.
        """
        self.rate = rate
        self.engine = engine
        self.piper_voice = piper_voice
        self.on_start = on_start or (lambda: None)
        self.on_done = on_done or (lambda: None)
        self._piper_warned = False
        self._q: queue.Queue = queue.Queue()
        self._speaking = threading.Event()
        _sweep_stale_wavs()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tts")
        self._thread.start()

    def _run(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                # close() may queue this sentinel while an utterance is still
                # in flight — the drain check below then never ran, so clear
                # the speaking flag and fire on_done here. Otherwise the mic
                # would stay muted forever after e.g. Settings' "Test voice".
                if self._speaking.is_set():
                    self._speaking.clear()
                    try:
                        self.on_done()
                    except Exception:
                        pass
                return
            self._speaking.set()
            try:
                self.on_start()
            except Exception:
                pass
            try:
                self._speak_once(text)
            except Exception:
                pass  # TTS failure must never take the app down
            # Only clear 'speaking' once the queue is drained, so the mic stays
            # muted across consecutive queued utterances.
            if self._q.empty():
                self._speaking.clear()
                try:
                    self.on_done()
                except Exception:
                    pass

    # -- engines -----------------------------------------------------------
    def _speak_once(self, text: str) -> None:
        if self.engine == ENGINE_PIPER:
            try:
                self._speak_piper(text)
                return
            except Exception as e:
                if not self._piper_warned:
                    log.warning("Piper TTS unavailable (%s: %s) — falling back "
                                "to SAPI.", type(e).__name__, e)
                    self._piper_warned = True
        self._speak_sapi(text)

    def _resolve_voice(self) -> Path:
        """Pick the .onnx voice model. 'auto' prefers an en_GB voice."""
        _, voices_dir = _root_dirs()
        voices = sorted(voices_dir.glob("*.onnx"))
        want = (self.piper_voice or "auto").strip()
        if want.lower() != "auto":
            for v in voices:
                if want in (v.stem, v.name):
                    return v
            p = Path(want)
            if p.is_file():
                return p
            raise PiperError(f"piper voice not found: {want!r} "
                             f"(have: {[v.stem for v in voices]})")
        if not voices:
            raise PiperError(f"no .onnx voices in {voices_dir}")
        for v in voices:
            if v.stem.lower().startswith("en_gb"):
                return v
        return voices[0]

    def _speak_piper(self, text: str) -> None:
        """piper.exe -> temp wav -> wave+numpy decode -> sounddevice playback.
        Raises on any failure (exe/voice missing, empty or silent result)."""
        import numpy as np
        import sounddevice as sd

        piper_dir, _ = _root_dirs()
        exe = piper_dir / "piper.exe"
        if not exe.is_file():
            raise PiperError(f"piper.exe not found at {exe}")
        model = self._resolve_voice()
        cfg = Path(str(model) + ".json")
        if not cfg.is_file():
            raise PiperError(f"voice config missing: {cfg}")

        # Wav goes inside the app root (zero host traces); only if the drive
        # is write-locked fall back to the system temp dir (mkstemp+unlink).
        tmp_dir = _tts_tmp_dir()
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="tbs_tts_",
                                            dir=str(tmp_dir))
        except OSError:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="tbs_tts_")
        os.close(fd)
        try:
            proc = subprocess.run(
                [str(exe), "--model", str(model), "--config", str(cfg),
                 "--output_file", wav_path],
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=_CREATE_NO_WINDOW, timeout=_PIPER_TIMEOUT)
            if proc.returncode != 0:
                err = proc.stderr.decode("utf-8", "replace").strip()[-300:]
                raise PiperError(f"piper exited {proc.returncode}: {err}")

            with wave.open(wav_path, "rb") as wf:
                n_frames = wf.getnframes()
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels)
            if len(samples) == 0 or int(np.abs(samples).max()) == 0:
                raise PiperError("piper produced empty/silent audio")
            sd.play(samples, sample_rate, blocking=True)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _speak_sapi(self, text: str) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            try:
                engine.setProperty("rate", int(self.rate))
            except Exception:
                pass
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception:
            pass  # even the fallback must never take the app down

    # -- public API --------------------------------------------------------
    def speak(self, text: str) -> None:
        if text and text.strip():
            self._q.put(text)

    def is_speaking(self) -> bool:
        return self._speaking.is_set() or not self._q.empty()

    def close(self) -> None:
        """Let queued utterances finish, then end the TTS thread."""
        self._q.put(None)

    def shutdown(self) -> None:
        # Drain pending utterances so shutdown is prompt, then poison-pill.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(None)
