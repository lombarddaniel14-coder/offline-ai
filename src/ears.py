"""Ears — offline speech-to-text + wake-word state machine for TBS Eyes.

Vosk (offline, no API key) recognizes a continuous 16 kHz mono mic stream
from sounddevice. Two listening states:

  IDLE      — wake word armed: partial results are scanned for the wake word
              ("buddy" by default). On detection the recognizer is reset and
              we switch to CAPTURING.
  CAPTURING — the user is speaking a command: partial + final results are
              accumulated; ~1.5s with no new words ends the command and fires
              on_command(text). A hard cap ends over-long commands.

While the assistant is speaking (TTS), call mute() so TBS doesn't hear
itself; unmute() re-arms with a fresh recognizer state.

All callbacks fire on the audio worker thread — marshal to the UI yourself.
Audio processing is separated into process_chunk() so tests can feed raw PCM
bytes without a microphone.
"""

import json
import queue
import threading
import time

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # 0.25 s per chunk — responsive partials

STATE_IDLE = "idle"            # wake word armed
STATE_CAPTURING = "capturing"  # command being spoken

# Pseudo-devices sounddevice lists that aren't real microphones.
_PSEUDO_DEVICES = ("microsoft sound mapper", "primary sound capture")


def list_microphones() -> list[tuple[int, str]]:
    """Return [(device_index, name)] — each physical input device ONCE.

    sounddevice lists every device 4+ times (once per host API: MME, DirectSound,
    WASAPI, WDM-KS). We restrict to a single host API — WASAPI if present, else
    MME — and skip the "Sound Mapper"/"Primary Sound Capture" pseudo-devices.
    Returns [] if sounddevice/PortAudio is unavailable (never raises).
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return []
    chosen = None
    for pref in ("wasapi", "mme"):
        for i, ha in enumerate(hostapis):
            if pref in ha["name"].lower():
                chosen = i
                break
        if chosen is not None:
            break
    mics = []
    for idx, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        if chosen is not None and d["hostapi"] != chosen:
            continue
        if any(p in d["name"].lower() for p in _PSEUDO_DEVICES):
            continue
        mics.append((idx, d["name"]))
    return mics


def resolve_mic_device(index, name: str = "", log=None):
    """Turn a saved (index, name) pair into a usable device index, or None
    for system default.

    Device indexes shift when USB devices are re-plugged, so the saved name is
    the source of truth: if the device at `index` no longer matches `name`
    (substring match either way — MME truncates names to ~31 chars), re-resolve
    by name across the deduped mic list. Falls back to None (system default)
    with a log line if the name can't be found. Never raises.
    """
    if index is None:
        return None
    try:
        import sounddevice as sd
        devices = sd.query_devices()
    except Exception:
        return None

    def name_matches(a: str, b: str) -> bool:
        a, b = a.strip().lower(), b.strip().lower()
        return bool(a) and bool(b) and (a in b or b in a)

    try:
        index = int(index)
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(devices) and devices[index]["max_input_channels"] > 0:
        if not name or name_matches(name, devices[index]["name"]):
            return index
    # Stale index — re-resolve by name.
    if name:
        for idx, dev_name in list_microphones():
            if name_matches(name, dev_name):
                if log:
                    log(f"Microphone '{name}' moved to device index {idx} — re-resolved by name.")
                return idx
    if log:
        log(f"Saved microphone '{name or index}' not found — using system default.")
    return None


class Ears:
    def __init__(self, model_path: str, wake_word: str = "buddy",
                 silence_timeout: float = 1.5, command_max_seconds: float = 15.0,
                 device: int | None = None,
                 on_wake=None, on_partial=None, on_command=None,
                 on_state=None, on_error=None):
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)  # silence Vosk's console spam
        self._KaldiRecognizer = KaldiRecognizer
        self._model = Model(str(model_path))
        self._rec = KaldiRecognizer(self._model, SAMPLE_RATE)

        self.device = device  # sounddevice input index; None = system default
        self.wake_word = (wake_word or "buddy").strip().lower()
        self.silence_timeout = silence_timeout
        self.command_max_seconds = command_max_seconds

        self.on_wake = on_wake or (lambda: None)
        self.on_partial = on_partial or (lambda text: None)
        self.on_command = on_command or (lambda text: None)
        self.on_state = on_state or (lambda state: None)
        self.on_error = on_error or (lambda msg: None)

        self.state = STATE_IDLE
        self.wake_enabled = True   # wake-word toggle (push-to-talk still works)
        self._muted = threading.Event()
        self._stop = threading.Event()
        self._reset_pending = threading.Event()
        self._force_capture = threading.Event()

        # capture accumulation
        self._finals: list[str] = []
        self._last_partial = ""
        self._capture_started = 0.0
        self._last_speech = 0.0

        self._audio_q: queue.Queue = queue.Queue(maxsize=50)
        self._worker: threading.Thread | None = None
        self._stream = None
        self.running = False

    # ---- lifecycle ----
    def start(self) -> bool:
        """Open the mic and start processing. Returns False (and fires
        on_error) if no input device is available — never raises."""
        self._stop.clear()
        try:
            import sounddevice as sd

            def callback(indata, frames, time_info, status):
                if self._stop.is_set():
                    return
                try:
                    self._audio_q.put_nowait(bytes(indata))
                except queue.Full:
                    pass  # drop audio rather than back up

            def open_stream(device):
                # WASAPI shared mode refuses non-native sample rates (we need
                # 16 kHz, devices run 44.1/48k) unless auto_convert is on.
                extra = None
                if device is not None:
                    try:
                        hostapi = sd.query_hostapis(
                            sd.query_devices(device)["hostapi"])["name"]
                        if "wasapi" in hostapi.lower():
                            extra = sd.WasapiSettings(auto_convert=True)
                    except Exception:
                        pass
                stream = sd.RawInputStream(
                    samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16",
                    channels=1, device=device, extra_settings=extra,
                    callback=callback)
                stream.start()
                return stream

            try:
                self._stream = open_stream(self.device)
            except Exception as e:
                if self.device is None:
                    raise
                # Selected device failed — fall back to the system default.
                self.on_error(f"Microphone device {self.device} failed ({e}) — "
                              "falling back to system default.")
                self.device = None
                self._stream = open_stream(None)
        except Exception as e:
            self._stream = None
            self.on_error(f"Microphone unavailable: {e}")
            return False

        self._worker = threading.Thread(target=self._run, daemon=True, name="ears")
        self._worker.start()
        self.running = True
        self._set_state(STATE_IDLE)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=3)
            self._worker = None
        self.running = False

    # ---- controls (any thread) ----
    def mute(self) -> None:
        """Stop hearing (used while TTS speaks). Drops audio + resets state."""
        self._muted.set()

    def unmute(self) -> None:
        self._reset_pending.set()  # fresh recognizer state on resume
        self._muted.clear()

    def start_command_capture(self) -> None:
        """Push-to-talk: jump straight to CAPTURING without the wake word."""
        self._force_capture.set()

    # ---- worker ----
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._audio_q.get(timeout=0.25)
            except queue.Empty:
                data = None
            try:
                self._tick(data)
            except Exception as e:
                self.on_error(f"STT error: {e}")
                self._reset_recognizer()
                self._to_idle()

    def _tick(self, data: bytes | None) -> None:
        """One scheduler step: handle control flags, then feed audio (if any)."""
        if self._force_capture.is_set():
            self._force_capture.clear()
            self._begin_capture()
        if self._muted.is_set():
            return  # discard audio while TTS is speaking
        if self._reset_pending.is_set():
            self._reset_pending.clear()
            self._reset_recognizer()
        if data is not None:
            self.process_chunk(data)
        elif self.state == STATE_CAPTURING:
            self._check_capture_end()  # silence can also elapse with no audio

    def process_chunk(self, data: bytes) -> None:
        """Feed one chunk of 16 kHz mono int16 PCM through the state machine.
        Public so tests can drive the pipeline without a microphone."""
        if self._rec.AcceptWaveform(data):
            text = json.loads(self._rec.Result()).get("text", "").strip()
            self._on_final(text)
        else:
            partial = json.loads(self._rec.PartialResult()).get("partial", "").strip()
            self._on_partial_text(partial)
        if self.state == STATE_CAPTURING:
            self._check_capture_end()

    # ---- state machine internals ----
    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.on_state(state)

    def _reset_recognizer(self) -> None:
        self._rec = self._KaldiRecognizer(self._model, SAMPLE_RATE)

    def _to_idle(self) -> None:
        self._finals = []
        self._last_partial = ""
        self._set_state(STATE_IDLE)

    def _begin_capture(self) -> None:
        self._reset_recognizer()
        self._finals = []
        self._last_partial = ""
        now = time.monotonic()
        self._capture_started = now
        self._last_speech = now
        self._set_state(STATE_CAPTURING)

    def _on_partial_text(self, partial: str) -> None:
        if self.state == STATE_IDLE:
            if self.wake_enabled and partial and self.wake_word in partial.lower().split():
                self.on_wake()
                self._begin_capture()
        elif self.state == STATE_CAPTURING:
            if partial != self._last_partial:
                self._last_partial = partial
                if partial:
                    self._last_speech = time.monotonic()
                self.on_partial(" ".join(self._finals + [partial]).strip())

    def _on_final(self, text: str) -> None:
        if self.state == STATE_IDLE:
            if self.wake_enabled and text and self.wake_word in text.lower().split():
                self.on_wake()
                self._begin_capture()
        elif self.state == STATE_CAPTURING:
            if text:
                self._finals.append(text)
                self._last_speech = time.monotonic()
                self.on_partial(" ".join(self._finals).strip())
            self._last_partial = ""

    def _check_capture_end(self) -> None:
        now = time.monotonic()
        heard = " ".join(self._finals + ([self._last_partial] if self._last_partial else [])).strip()
        timed_out = now - self._capture_started > self.command_max_seconds
        silent = now - self._last_speech > self.silence_timeout
        if timed_out or (silent and heard):
            self._finish_capture()
        elif silent and not heard and now - self._capture_started > 5.0:
            self._to_idle()  # woke up but nothing was said — stand down

    def _finish_capture(self) -> None:
        # Flush whatever the recognizer still holds.
        try:
            tail = json.loads(self._rec.FinalResult()).get("text", "").strip()
        except Exception:
            tail = ""
        if tail:
            self._finals.append(tail)
        command = " ".join(self._finals).strip()
        # Drop a leading wake word ("tbs what time is it" -> "what time is it").
        words = command.split()
        if words and words[0].lower() == self.wake_word:
            command = " ".join(words[1:]).strip()
        self._reset_recognizer()
        self._to_idle()
        if command:
            self.on_command(command)
