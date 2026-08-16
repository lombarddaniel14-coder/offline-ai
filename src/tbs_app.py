"""TBS — OFFLINE AI. Voice-first desktop assistant with eyes, fully portable.

Orchestrates ears (Vosk wake word + STT), eyes (camera), brain (local
llama.cpp server and/or Claude, behind the brain facade), and mouth
(Piper/SAPI TTS) behind a CustomTkinter GUI. Everything resolves paths
through portable.app_root() — the folder runs from any drive letter.

App states (status ring, top-left):
  OFFLINE    — mic unavailable or STT model missing; push-to-talk/voice dead,
               camera + text path may still work
  LISTENING  — idle, wake word armed ("buddy")
  CAPTURING  — command being spoken; live transcript shown
  THINKING   — query in flight to the brain
  SPEAKING   — TTS talking (mic muted so TBS doesn't hear itself)

Run from source:  _dev\\venv\\Scripts\\python.exe src\\tbs_app.py
"""

import inspect
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
from PIL import Image

try:
    import anthropic
except Exception:  # anthropic SDK optional — local-only installs work without it
    anthropic = None

import portable
import config

# The brain facade may still be mid-migration (local_brain/server_manager land
# separately) — guard the import so the GUI always launches.
try:
    from brain import Brain, NoAPIKeyError, NOTABLE_TAG, NO_KEY_MSG
    BRAIN_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - only when brain module is broken
    Brain = None
    BRAIN_IMPORT_ERROR = _e
    NOTABLE_TAG = "[NOTABLE]"
    NO_KEY_MSG = "I need an API key before I can think, sir. Add one in Settings."

    class NoAPIKeyError(Exception):
        pass

from ears import Ears, STATE_CAPTURING, list_microphones, resolve_mic_device
from eyes import Camera, CameraError, CameraFeed, frame_to_jpeg_b64, list_cameras, save_frame
from mouth import Mouth

# Exception classes for the command/watch workers. With no anthropic SDK the
# placeholders never match and the generic handler catches everything.
if anthropic is not None:
    RATE_LIMIT_ERRORS = (anthropic.RateLimitError,)
    CONNECTION_ERRORS = (anthropic.APIConnectionError,)
    API_ERRORS = (anthropic.APIError,)
else:
    class _NeverRaised(Exception):
        pass
    RATE_LIMIT_ERRORS = CONNECTION_ERRORS = API_ERRORS = (_NeverRaised,)

# ----- Theme (arc-reactor: deep navy, cyan, amber) ------------------------
ACCENT = "#22d3ee"        # cyan
ACCENT_BRIGHT = "#67e8f9"
ACCENT_DIM = "#0e7490"
AMBER = "#fbbf24"
AMBER_DEEP = "#f59e0b"
BG = "#0b1020"            # deep navy
PANEL = "#111831"
PANEL_2 = "#0e142a"
TEXT_DIM = "#8b9dc3"
OFFLINE_GRAY = "#475569"

MODEL_CHOICES = ["claude-opus-5", "claude-sonnet-4-6", "claude-haiku-4-5"]

MIC_DEFAULT_LABEL = "System default"

ENGINE_LABELS = {"auto": "Auto", "local": "Local only", "claude": "Claude only"}
ENGINE_VALUES = {v: k for k, v in ENGINE_LABELS.items()}

LOCAL_MODEL_AUTO_LABEL = "Auto (recommended)"
VOICE_AUTO_LABEL = "Auto (best Piper voice)"
VOICE_CLASSIC_LABEL = "Classic (SAPI)"

TEST_LINE = "Good afternoon, sir. All systems are fully operational."
LOADING_LOCAL_MSG = "Loading local brain… (first load can take a minute)"

PREVIEW_W, PREVIEW_H = 480, 270

STATE_COLORS = {
    "OFFLINE": OFFLINE_GRAY,
    "LISTENING": ACCENT_DIM,
    "CAPTURING": ACCENT_BRIGHT,
    "THINKING": AMBER,
    "SPEAKING": AMBER_DEEP,
}


def resource_path(rel: str) -> Path:
    """Resolve a bundled resource both from source and from a PyInstaller build."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def construct_compat(cls, **kwargs):
    """Instantiate cls passing only the keyword args its __init__ accepts.

    The brain and mouth modules are migrating to new signatures on separate
    branches; this keeps the GUI working with either generation. If __init__
    takes **kwargs everything is passed through.
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return cls(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cls(**kwargs)
    return cls(**{k: v for k, v in kwargs.items() if k in params})


def load_llm_models() -> list[dict]:
    """Entries from <root>\\models\\llm\\MODELS.json ([] if missing/corrupt)."""
    try:
        with open(portable.MODELS / "llm" / "MODELS.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return [d for d in data if isinstance(d, dict) and d.get("model")] \
            if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def list_piper_voices() -> list[str]:
    """Voice names (onnx stems) available in <root>\\models\\piper."""
    try:
        return sorted(p.stem for p in (portable.MODELS / "piper").glob("*.onnx"))
    except OSError:
        return []


def wake_ack() -> None:
    """Short non-TTS acknowledgment beep (TTS would mute the mic and eat the
    start of the command). Never raises."""
    def _beep():
        try:
            import winsound
            winsound.Beep(1200, 120)
            winsound.Beep(1600, 100)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True, name="wake-beep").start()


# ----- Status ring (arc-reactor style) ------------------------------------
class StatusRing(ctk.CTkFrame):
    """Concentric rings whose color reflects the app state."""

    SIZE = 84

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        import tkinter as tk
        self.canvas = tk.Canvas(self, width=self.SIZE, height=self.SIZE,
                                bg=PANEL_2, highlightthickness=0)
        self.canvas.pack()
        self._render(OFFLINE_GRAY)

    def _render(self, color: str):
        c = self.canvas
        c.delete("all")
        m = self.SIZE / 2
        for r, w in ((36, 3), (27, 2), (18, 2)):
            c.create_oval(m - r, m - r, m + r, m + r, outline=color, width=w)
        c.create_oval(m - 8, m - 8, m + 8, m + 8, fill=color, outline="")

    def set_state(self, state: str):
        self._render(STATE_COLORS.get(state, OFFLINE_GRAY))


# ----- Settings dialog ----------------------------------------------------
class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, app: "TBSApp"):
        super().__init__(app)
        self.app = app
        self.title("TBS — Settings")
        # Fit small host screens (e.g. 1366x768 laptops): clamp the height so
        # the bottom Save/Cancel row is always reachable; the body scrolls.
        height = min(920, max(420, self.winfo_screenheight() - 120))
        self.geometry(f"480x{height}")
        self.resizable(False, True)
        self.configure(fg_color=PANEL)
        self.transient(app)
        self.after(120, self._grab)  # grab after the window actually maps

        s = config.load_settings()
        pad = {"padx": 18, "pady": (10, 0)}

        ctk.CTkLabel(self, text="Settings", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=ACCENT).pack(anchor="w", padx=18, pady=(14, 2))

        # Buttons pack FIRST (side=bottom) so they always stay visible; the
        # settings body scrolls in whatever space is left.
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(side="bottom", fill="x", padx=18, pady=12)
        ctk.CTkButton(btn_row, text="Save", width=120, fg_color=ACCENT_DIM,
                      hover_color=ACCENT, command=self._save).pack(side="right")
        ctk.CTkButton(btn_row, text="Cancel", width=100, fg_color="#334155",
                      hover_color="#475569", command=self.destroy).pack(side="right", padx=(0, 10))

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=2)

        ctk.CTkLabel(body, text="Engine", text_color=TEXT_DIM).pack(anchor="w", **pad)
        self.engine_seg = ctk.CTkSegmentedButton(
            body, values=list(ENGINE_LABELS.values()),
            selected_color=ACCENT_DIM, selected_hover_color=ACCENT)
        self.engine_seg.set(ENGINE_LABELS.get(s["engine"], ENGINE_LABELS["local"]))
        self.engine_seg.pack(anchor="w", padx=18)

        ctk.CTkLabel(body, text="Local model", text_color=TEXT_DIM).pack(anchor="w", **pad)
        self._local_map: dict[str, str] = {}  # dropdown label -> gguf filename
        local_values = [LOCAL_MODEL_AUTO_LABEL]
        for entry in load_llm_models():
            label = f"{str(entry.get('tier', '?')).capitalize()} — {entry['model']}"
            self._local_map[label] = entry["model"]
            local_values.append(label)
        self.local_box = ctk.CTkComboBox(body, values=local_values, width=420,
                                         state="readonly",
                                         button_color=ACCENT_DIM, border_color=ACCENT_DIM)
        selected_local = LOCAL_MODEL_AUTO_LABEL
        for label, fname in self._local_map.items():
            if fname == s["local_model"]:
                selected_local = label
                break
        self.local_box.set(selected_local)
        self.local_box.pack(anchor="w", padx=18)

        ctk.CTkLabel(body, text="Anthropic API key (optional — Claude/Auto engines)",
                     text_color=TEXT_DIM).pack(anchor="w", **pad)
        self.key_entry = ctk.CTkEntry(body, show="•", width=420, placeholder_text="sk-ant-…")
        if s["api_key"]:
            self.key_entry.insert(0, s["api_key"])
        self.key_entry.pack(anchor="w", padx=18)

        ctk.CTkLabel(body, text="Claude model", text_color=TEXT_DIM).pack(anchor="w", **pad)
        values = MODEL_CHOICES if s["model"] in MODEL_CHOICES else [s["model"]] + MODEL_CHOICES
        self.model_box = ctk.CTkComboBox(body, values=values, width=420,
                                         button_color=ACCENT_DIM, border_color=ACCENT_DIM)
        self.model_box.set(s["model"])
        self.model_box.pack(anchor="w", padx=18)

        ctk.CTkLabel(body, text="Wake word", text_color=TEXT_DIM).pack(anchor="w", **pad)
        self.wake_entry = ctk.CTkEntry(body, width=200)
        self.wake_entry.insert(0, s["wake_word"])
        self.wake_entry.pack(anchor="w", padx=18)

        ctk.CTkLabel(body, text="Microphone", text_color=TEXT_DIM).pack(anchor="w", **pad)
        mic_row = ctk.CTkFrame(body, fg_color="transparent")
        mic_row.pack(anchor="w", padx=18, fill="x")
        self._mic_map: dict[str, int] = {}  # dropdown label -> device index
        self.mic_box = ctk.CTkComboBox(mic_row, values=[MIC_DEFAULT_LABEL], width=320,
                                       state="readonly",
                                       button_color=ACCENT_DIM, border_color=ACCENT_DIM)
        self.mic_box.set(MIC_DEFAULT_LABEL)
        self.mic_box.pack(side="left")
        self.mic_rescan_btn = ctk.CTkButton(mic_row, text="Rescan", width=90,
                                            fg_color=ACCENT_DIM, hover_color=ACCENT,
                                            command=self._rescan_mics)
        self.mic_rescan_btn.pack(side="left", padx=(10, 0))
        self._populate_mics(list_microphones(), s["mic_device"], s["mic_device_name"])

        ctk.CTkLabel(body, text="Camera", text_color=TEXT_DIM).pack(anchor="w", **pad)
        cam_row = ctk.CTkFrame(body, fg_color="transparent")
        cam_row.pack(anchor="w", padx=18, fill="x")
        self.cam_box = ctk.CTkComboBox(cam_row, values=[str(s["camera_index"])], width=320,
                                       button_color=ACCENT_DIM, border_color=ACCENT_DIM)
        self.cam_box.set(str(s["camera_index"]))
        self.cam_box.pack(side="left")
        self.rescan_btn = ctk.CTkButton(cam_row, text="Rescan", width=90,
                                        fg_color=ACCENT_DIM, hover_color=ACCENT,
                                        command=self._rescan)
        self.rescan_btn.pack(side="left", padx=(10, 0))

        ctk.CTkLabel(body, text="Watch interval (seconds)", text_color=TEXT_DIM).pack(anchor="w", **pad)
        self.interval_entry = ctk.CTkEntry(body, width=120)
        self.interval_entry.insert(0, str(int(s["watch_interval"])))
        self.interval_entry.pack(anchor="w", padx=18)

        ctk.CTkLabel(body, text="Voice", text_color=TEXT_DIM).pack(anchor="w", **pad)
        voice_values = [VOICE_AUTO_LABEL] + list_piper_voices() + [VOICE_CLASSIC_LABEL]
        self.voice_box = ctk.CTkComboBox(body, values=voice_values, width=420,
                                         state="readonly",
                                         button_color=ACCENT_DIM, border_color=ACCENT_DIM)
        if s["voice_engine"] == "classic":
            self.voice_box.set(VOICE_CLASSIC_LABEL)
        elif s["piper_voice"] in voice_values:
            self.voice_box.set(s["piper_voice"])
        else:
            self.voice_box.set(VOICE_AUTO_LABEL)
        self.voice_box.pack(anchor="w", padx=18)

        ctk.CTkLabel(body, text="Classic voice rate (words/min)", text_color=TEXT_DIM).pack(anchor="w", **pad)
        rate_row = ctk.CTkFrame(body, fg_color="transparent")
        rate_row.pack(anchor="w", padx=18, fill="x", pady=(0, 12))
        self.rate_entry = ctk.CTkEntry(rate_row, width=120)
        self.rate_entry.insert(0, str(int(s["voice_rate"])))
        self.rate_entry.pack(side="left")
        self.test_btn = ctk.CTkButton(rate_row, text="🔊  Test voice", width=140,
                                      fg_color=ACCENT_DIM, hover_color=ACCENT,
                                      command=self._test_voice)
        self.test_btn.pack(side="right")

    def _grab(self):
        try:
            self.grab_set()
        except Exception:
            pass

    def _selected_voice(self) -> tuple[str, str]:
        """Dialog's current voice choice -> (voice_engine, piper_voice)."""
        label = self.voice_box.get()
        if label == VOICE_CLASSIC_LABEL:
            return "classic", config.PIPER_VOICE
        if label == VOICE_AUTO_LABEL:
            return "piper", "auto"
        return "piper", label

    def _test_voice(self):
        """Speak a sample line through the engine/voice currently selected in
        the dialog (unsaved values), muting the mic via the app's callbacks."""
        # One voice at a time: sounddevice shares one default output stream,
        # so a second play would cut off whatever is currently speaking.
        prev = getattr(self, "_test_mouth", None)
        if (prev is not None and prev.is_speaking()) or self.app.mouth.is_speaking():
            self.app.set_status("Already speaking — try again in a moment.")
            return
        voice_engine, piper_voice = self._selected_voice()
        m = construct_compat(Mouth, rate=self._selected_rate(),
                             engine=voice_engine, piper_voice=piper_voice,
                             on_start=self.app._on_tts_start, on_done=self.app._on_tts_done)
        m.speak(TEST_LINE)
        m.close()  # thread exits after the sample line (Mouth._run fires
        #            on_done/unmute even with the sentinel queued early)
        self._test_mouth = m

    def _selected_rate(self) -> int:
        try:
            return max(80, int(self.rate_entry.get()))
        except ValueError:
            return int(config.VOICE_RATE)

    def _populate_mics(self, mics: list[tuple[int, str]], saved_index, saved_name: str):
        """Fill the mic dropdown: 'System default' + one entry per physical mic.
        Preselects the saved device (by name first — indexes shift)."""
        self._mic_map = {name: idx for idx, name in mics}
        values = [MIC_DEFAULT_LABEL] + [name for _, name in mics]
        self.mic_box.configure(values=values)
        selected = MIC_DEFAULT_LABEL
        if saved_index is not None:
            for idx, name in mics:
                if saved_name and (saved_name.lower() in name.lower()
                                   or name.lower() in saved_name.lower()):
                    selected = name
                    break
            else:
                for idx, name in mics:
                    if idx == saved_index:
                        selected = name
                        break
        self.mic_box.set(selected)

    def _rescan_mics(self):
        self.mic_rescan_btn.configure(state="disabled", text="Scanning…")

        def worker():
            mics = list_microphones()
            def apply():
                if not self.winfo_exists():
                    return
                current = self.mic_box.get()
                self._populate_mics(mics, self._mic_map.get(current), current)
                self.mic_rescan_btn.configure(state="normal", text="Rescan")
            self.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _rescan(self):
        self.rescan_btn.configure(state="disabled", text="Scanning…")

        def worker():
            found = list_cameras()
            def apply():
                if not self.winfo_exists():
                    return
                values = [str(i) for i in found] or [self.cam_box.get()]
                self.cam_box.configure(values=values)
                if self.cam_box.get() not in values:
                    self.cam_box.set(values[0])
                self.rescan_btn.configure(state="normal", text="Rescan")
            self.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _save(self):
        s = config.load_settings()
        s["engine"] = ENGINE_VALUES.get(self.engine_seg.get(), "local")
        s["local_model"] = self._local_map.get(self.local_box.get(), "auto")
        s["api_key"] = self.key_entry.get().strip()
        s["model"] = self.model_box.get().strip() or s["model"]
        s["wake_word"] = self.wake_entry.get().strip().lower() or s["wake_word"]
        mic_label = self.mic_box.get()
        if mic_label == MIC_DEFAULT_LABEL:
            s["mic_device"] = None
            s["mic_device_name"] = ""
        elif mic_label in self._mic_map:
            s["mic_device"] = self._mic_map[mic_label]
            s["mic_device_name"] = mic_label
        try:
            s["camera_index"] = int(self.cam_box.get())
        except ValueError:
            pass
        try:
            s["watch_interval"] = max(3.0, float(self.interval_entry.get()))
        except ValueError:
            pass
        try:
            s["voice_rate"] = max(80, int(self.rate_entry.get()))
        except ValueError:
            pass
        s["voice_engine"], s["piper_voice"] = self._selected_voice()
        try:
            config.save_settings(s)
        except OSError as e:
            self.app.log_error(f"Could not save settings: {e}")
        self.app.apply_settings()
        self.destroy()


# ----- Main app -----------------------------------------------------------
class TBSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("TBS — OFFLINE AI")
        self.geometry("1000x680")
        self.minsize(880, 600)
        self.configure(fg_color=BG)
        try:
            ico = resource_path("build/icon.ico")
            if ico.exists():
                self.iconbitmap(str(ico))
        except Exception:
            pass

        # -- core components --
        self.brain = self._make_brain()
        self.mouth = construct_compat(
            Mouth, rate=config.VOICE_RATE, engine=config.VOICE_ENGINE,
            piper_voice=config.PIPER_VOICE,
            on_start=self._on_tts_start, on_done=self._on_tts_done)
        self._thinking = False
        self._watch_stop: threading.Event | None = None
        self._brain_boot_gen = 0  # invalidates stale boot threads after settings change
        self._brain_swap_lock = threading.Lock()  # serializes brain swaps/shutdown

        self._build_ui()

        self.feed = CameraFeed(config.CAMERA_INDEX, on_status=self.set_status_threadsafe)
        self.feed.start()
        self._blank_preview = True

        # Ears arm independently of (and concurrently with) the brain boot —
        # the wake word works even while the local model is still loading.
        self.ears: Ears | None = None
        self._ears_mic = (config.MIC_DEVICE, config.MIC_DEVICE_NAME)
        threading.Thread(target=self._start_ears, daemon=True, name="ears-init").start()
        threading.Thread(target=self._boot_brain, args=(self._brain_boot_gen,),
                         daemon=True, name="brain-boot").start()

        self._app_state = "OFFLINE"
        self.after(60, self._update_preview)
        self.after(200, self._refresh_state)
        self.after(1000, self._poll_engine_label)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.log_line("TBS online. Say the wake word, use Push to talk, or type below.",
                      tag="system")
        if BRAIN_IMPORT_ERROR is not None:
            self.log_line(f"Brain module failed to load ({BRAIN_IMPORT_ERROR}) — "
                          "voice loop and transcript still work.", tag="error")
        elif config.ENGINE == "claude" and not getattr(self.brain, "has_key", False):
            self.log_line("Engine is set to Claude only but no API key is set — "
                          "add one in Settings (⚙) or switch to the local engine.",
                          tag="system")

    def _make_brain(self):
        """Build the brain facade; tolerant of both the legacy Claude-only
        signature and the new facade (extra kwargs are filtered to whatever
        Brain.__init__ actually accepts)."""
        if Brain is None:
            return None
        try:
            return construct_compat(
                Brain,
                api_key=config.ANTHROPIC_API_KEY, model=config.CLAUDE_MODEL,
                max_tokens=config.MAX_TOKENS, history_turns=config.HISTORY_TURNS,
                engine=config.ENGINE, local_model=config.LOCAL_MODEL,
                prewarm=False)  # _boot_brain owns the boot + status display
        except Exception as e:
            self.after(0, lambda: self.log_line(f"Brain failed to initialize: {e}",
                                                tag="error"))
            return None

    # -- brain startup (local server boot happens off the UI thread) --
    def _boot_brain(self, gen: int):
        """If the engine RESOLVED to local, boot the llama server in the
        background so the UI never freezes (ears arm independently). Uses the
        facade's readiness hook — brain._ensure_local() blocks until the
        server is healthy and returns None, or an error string. The RLock in
        LocalServer makes this safe alongside the facade's own prewarm thread."""
        brain = self.brain
        if brain is None:
            return
        # Only boot when the facade actually resolved to the local engine
        # (auto may have picked Claude — don't spin up llama for nothing).
        if not self._engine_label().startswith("LOCAL"):
            self.after(0, self._refresh_engine_label)
            return
        hook = None
        for name in ("ensure_ready", "warm_up", "start_local", "_ensure_local"):
            fn = getattr(brain, name, None)
            if callable(fn):
                hook = fn
                break
        if hook is not None:
            self.set_status_threadsafe(LOADING_LOCAL_MSG)
            try:
                err = hook()  # facade returns None when ready, else a message
                if isinstance(err, str) and err:
                    self.log_threadsafe(err, tag="error")
            except Exception as e:
                self.log_threadsafe(f"Local brain failed to start: {e}", tag="error")
        if gen == self._brain_boot_gen:
            self.set_status_threadsafe("Ready")
        self.after(0, self._refresh_engine_label)

    def _engine_label(self) -> str:
        """Which brain is live, for the status bar."""
        if self.brain is None:
            return "BRAIN OFFLINE"
        fn = getattr(self.brain, "engine_label", None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                pass
        # Legacy Claude-only brain: derive a label ourselves.
        if getattr(self.brain, "has_key", False):
            return f"CLAUDE — {config.CLAUDE_MODEL}"
        return "NO BRAIN — set a local model or API key"

    def _refresh_engine_label(self):
        try:
            self.engine_label.configure(text=self._engine_label())
        except Exception:
            pass

    def _poll_engine_label(self):
        self._refresh_engine_label()
        self.after(2000, self._poll_engine_label)

    # -- ears startup (blocking model load happens off the UI thread) --
    def _start_ears(self):
        # Remember which mic setting this attempt used, so apply_settings can
        # tell whether a restart is needed. Set even on early-return paths.
        self._ears_mic = (config.MIC_DEVICE, config.MIC_DEVICE_NAME)
        model_path = config.vosk_model_path()
        if model_path is None:
            self.log_threadsafe("Speech model not found (models\\vosk\\"
                                + config.MODEL_DIR_NAME
                                + ") — voice input disabled.", tag="error")
            return
        # Saved index may be stale (USB re-plug shifts indexes) — the saved
        # name wins; unknown name falls back to system default with a log line.
        device = resolve_mic_device(
            config.MIC_DEVICE, config.MIC_DEVICE_NAME,
            log=lambda m: self.log_threadsafe(m, tag="system"))
        try:
            ears = Ears(
                model_path=str(model_path),
                wake_word=config.WAKE_WORD,
                silence_timeout=config.SILENCE_TIMEOUT,
                command_max_seconds=config.COMMAND_MAX_SECONDS,
                device=device,
                on_wake=self._on_wake,
                on_partial=self._on_partial,
                on_command=self._on_command,
                on_error=lambda m: self.log_threadsafe(m, tag="error"),
            )
        except Exception as e:
            self.log_threadsafe(f"Speech engine failed to load: {e}", tag="error")
            return
        if ears.start():
            self.ears = ears
            if self.mouth.is_speaking():
                ears.mute()
            self.log_threadsafe(f'Listening — say "{config.WAKE_WORD}" to talk.', tag="system")
        else:
            self.log_threadsafe("No microphone detected — use Push to talk after plugging one in "
                                "and restarting, or type below.", tag="error")

    # -- UI construction --
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left: preview + status ring
        left = ctk.CTkFrame(self, fg_color=PANEL_2, corner_radius=12)
        left.grid(row=0, column=0, sticky="nsw", padx=(14, 7), pady=(14, 4))

        ctk.CTkLabel(left, text="LIVE", text_color=ACCENT,
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=14, pady=(10, 4))
        self.preview = ctk.CTkLabel(left, text="Waiting for camera…", text_color=TEXT_DIM,
                                    width=PREVIEW_W, height=PREVIEW_H, fg_color="#060a16",
                                    corner_radius=8)
        self.preview.pack(padx=14, pady=(0, 10))

        ring_row = ctk.CTkFrame(left, fg_color="transparent")
        ring_row.pack(fill="x", padx=14, pady=(4, 6))
        self.ring = StatusRing(ring_row)
        self.ring.pack(side="left")
        state_col = ctk.CTkFrame(ring_row, fg_color="transparent")
        state_col.pack(side="left", padx=(14, 0), fill="x", expand=True)
        self.state_label = ctk.CTkLabel(state_col, text="OFFLINE", text_color=OFFLINE_GRAY,
                                        font=ctk.CTkFont(size=22, weight="bold"))
        self.state_label.pack(anchor="w")
        self.partial_label = ctk.CTkLabel(state_col, text="", text_color=TEXT_DIM,
                                          wraplength=330, justify="left",
                                          font=ctk.CTkFont(size=12))
        self.partial_label.pack(anchor="w", pady=(2, 0))

        self.ptt_btn = ctk.CTkButton(left, text="🎙  PUSH TO TALK", height=52,
                                     fg_color=ACCENT_DIM, hover_color=ACCENT,
                                     font=ctk.CTkFont(size=16, weight="bold"),
                                     command=self._push_to_talk)
        self.ptt_btn.pack(fill="x", padx=14, pady=(4, 14))

        # Right: conversation + controls
        right = ctk.CTkFrame(self, fg_color=PANEL_2, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 14), pady=(14, 4))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # toggles bar
        bar = ctk.CTkFrame(right, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 0))

        self.wake_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(bar, text="Wake word", variable=self.wake_var,
                      progress_color=ACCENT_DIM, command=self._toggle_wake).pack(side="left", padx=(0, 12))

        self.watch_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(bar, text="Watch mode", variable=self.watch_var,
                      progress_color=ACCENT_DIM, command=self._toggle_watch).pack(side="left")
        ctk.CTkLabel(bar, text="every", text_color=TEXT_DIM).pack(side="left", padx=(6, 4))
        self.interval_entry = ctk.CTkEntry(bar, width=44, justify="center")
        self.interval_entry.insert(0, str(int(config.WATCH_INTERVAL)))
        self.interval_entry.pack(side="left")
        ctk.CTkLabel(bar, text="s", text_color=TEXT_DIM).pack(side="left", padx=(3, 12))

        self.speak_var = ctk.BooleanVar(value=bool(config.SPEAK_RESPONSES))
        ctk.CTkSwitch(bar, text="Voice replies", variable=self.speak_var,
                      progress_color=ACCENT_DIM).pack(side="left")

        ctk.CTkButton(bar, text="⚙", width=40, fg_color="#334155", hover_color="#475569",
                      font=ctk.CTkFont(size=16), command=self.open_settings).pack(side="right")

        # conversation log
        self.log = ctk.CTkTextbox(right, fg_color="#060a16", text_color="#dbe4ff",
                                  wrap="word", corner_radius=8, font=ctk.CTkFont(size=13))
        self.log.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(10, 6))
        tb = self.log._textbox  # underlying tkinter.Text for color tags
        tb.tag_configure("you", foreground=ACCENT)
        tb.tag_configure("tbs", foreground="#dbe4ff")
        tb.tag_configure("notable", foreground=AMBER)
        tb.tag_configure("error", foreground="#f87171")
        tb.tag_configure("system", foreground=TEXT_DIM)
        tb.tag_configure("stamp", foreground="#475569")
        self.log.configure(state="disabled")

        # typed input (works without a mic)
        self.entry = ctk.CTkEntry(right, placeholder_text="Type a command…  (Enter to send)",
                                  height=38)
        self.entry.grid(row=2, column=0, sticky="ew", padx=(12, 6), pady=(0, 12))
        self.entry.bind("<Return>", lambda e: self._send_typed())
        ctk.CTkButton(right, text="Send", width=80, height=38,
                      fg_color=ACCENT_DIM, hover_color=ACCENT,
                      font=ctk.CTkFont(weight="bold"),
                      command=self._send_typed).grid(row=2, column=1, sticky="e",
                                                     padx=(0, 12), pady=(0, 12))

        # status bar: transient status on the left, live engine on the right
        status = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=28)
        status.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_label = ctk.CTkLabel(status, text="Starting…", text_color=TEXT_DIM,
                                         font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=14)
        self.engine_label = ctk.CTkLabel(status, text=self._engine_label(),
                                         text_color=ACCENT,
                                         font=ctk.CTkFont(size=12, weight="bold"))
        self.engine_label.pack(side="right", padx=14)

    # -- Logging / status --
    def log_line(self, text: str, tag: str = "tbs", sender: str | None = None):
        self.log.configure(state="normal")
        tb = self.log._textbox
        tb.insert("end", f"[{ts()}] ", ("stamp",))
        if sender:
            tb.insert("end", f"{sender}: ", (tag,))
            tb.insert("end", text + "\n\n", ("tbs",) if tag == "you" else (tag,))
        else:
            tb.insert("end", text + "\n\n", (tag,))
        self.log.configure(state="disabled")
        self.log.see("end")

    def log_threadsafe(self, text: str, tag: str = "tbs", sender: str | None = None):
        self.after(0, lambda: self.log_line(text, tag, sender))

    def log_error(self, text: str):
        self.log_threadsafe(text, tag="error")
        self.set_status_threadsafe(text[:90])

    def set_status(self, text: str):
        self.status_label.configure(text=text)

    def set_status_threadsafe(self, text: str):
        self.after(0, lambda: self.set_status(text))

    # -- State ring (polled; priority: SPEAKING > THINKING > CAPTURING > LISTENING) --
    def _compute_state(self) -> str:
        if self.mouth.is_speaking():
            return "SPEAKING"
        if self._thinking:
            return "THINKING"
        if self.ears is not None and self.ears.running:
            if self.ears.state == STATE_CAPTURING:
                return "CAPTURING"
            return "LISTENING"
        return "OFFLINE"

    def _refresh_state(self):
        state = self._compute_state()
        if state != self._app_state:
            self._app_state = state
            self.ring.set_state(state)
            self.state_label.configure(text=state, text_color=STATE_COLORS[state])
            if state != "CAPTURING":
                self.partial_label.configure(text="")
        self.after(200, self._refresh_state)

    # -- TTS <-> ears coordination (mouth thread) --
    def _on_tts_start(self):
        if self.ears is not None:
            self.ears.mute()

    def _on_tts_done(self):
        if self.ears is not None:
            self.ears.unmute()

    # -- Ears callbacks (audio thread) --
    def _on_wake(self):
        wake_ack()
        self.set_status_threadsafe("Wake word heard — listening for command…")

    def _on_partial(self, text: str):
        self.after(0, lambda: self.partial_label.configure(text=text))

    def _on_command(self, command: str):
        self.after(0, lambda: self._handle_command(command))

    # -- Command handling (UI thread entry; heavy work in a worker thread) --
    def _handle_command(self, command: str):
        """Spoken or typed command -> log it, attach current frame, ask the
        brain, speak the reply. Fully guarded: a missing brain, missing API
        key, or engine errors never crash."""
        self.log_line(command, tag="you", sender="You")
        if self.brain is None:
            self.log_line("The brain module is unavailable — check the install.", tag="error")
            return
        if self._thinking:
            self.log_line("Still working on the last request, sir.", tag="system")
            return
        self._thinking = True
        self.set_status("Thinking…")
        threading.Thread(target=self._command_worker, args=(command,),
                         daemon=True, name="brain").start()

    def _command_worker(self, command: str):
        try:
            frame = self.feed.get_frame()
            b64 = None
            if frame is not None:
                try:
                    b64 = frame_to_jpeg_b64(frame, config.JPEG_QUALITY, config.MAX_IMAGE_DIM)
                except Exception:
                    b64 = None
            answer = self.brain.ask(command, b64)
        except NoAPIKeyError:
            answer = NO_KEY_MSG
        except RATE_LIMIT_ERRORS:
            answer = "The API is rate limiting me, sir. Give it a minute."
        except CONNECTION_ERRORS:
            answer = "I can't reach the API, sir. Check the connection."
        except API_ERRORS as e:
            answer = "The API returned an error, sir."
            self.log_threadsafe(f"API error: {e}", tag="error")
        except Exception as e:
            answer = "Something went wrong on my end, sir."
            self.log_threadsafe(f"Error: {e}", tag="error")
        finally:
            self._thinking = False
        self._say(answer)
        self.set_status_threadsafe("Ready")

    def _say(self, text: str):
        """Log + (optionally) speak — everything TBS says hits the log."""
        self.log_threadsafe(text, tag="tbs", sender="Buddy")
        if self.speak_var.get():
            self.mouth.speak(text)

    def _send_typed(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._handle_command(text)

    # -- Push to talk --
    def _push_to_talk(self):
        if self.ears is None or not self.ears.running:
            self.log_line("Voice input is offline — no microphone or speech model.", tag="error")
            return
        if self.mouth.is_speaking():
            self.log_line("Wait for me to finish speaking, sir.", tag="system")
            return
        self.ears.start_command_capture()
        self.set_status("Listening for your command…")

    # -- Toggles --
    def _toggle_wake(self):
        if self.ears is not None:
            self.ears.wake_enabled = bool(self.wake_var.get())
        self.set_status("Wake word " + ("armed" if self.wake_var.get() else "disabled"))

    def _get_interval(self) -> float:
        try:
            return max(3.0, float(self.interval_entry.get()))
        except ValueError:
            return float(config.WATCH_INTERVAL)

    def _toggle_watch(self):
        if self.watch_var.get():
            if self.brain is None:
                self.watch_var.set(False)
                self.log_line("The brain module is unavailable — watch mode needs it.", tag="error")
                return
            # Only hard-block when Claude is the ONLY allowed engine and there
            # is no key; local/auto engines surface their own errors if the
            # local model turns out to be missing.
            if config.ENGINE == "claude" and not getattr(self.brain, "has_key", False):
                self.watch_var.set(False)
                self._say(NO_KEY_MSG)
                return
            self._watch_stop = threading.Event()
            threading.Thread(target=self._watch_loop, args=(self._watch_stop,),
                             daemon=True, name="watch").start()
            self.log_line(f"Watch mode ON — a frame every {self._get_interval():.0f}s "
                          "(or as fast as the model keeps up); I'll speak up if "
                          "something notable happens.", tag="system")
        else:
            if self._watch_stop is not None:
                self._watch_stop.set()
                self._watch_stop = None
            self.log_line("Watch mode OFF.", tag="system")
            self.set_status("Ready")

    def _watch_loop(self, stop: threading.Event):
        previous = None
        while not stop.is_set():
            started = time.time()
            frame = self.feed.get_frame()
            if frame is None:
                self.log_threadsafe("Watch: no camera frame available.", tag="error")
            else:
                try:
                    b64 = frame_to_jpeg_b64(frame, config.JPEG_QUALITY, config.MAX_IMAGE_DIM)
                    text = self.brain.narrate(b64, previous)
                    if stop.is_set():
                        break
                    if text.startswith(NOTABLE_TAG):
                        text = text[len(NOTABLE_TAG):].strip()
                        path = self._save_screenshot(frame, "notable")
                        self.log_threadsafe(text, tag="notable", sender="Buddy")
                        if path:
                            self.log_threadsafe(f"Saved screenshot: {path}", tag="system")
                        else:
                            self.log_threadsafe("Screenshot could not be saved "
                                                "(drive full or write-locked?).",
                                                tag="error")
                        if self.speak_var.get():
                            self.mouth.speak(text)  # speaks up proactively
                    else:
                        self.log_threadsafe(text, tag="system", sender="watch")
                    previous = text
                except NoAPIKeyError:
                    self.after(0, lambda: self.watch_var.set(False))
                    self._say(NO_KEY_MSG)
                    break
                except RATE_LIMIT_ERRORS:
                    self.log_threadsafe("Watch: rate limited — waiting 60s.", tag="error")
                    if stop.wait(60):
                        break
                    continue
                except API_ERRORS as e:
                    self.log_threadsafe(f"Watch: API error: {e}", tag="error")
                except Exception as e:
                    self.log_threadsafe(f"Watch: error: {e}", tag="error")
            remaining = self._get_interval() - (time.time() - started)
            if stop.wait(max(0.5, remaining)):
                break

    def _save_screenshot(self, frame, reason: str) -> str | None:
        """Save a frame to <root>\\screenshots. Returns the path, or None if
        the write failed (drive full/write-locked) — never raises."""
        try:
            portable.ensure(config.SCREENSHOT_DIR)
            name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{reason}.jpg"
            path = config.SCREENSHOT_DIR / name
            if save_frame(frame, path):
                return str(path)
        except OSError:
            pass
        return None

    # -- Preview loop --
    def _update_preview(self):
        frame = self.feed.get_frame()
        if frame is not None:
            import cv2
            h, w = frame.shape[:2]
            scale = min(PREVIEW_W / w, PREVIEW_H / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ctk.CTkImage(light_image=Image.fromarray(rgb),
                               size=(frame.shape[1], frame.shape[0]))
            self.preview.configure(image=img, text="")
            self._blank_preview = False
        elif not self._blank_preview:
            self.preview.configure(image=None, text="No camera signal", text_color=TEXT_DIM)
            self._blank_preview = True
        self.after(66, self._update_preview)  # ~15 fps

    # -- Settings --
    def open_settings(self):
        for w in self.winfo_children():
            if isinstance(w, SettingsDialog) and w.winfo_exists():
                w.lift()
                w.focus_force()
                return
        SettingsDialog(self)

    def apply_settings(self):
        """Called after the Settings dialog saves. config globals are updated.

        The whole brain swap runs OFF the UI thread: shutting down the old
        brain blocks on the local server lock for as long as a model load is
        in flight (up to 180s — likeliest right after launch, exactly when
        users open Settings), and building the new brain may probe the
        network (auto engine). Same reasoning as _on_close's finish()."""
        self._brain_boot_gen += 1
        gen = self._brain_boot_gen

        def swap():
            with self._brain_swap_lock:  # serialize rapid re-saves
                new_brain = self._make_brain()
                old_brain, self.brain = self.brain, new_brain
                self._shutdown_brain(old_brain)  # may block on model load
            self._boot_brain(gen)
        threading.Thread(target=swap, daemon=True, name="brain-reload").start()
        self.mouth.rate = config.VOICE_RATE
        self.mouth.engine = config.VOICE_ENGINE
        if hasattr(self.mouth, "piper_voice"):
            self.mouth.piper_voice = config.PIPER_VOICE
        self.interval_entry.delete(0, "end")
        self.interval_entry.insert(0, str(int(config.WATCH_INTERVAL)))
        if self.ears is not None:
            self.ears.wake_word = config.WAKE_WORD.strip().lower()
        if (config.MIC_DEVICE, config.MIC_DEVICE_NAME) != self._ears_mic:
            self._restart_ears()
        if config.CAMERA_INDEX != self.feed.index:
            self._switch_camera(config.CAMERA_INDEX)
        self._refresh_engine_label()
        self.log_line("Settings saved.", tag="system")

    @staticmethod
    def _shutdown_brain(brain) -> None:
        """Ask a brain facade to release its resources (local llama server).
        Tolerates the legacy brain, which has no shutdown()."""
        fn = getattr(brain, "shutdown", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    def _restart_ears(self):
        """Restart the STT stream on the newly configured mic device — no app
        restart. If the new device fails, Ears.start() itself falls back to
        the system default (log line, never a crash)."""
        self.set_status("Switching microphone…")
        old = self.ears
        self.ears = None  # state ring shows OFFLINE during the swap

        def worker():
            if old is not None:
                old.stop()
            self._start_ears()
        threading.Thread(target=worker, daemon=True, name="mic-switch").start()

    def _switch_camera(self, new_index: int):
        """Swap the running capture thread to a new camera: probe the new
        device first; on failure log and KEEP the old camera running."""
        self.set_status(f"Switching to camera {new_index}…")

        def worker():
            try:
                Camera(new_index).release()  # probe: can it open at all?
            except CameraError as e:
                self.log_threadsafe(
                    f"Camera {new_index} failed to open — keeping camera "
                    f"{self.feed.index}. ({e})", tag="error")
                return
            old = self.feed
            new = CameraFeed(new_index, on_status=self.set_status_threadsafe)
            old.stop()          # release the old device before opening the new
            new.start()
            self.feed = new     # preview loop picks this up next tick
        threading.Thread(target=worker, daemon=True, name="camera-switch").start()

    # -- Shutdown --
    def _on_close(self):
        if getattr(self, "_closing", False):
            return
        self._closing = True
        if self._watch_stop is not None:
            self._watch_stop.set()
        if self.ears is not None:
            self.ears.stop()
        self.mouth.shutdown()
        self.feed.stop()
        self.withdraw()  # window disappears immediately

        def finish():
            # May block briefly if the local model is still mid-load (stop()
            # waits on the server lock) — done off the UI thread so the app
            # never looks hung, and the llama server is never orphaned. The
            # swap lock serializes with an in-flight apply_settings swap.
            with self._brain_swap_lock:
                self._shutdown_brain(self.brain)
            self.after(0, self.destroy)
        threading.Thread(target=finish, daemon=True, name="shutdown").start()


def main():
    app = TBSApp()
    app.mainloop()


if __name__ == "__main__":
    main()
