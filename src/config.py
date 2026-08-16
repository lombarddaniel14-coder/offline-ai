"""Configuration for TBS — Offline AI. Fully portable.

Settings persist INSIDE the app folder at <root>\\config\\settings.json —
never %APPDATA%, never the registry — so the whole folder can be copied to a
flash drive and run on any PC with zero installation.

Precedence: settings.json (written by the GUI's Settings dialog) > defaults
below. Fresh app: nothing is migrated from the old TBS Eyes config.
"""

import json
import os

import portable

# --- Brain / engine ---
ENGINE = "local"           # 'auto' | 'local' | 'claude'
LOCAL_MODEL = "auto"       # 'auto' or a .gguf filename from models\llm\MODELS.json
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # optional — app runs fine empty
CLAUDE_MODEL = "claude-opus-5"
MAX_TOKENS = 1024
HISTORY_TURNS = 12  # rolling exchanges kept in conversation history

# --- Camera ---
CAMERA_INDEX = 0
WATCH_INTERVAL = 20.0  # seconds between watch-mode frames

# --- Image encoding ---
JPEG_QUALITY = 85
MAX_IMAGE_DIM = 1024

# --- Ears (speech-to-text) ---
WAKE_WORD = "buddy"
# Microphone: sounddevice input device index; None = system default. Indexes
# shift when USB devices are re-plugged, so the device NAME is stored too and
# is the source of truth (see ears.resolve_mic_device).
MIC_DEVICE = None
MIC_DEVICE_NAME = ""
SILENCE_TIMEOUT = 1.5        # seconds of silence that ends a command
COMMAND_MAX_SECONDS = 15.0   # hard cap on one command's length

# --- Mouth (TTS) ---
SPEAK_RESPONSES = True
VOICE_ENGINE = "piper"   # 'piper' (offline neural, default) | 'classic' (SAPI)
PIPER_VOICE = "auto"     # 'auto' or an .onnx voice name from models\piper
VOICE_RATE = 180         # classic/SAPI speed, words per minute

# --- Paths (all inside the app root — portable) ---
SCREENSHOT_DIR = portable.SHOTS
MODEL_DIR_NAME = "vosk-model-small-en-us-0.15"
CONFIG_JSON = portable.CONFIG / "settings.json"


def vosk_model_path():
    """Find the Vosk STT model folder under <root>\\models\\vosk. None if missing."""
    exact = portable.MODELS / "vosk" / MODEL_DIR_NAME
    if exact.is_dir():
        return exact
    try:  # any other vosk model dropped in works too
        for p in sorted((portable.MODELS / "vosk").glob("vosk-model-*")):
            if p.is_dir():
                return p
    except OSError:
        pass
    return None


def _read_json() -> dict:
    """Load settings.json, returning {} on any problem (missing, corrupt).

    ValueError covers both JSONDecodeError AND UnicodeDecodeError — a
    settings.json with invalid UTF-8 bytes (flash drive yanked mid-write,
    ANSI-editor edit) must fall back to defaults, never kill the app at
    import time."""
    try:
        with open(CONFIG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_settings() -> dict:
    """Return the fully resolved settings dict (settings.json > defaults)."""
    return {
        "engine": ENGINE,
        "local_model": LOCAL_MODEL,
        "api_key": ANTHROPIC_API_KEY,
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "camera_index": CAMERA_INDEX,
        "watch_interval": WATCH_INTERVAL,
        "jpeg_quality": JPEG_QUALITY,
        "max_image_dim": MAX_IMAGE_DIM,
        "wake_word": WAKE_WORD,
        "mic_device": MIC_DEVICE,
        "mic_device_name": MIC_DEVICE_NAME,
        "speak_responses": SPEAK_RESPONSES,
        "voice_engine": VOICE_ENGINE,
        "piper_voice": PIPER_VOICE,
        "voice_rate": VOICE_RATE,
    }


def save_settings(settings: dict) -> None:
    """Persist settings to <root>\\config\\settings.json and apply them to
    this module's globals so the running process sees them immediately.

    Atomic: written to a temp file then os.replace()d, so a flash drive
    yanked mid-save can never leave a half-written settings.json behind."""
    portable.ensure(portable.CONFIG)
    tmp = CONFIG_JSON.with_name(CONFIG_JSON.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CONFIG_JSON)
    _apply(settings)


def _apply(overrides: dict) -> None:
    """Overlay a dict of settings onto this module's globals (bad values ignored)."""
    g = globals()
    casts = {
        "engine": ("ENGINE", str),
        "local_model": ("LOCAL_MODEL", str),
        "api_key": ("ANTHROPIC_API_KEY", str),
        "model": ("CLAUDE_MODEL", str),
        "max_tokens": ("MAX_TOKENS", int),
        "camera_index": ("CAMERA_INDEX", int),
        "watch_interval": ("WATCH_INTERVAL", float),
        "jpeg_quality": ("JPEG_QUALITY", int),
        "max_image_dim": ("MAX_IMAGE_DIM", int),
        "wake_word": ("WAKE_WORD", str),
        "speak_responses": ("SPEAK_RESPONSES", bool),
        "voice_engine": ("VOICE_ENGINE", str),
        "piper_voice": ("PIPER_VOICE", str),
        "voice_rate": ("VOICE_RATE", int),
    }
    # mic_device is special: None is a VALID value (system default), so it's
    # handled outside the casts loop below (which skips None). settings.json
    # files without a key simply leave the default in place.
    if "mic_device" in overrides:
        value = overrides["mic_device"]
        if value is None:
            g["MIC_DEVICE"] = None
        else:
            try:
                g["MIC_DEVICE"] = int(value)
            except (TypeError, ValueError):
                pass
    if "mic_device_name" in overrides and isinstance(overrides["mic_device_name"], str):
        g["MIC_DEVICE_NAME"] = overrides["mic_device_name"]
    for key, (name, cast) in casts.items():
        if key in overrides and overrides[key] is not None:
            try:
                value = cast(overrides[key])
            except (TypeError, ValueError):
                continue
            if key == "wake_word":
                value = value.strip().lower() or g[name]
            if key == "engine" and value not in ("auto", "local", "claude"):
                continue
            # 'classic' and 'sapi' both mean the SAPI fallback (mouth treats
            # any non-'piper' engine as SAPI).
            if key == "voice_engine" and value not in ("piper", "classic", "sapi"):
                continue
            g[name] = value


# Overlay settings.json on top of the defaults at import time.
_apply(_read_json())
