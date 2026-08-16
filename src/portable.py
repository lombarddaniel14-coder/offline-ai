"""Portable path resolution — the single source of truth for where the app lives.

The whole product ships as one folder ("Offline AI") that can sit on any drive
letter (flash drive, Desktop, wherever). NOTHING may use absolute paths,
%APPDATA%, or the registry — every module resolves paths through app_root().

Layout:
  <root>\\app\\TBSOffline\\TBSOffline.exe   (frozen / PyInstaller)
  <root>\\src\\*.py                                (running from source)
  <root>\\engine\\   llama-server.exe + DLLs
  <root>\\models\\   llm\\, piper\\, vosk\\
  <root>\\piper\\    piper.exe + espeak data
  <root>\\config\\   settings.json (the ONLY place state is persisted)
  <root>\\screenshots\\
"""

import sys
from pathlib import Path


def app_root() -> Path:
    """The "Offline AI" folder root.

    Frozen: exe lives at <root>\\app\\TBSOffline\\TBSOffline.exe, so the
    root is parents[2] (parents[0]=TBSOffline, parents[1]=app, parents[2]=root).
    Source: this file lives at <root>\\src\\portable.py, so root is parents[1].
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parents[2]
    return Path(__file__).resolve().parents[1]


# Well-known folders inside the root. Plain Path constants so every module can
# import them; call ensure() before writing into one (auto-create on demand).
ENGINE = app_root() / "engine"
MODELS = app_root() / "models"
PIPER = app_root() / "piper"
CONFIG = app_root() / "config"
SHOTS = app_root() / "screenshots"


def ensure(path: Path) -> Path:
    """mkdir -p the folder and return it (auto-create on demand)."""
    path.mkdir(parents=True, exist_ok=True)
    return path
