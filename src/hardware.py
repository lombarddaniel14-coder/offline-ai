"""Hardware-aware local model selection.

recommend_model() scans <root>/models/llm/*.gguf and picks the best
model+mmproj pair this machine can run: total RAM >= 7 GB prefers the
gemma-3-4b (quality) pair, otherwise the LFM2-VL (light) pair. Falls back
to whichever complete pair exists; returns None cleanly if none do.

Pair definitions come from models/llm/MODELS.json (written by the download
step) when present, with a filename heuristic as fallback — so a user can
drop in the files without the json and it still works.
"""

import ctypes
import json
import os
from pathlib import Path

from portable import app_root

RAM_THRESHOLD_GB = 7.0

_TIER_LABELS = {"light": "LFM2-VL 1.6B", "quality": "Gemma 3 4B"}


def total_ram_gb() -> float:
    """Total physical RAM in GiB (0.0 if it can't be determined)."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
        except OSError:
            pass
        return 0.0
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return 0.0


def _pairs_from_json(llm_dir: Path) -> dict[str, dict]:
    """tier -> {model_path, mmproj_path, label} for pairs whose BOTH files exist."""
    manifest = llm_dir / "MODELS.json"
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    pairs: dict[str, dict] = {}
    if not isinstance(entries, list):
        return {}
    for e in entries:
        try:
            tier = e["tier"]
            model = llm_dir / e["model"]
            mmproj = llm_dir / e["mmproj"]
        except (TypeError, KeyError):
            continue
        if model.is_file() and mmproj.is_file():
            pairs[tier] = {
                "model_path": str(model),
                "mmproj_path": str(mmproj),
                "label": _TIER_LABELS.get(tier, model.stem),
            }
    return pairs


def _pairs_from_scan(llm_dir: Path) -> dict[str, dict]:
    """Filename heuristic when MODELS.json is missing."""
    ggufs = sorted(llm_dir.glob("*.gguf"))
    mmprojs = [f for f in ggufs if "mmproj" in f.name.lower()]
    models = [f for f in ggufs if f not in mmprojs]
    pairs: dict[str, dict] = {}
    for m in models:
        name = m.name.lower()
        if "lfm2" in name:
            tier = "light"
            mp = next((p for p in mmprojs if "lfm2" in p.name.lower()), None)
        elif "gemma" in name:
            tier = "quality"
            mp = (next((p for p in mmprojs if "gemma" in p.name.lower()), None)
                  or next((p for p in mmprojs if "lfm2" not in p.name.lower()), None))
        else:
            continue
        if mp is not None and tier not in pairs:
            pairs[tier] = {
                "model_path": str(m),
                "mmproj_path": str(mp),
                "label": _TIER_LABELS[tier],
            }
    return pairs


def available_pairs() -> dict[str, dict]:
    """tier -> pair dict for every complete model+mmproj pair on disk."""
    llm_dir = app_root() / "models" / "llm"
    if not llm_dir.is_dir():
        return {}
    return _pairs_from_json(llm_dir) or _pairs_from_scan(llm_dir)


def pair_for_model(filename: str) -> dict | None:
    """The pair whose model .gguf filename matches (Settings' explicit pick),
    or None if that file isn't on disk with its mmproj."""
    if not filename or filename == "auto":
        return None
    for pair in available_pairs().values():
        if Path(pair["model_path"]).name == filename:
            return pair
    return None


def recommend_model() -> dict | None:
    """Best available pair for this machine, or None if no complete pair.

    Returns {"model_path": str, "mmproj_path": str, "label": str}.
    """
    pairs = available_pairs()
    if not pairs:
        return None
    preferred = "quality" if total_ram_gb() >= RAM_THRESHOLD_GB else "light"
    fallback = "light" if preferred == "quality" else "quality"
    return pairs.get(preferred) or pairs.get(fallback)
