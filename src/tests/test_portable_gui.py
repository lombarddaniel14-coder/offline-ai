"""Portability + GUI smoke tests for TBS — Offline AI.

Run:  _dev\\venv\\Scripts\\python.exe src\\tests\\test_portable_gui.py

1. portable.app_root() resolves to the "Offline AI" folder from source.
2. Config round-trip: save_settings persists to <root>\\config\\settings.json,
   values survive a fresh module load, and NOTHING is written outside the
   app folder (%APPDATA% stays untouched).
3. GUI smoke: tbs_app.py launches from the venv, stays alive 8 seconds,
   and prints nothing to stderr (brain module may be mid-migration — the
   import guard must keep the window up regardless).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # ...\Offline AI
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import portable  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ---- 1. portable path resolution ------------------------------------------
def test_portable():
    print("[1] portable.app_root()")
    check("app_root == Offline AI folder", portable.app_root() == ROOT,
          str(portable.app_root()))
    check("ENGINE under root", portable.ENGINE == ROOT / "engine")
    check("MODELS under root", portable.MODELS == ROOT / "models")
    check("PIPER under root", portable.PIPER == ROOT / "piper")
    check("CONFIG under root", portable.CONFIG == ROOT / "config")
    check("SHOTS under root", portable.SHOTS == ROOT / "screenshots")
    check("no absolute app path baked in",
          not str(portable.app_root()).lower().startswith("c:\\windows"))


# ---- 2. config round-trip ---------------------------------------------------
def test_config_roundtrip():
    print("[2] config round-trip (settings.json inside the folder only)")
    import config

    cfg_file = ROOT / "config" / "settings.json"
    check("CONFIG_JSON path is inside <root>\\config",
          config.CONFIG_JSON == cfg_file, str(config.CONFIG_JSON))

    backup = cfg_file.read_bytes() if cfg_file.exists() else None

    appdata = os.environ.get("APPDATA", "")
    legacy_dir = Path(appdata) / "TBSEyes" if appdata else None
    legacy_before = legacy_dir.exists() if legacy_dir else False
    appdata_before = set(os.listdir(appdata)) if appdata else set()

    try:
        s = config.load_settings()
        s["engine"] = "auto"
        s["local_model"] = "gemma-3-4b-it-Q4_K_M.gguf"
        s["voice_engine"] = "piper"
        s["piper_voice"] = "en_GB-alan-medium"
        s["wake_word"] = "buddy"
        s["watch_interval"] = 25.0
        config.save_settings(s)

        check("settings.json written", cfg_file.is_file())
        on_disk = json.loads(cfg_file.read_text(encoding="utf-8"))
        check("values persisted", on_disk["engine"] == "auto"
              and on_disk["local_model"] == "gemma-3-4b-it-Q4_K_M.gguf"
              and on_disk["piper_voice"] == "en_GB-alan-medium"
              and on_disk["watch_interval"] == 25.0)
        check("applied to running module", config.ENGINE == "auto"
              and config.LOCAL_MODEL == "gemma-3-4b-it-Q4_K_M.gguf")

        # Fresh interpreter sees the saved values (true persistence).
        code = ("import sys; sys.path.insert(0, r'%s'); import config; "
                "print(config.ENGINE, config.LOCAL_MODEL, config.PIPER_VOICE)" % SRC)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=60)
        check("fresh process reloads saved values",
              out.stdout.split() == ["auto", "gemma-3-4b-it-Q4_K_M.gguf",
                                     "en_GB-alan-medium"],
              out.stdout.strip() or out.stderr.strip()[:120])

        if legacy_dir:
            check("no %APPDATA%\\TBSEyes created",
                  legacy_dir.exists() == legacy_before)
            check("nothing new in %APPDATA%",
                  set(os.listdir(appdata)) == appdata_before)

        check("vosk model found under portable.MODELS",
              config.vosk_model_path() is not None
              and str(config.vosk_model_path()).startswith(str(portable.MODELS)),
              str(config.vosk_model_path()))
        check("screenshots dir is portable.SHOTS",
              config.SCREENSHOT_DIR == portable.SHOTS)
    finally:
        if backup is None:
            cfg_file.unlink(missing_ok=True)  # leave the app factory-fresh
        else:
            cfg_file.write_bytes(backup)


# ---- 3. GUI smoke -----------------------------------------------------------
def test_gui_smoke():
    print("[3] GUI smoke (launch, alive 8s, no stderr)")
    venv_py = ROOT / "_dev" / "venv" / "Scripts" / "python.exe"
    py = venv_py if venv_py.is_file() else Path(sys.executable)
    proc = subprocess.Popen([str(py), str(SRC / "tbs_app.py")],
                            cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    time.sleep(8)
    alive = proc.poll() is None
    check("process alive after 8s", alive,
          "" if alive else f"exit code {proc.returncode}")
    proc.kill()
    try:
        _, err = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        _, err = b"", b"(communicate timeout)"
    err_text = err.decode("utf-8", "replace").strip()
    check("no stderr output", err_text == "", err_text[:400])

    # The hard kill above bypasses the app's clean shutdown, which would
    # orphan a llama-server the brain facade booted during the 8s window.
    # Stop only servers running from THIS app's engine folder.
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | "
          "Where-Object { $_.ExecutablePath -like '%s*' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
          % str(ROOT / "engine").replace("'", "''"))
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, timeout=30)
    except Exception:
        pass


if __name__ == "__main__":
    test_portable()
    test_config_roundtrip()
    test_gui_smoke()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
