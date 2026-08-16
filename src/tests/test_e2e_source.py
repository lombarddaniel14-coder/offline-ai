"""FULL SOURCE E2E — the whole offline loop in ONE process, no API key.

Launches the real TBSApp (engine forced to 'local'), waits for the local
llama-server to become healthy, waits for a real camera frame, then injects a
synthetic command through app._on_command() — the exact callback the wake-word
pipeline (Ears.on_command) fires — and asserts:

  1. a non-empty local-brain answer (not an error string) lands in the log
  2. the answer was queued to TTS (mouth.speak called with it)

Run:  _dev\\venv\\Scripts\\python.exe src\\tests\\test_e2e_source.py
"""

import sys
import threading
import time
import urllib.request
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import config  # noqa: E402

config.ENGINE = "local"          # force the offline path, whatever settings say
config.SPEAK_RESPONSES = True

import brain as brain_mod        # noqa: E402
import local_brain as lb         # noqa: E402
from tbs_app import TBSApp # noqa: E402

COMMAND = "What do you see in front of the camera right now?"
HEALTH_URL = "http://127.0.0.1:8090/health"
HEALTH_TIMEOUT = 300   # first CPU load of the quality model can take minutes
ANSWER_TIMEOUT = 180
ERROR_ANSWERS = {
    lb.SERVER_DOWN_MSG, lb.TIMEOUT_MSG, lb.MALFORMED_MSG,
    brain_mod.NO_KEY_MSG, brain_mod.NO_LOCAL_MODEL_MSG,
    brain_mod.LOCAL_START_FAIL_MSG,
    "Something went wrong on my end, sir.",
}

results = {"health": False, "frame": False, "spoken": [], "log": "", "fail": ""}


def health_ok() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def driver(app: TBSApp):
    try:
        # 1. local server healthy (the GUI's own _boot_brain starts it)
        deadline = time.time() + HEALTH_TIMEOUT
        while time.time() < deadline and not health_ok():
            time.sleep(2)
        results["health"] = health_ok()
        if not results["health"]:
            results["fail"] = f"llama-server not healthy within {HEALTH_TIMEOUT}s"
            return

        # 2. real camera frame available
        deadline = time.time() + 30
        while time.time() < deadline and app.feed.get_frame() is None:
            time.sleep(1)
        results["frame"] = app.feed.get_frame() is not None
        if not results["frame"]:
            results["fail"] = "no camera frame within 30s (camera 0 missing?)"
            return

        # 3. record what gets queued to TTS, then inject the command through
        #    the SAME handler the wake word fires (Ears.on_command -> _on_command)
        real_speak = app.mouth.speak

        def spy_speak(text):
            results["spoken"].append(text)
            real_speak(text)
        app.mouth.speak = spy_speak

        app._on_command(COMMAND)  # exactly what ears does, same thread pattern

        # 4. wait for the brain's answer to be queued to TTS
        deadline = time.time() + ANSWER_TIMEOUT
        while time.time() < deadline and not results["spoken"]:
            time.sleep(1)
        if not results["spoken"]:
            results["fail"] = f"no TTS queued within {ANSWER_TIMEOUT}s"
    finally:
        def snap_and_close():
            try:
                results["log"] = app.log._textbox.get("1.0", "end")
            except Exception as e:
                results["log"] = f"(log read failed: {e})"
            app._on_close()
        app.after(0, snap_and_close)


def main() -> int:
    app = TBSApp()
    threading.Thread(target=driver, args=(app,), daemon=True, name="e2e").start()
    # Watchdog: never hang the harness.
    threading.Thread(target=lambda: (time.sleep(HEALTH_TIMEOUT + ANSWER_TIMEOUT + 90),
                                     app.after(0, app._on_close)),
                     daemon=True, name="watchdog").start()
    app.mainloop()

    print(f"health_ok:  {results['health']}")
    print(f"frame_ok:   {results['frame']}")
    print(f"tts_queued: {len(results['spoken'])} utterance(s)")
    if results["fail"]:
        print("FAIL:", results["fail"])
        return 1

    answer = results["spoken"][0]
    print(f"ANSWER: {answer}")
    ok = True
    if not answer.strip():
        print("FAIL: empty answer");  ok = False
    if answer in ERROR_ANSWERS:
        print("FAIL: error string, not a real answer");  ok = False
    if f"You: {COMMAND}" not in results["log"]:
        print("FAIL: command never hit the log");  ok = False
    if answer.split("!")[0][:40] not in results["log"]:
        print("FAIL: answer not found in the log");  ok = False
    print("E2E " + ("PASSED — full offline loop verified in one process"
                    if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
