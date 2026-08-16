"""Real end-to-end test of the local brain (the heart of the project).

For each model tier (light, then quality):
  1. LocalServer.ensure_running() starts <root>/engine/llama-server.exe
  2. a PIL-generated 800x600 image with huge black text "TEST 42" is sent as a
     real vision request via LocalBrain.ask()
  3. the response must be non-empty and not one of the error strings
     (small models may imperfectly OCR — mentioning anything from the image
     counts; empty/error = fail)

Also exercises: model hot-swap (light -> quality on the same LocalServer),
server reuse, hardware.recommend_model(), and one full facade Brain.ask()
through the local path.

Run:  _dev\\venv\\Scripts\\python.exe src\\tests\\test_local_brain.py
"""

import base64
import importlib.util
import io
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]          # <root>/src
ROOT = SRC.parent                                  # <root>
sys.path.insert(0, str(SRC))

# portable.py is owned by the GUI agent; fall back to the dev stub until it lands.
try:
    import portable  # noqa: F401
except ImportError:
    _stub = ROOT / "_dev" / "portable_stub.py"
    _spec = importlib.util.spec_from_file_location("portable", _stub)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules["portable"] = _mod

import hardware                                    # noqa: E402
import local_brain as lb_mod                       # noqa: E402
from local_brain import LocalBrain                 # noqa: E402
from server_manager import LocalServer             # noqa: E402

PORT = 8090
LLM_DIR = ROOT / "models" / "llm"
TIERS = [  # (name, model file, mmproj file) — matches models/llm/MODELS.json
    ("light", "LFM2-VL-1.6B-Q4_0.gguf", "mmproj-LFM2-VL-1.6B-Q8_0.gguf"),
    ("quality", "gemma-3-4b-it-Q4_K_M.gguf", "mmproj-model-f16.gguf"),
]
ERROR_STRINGS = {lb_mod.SERVER_DOWN_MSG, lb_mod.TIMEOUT_MSG, lb_mod.MALFORMED_MSG}


def make_test_image_b64() -> str:
    """800x600, solid white background, huge black text 'TEST 42'."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 150)
    except OSError:
        font = ImageFont.load_default(size=150)
    text = "TEST 42"
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((800 - (box[2] - box[0])) / 2 - box[0],
               (600 - (box[3] - box[1])) / 2 - box[1]),
              text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def check_answer(resp: str, ctx: str) -> None:
    assert isinstance(resp, str) and resp.strip(), f"{ctx}: empty response"
    assert resp not in ERROR_STRINGS, f"{ctx}: error string returned: {resp}"
    assert not resp.startswith("The local model refused"), f"{ctx}: {resp}"


def run_tier(server: LocalServer, name: str, model: str, mmproj: str,
             b64: str) -> tuple[str, float, float]:
    model_p, mmproj_p = LLM_DIR / model, LLM_DIR / mmproj
    assert model_p.is_file(), f"missing {model_p}"
    assert mmproj_p.is_file(), f"missing {mmproj_p}"

    t0 = time.time()
    ok = server.ensure_running(model_p, mmproj_p, port=PORT)
    load_s = time.time() - t0
    assert ok, f"[{name}] server failed to start: {server.last_error}"
    assert server.is_running(), f"[{name}] is_running() False after start"

    # reuse check: second call with same model must be near-instant
    t0 = time.time()
    assert server.ensure_running(model_p, mmproj_p, port=PORT)
    reuse_s = time.time() - t0
    assert reuse_s < 5, f"[{name}] reuse took {reuse_s:.1f}s — not reusing"

    brain = LocalBrain(port=PORT)
    t0 = time.time()
    resp = brain.ask("What text do you see in this image?", b64, [])
    infer_s = time.time() - t0
    check_answer(resp, f"[{name}] ask")
    return resp, load_s, infer_s


def main() -> int:
    # --- cheap unit checks (no server) ---
    rec = hardware.recommend_model()
    assert rec is not None, "recommend_model() returned None with models on disk"
    assert Path(rec["model_path"]).is_file() and Path(rec["mmproj_path"]).is_file()
    ram = hardware.total_ram_gb()
    expect = "gemma" if ram >= 7 else "lfm2"
    assert expect in Path(rec["model_path"]).name.lower(), \
        f"RAM {ram:.1f}GB but recommended {rec['model_path']}"
    print(f"UNIT hardware: RAM {ram:.1f}GB -> {rec['label']} ({Path(rec['model_path']).name})")

    from brain import Brain
    b = Brain("", "x", engine="local", prewarm=False)
    assert b.engine_label().startswith("LOCAL — "), b.engine_label()
    assert b.has_key, "local engine with models on disk should report available"
    bc = Brain("", "claude-opus-5", engine="claude", prewarm=False)
    assert bc.engine_label() == "CLAUDE — claude-opus-5"
    assert not bc.has_key, "claude engine with no key must report unavailable"
    print(f"UNIT facade: labels ok ({b.engine_label()!r} / {bc.engine_label()!r})")

    # --- the real thing ---
    b64 = make_test_image_b64()
    print(f"test image: 800x600 'TEST 42', {len(b64)} b64 chars")
    server = LocalServer()
    results = {}
    try:
        for name, model, mmproj in TIERS:
            print(f"\n=== {name.upper()} tier: {model} ===")
            resp, load_s, infer_s = run_tier(server, name, model, mmproj, b64)
            results[name] = (resp, load_s, infer_s)
            print(f"[{name}] load {load_s:.1f}s | inference {infer_s:.1f}s")
            print(f"[{name}] RESPONSE: {resp}")

        # facade through the local path, reusing the running quality server
        brain = Brain("", "x", engine="local", prewarm=False)
        brain._server = server  # share the already-running server instance
        t0 = time.time()
        resp = brain.ask("Describe this image in one short sentence.", b64)
        check_answer(resp, "[facade] ask")
        print(f"\n[facade] LOCAL ask via Brain(): {time.time() - t0:.1f}s")
        print(f"[facade] RESPONSE: {resp}")
        results["facade"] = (resp, 0.0, time.time() - t0)
    finally:
        server.stop()
    assert not server.is_running(), "server still running after stop()"
    print("\nserver stopped cleanly — ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
