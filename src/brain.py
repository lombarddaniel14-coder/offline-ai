"""Brain — engine facade: Claude API or the local llama-server, one interface.

Brain.engine is 'auto' | 'local' | 'claude':
  * 'claude' — original TBS Eyes behavior (Anthropic API, full persona).
  * 'local'  — bundled llama-server + best on-disk vision model (offline).
  * 'auto'   — claude when an API key is set AND https://api.anthropic.com
               answers within 3s; otherwise local.

tbs_app.py keeps calling the exact same surface it always has:
Brain(api_key, model, max_tokens, history_turns), .ask(command, jpeg_b64),
.narrate(jpeg_b64, previous), .has_key, .clear_history(). New for the status
bar: engine_label() -> 'LOCAL — <model>' / 'CLAUDE — <model>'.

The rolling text-only history lives here in the facade, shared by both
engines. The local path NEVER raises into the GUI — it returns spoken-word
error strings; the Claude path raises the same anthropic/NoAPIKeyError
exceptions tbs_app already handles.
"""

import threading
import urllib.error
import urllib.request

import anthropic

import hardware
from local_brain import LocalBrain, NOTABLE_TAG  # noqa: F401 (re-exported)
from server_manager import LocalServer

# Watch mode asks the model to prefix genuinely noteworthy observations with
# this tag so the app knows to speak up and save a screenshot of that frame.
# (NOTABLE_TAG itself is defined in local_brain and re-exported here so both
# engines share one constant.)

TBS_SYSTEM = (
    "You are Buddy, a voice assistant with eyes: you see through a live camera "
    "connected to the user's computer. Each user message may include the camera's "
    "current view — treat it as what you see right now, in the room, not as an "
    "uploaded photo. Your replies are read aloud by text-to-speech, so keep them "
    "brief and spoken-word-natural: 1-3 sentences unless the user asks for detail. "
    "No markdown, no lists, no emoji. Address the user as 'Sir' occasionally — "
    "naturally, not in every reply and never comically. Reference what you see "
    "when it's relevant to the question. If the image is too dark, blurry, or the "
    "subject is out of frame, say so plainly."
)

NARRATE_SYSTEM = (
    "You are Buddy, watching a live camera feed, one frame every few seconds. "
    "Describe what you see in 1-2 short spoken sentences. You are given your "
    "previous observation — focus on what CHANGED since then; if nothing "
    "meaningful changed, say so in a few words. If you see something genuinely "
    "noteworthy (a person appearing or leaving, an object moving, something "
    f"unusual), start your response with {NOTABLE_TAG} followed by a short "
    "proactive remark, the kind TBS would say unprompted "
    "(e.g. 'Sir — someone just walked in.')."
)

NO_KEY_MSG = ("I can't think yet, sir — no local model files and no API key. "
              "Add models to the models folder, or a key in Settings.")
NO_LOCAL_MODEL_MSG = ("I couldn't find any local model files, sir. The models "
                      "folder seems to be empty.")
LOCAL_START_FAIL_MSG = ("I couldn't start the local model, sir. Check the "
                        "engine and models folders.")

ANTHROPIC_PROBE_URL = "https://api.anthropic.com"
PROBE_TIMEOUT_S = 3.0


class NoAPIKeyError(Exception):
    """Raised when a Claude query needs the API and no key is configured."""


def _anthropic_reachable(timeout: float = PROBE_TIMEOUT_S) -> bool:
    """True if api.anthropic.com answers ANY HTTP response within timeout.

    The probe runs in a daemon thread with a hard join deadline: urlopen's
    timeout does NOT cover DNS resolution (getaddrinfo can hang 10s+ on some
    offline hosts), and this is called during Brain construction — it must
    never hang the caller beyond ~timeout+1s."""
    result: list[bool] = []

    def probe():
        try:
            req = urllib.request.Request(ANTHROPIC_PROBE_URL, method="GET")
            with urllib.request.urlopen(req, timeout=timeout):
                result.append(True)
        except urllib.error.HTTPError:
            result.append(True)  # server answered (4xx/5xx) — network is up
        except Exception:
            result.append(False)

    t = threading.Thread(target=probe, daemon=True, name="api-probe")
    t.start()
    t.join(timeout + 1.0)
    return bool(result and result[0])


class Brain:
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024,
                 history_turns: int = 12, engine: str = "auto",
                 local_model: str = "auto", port: int = 8090,
                 prewarm: bool = True):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.history_turns = history_turns
        self.engine = engine if engine in ("auto", "local", "claude") else "auto"
        self.local_model = local_model or "auto"  # 'auto' or a .gguf filename

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else None
        # history: list of {"role": "user"/"assistant", "text": str} — text only
        self._history: list[dict] = []
        self._lock = threading.Lock()

        self._local = LocalBrain(port)
        self._server = LocalServer()
        self._rec: dict | None = None      # hardware.recommend_model() cache
        self._rec_checked = False

        self._resolved = self._resolve_engine()
        if prewarm and self._resolved == "local":
            # Model load takes 60-120s on CPU — start it now, off-thread, so
            # the first question doesn't eat the whole wait.
            threading.Thread(target=self._ensure_local, daemon=True,
                             name="local-prewarm").start()

    # -- engine selection ------------------------------------------------

    def _resolve_engine(self) -> str:
        if self.engine == "local":
            return "local"
        if self.engine == "claude":
            return "claude"
        # auto: claude only when a key is set AND the API is reachable fast
        if self.api_key and _anthropic_reachable():
            return "claude"
        return "local"

    def refresh_engine(self) -> str:
        """Re-run auto engine selection (e.g. after settings change)."""
        self._resolved = self._resolve_engine()
        return self._resolved

    def engine_label(self) -> str:
        if self._resolved == "claude":
            return f"CLAUDE — {self.model}"
        rec = self._recommendation()
        return f"LOCAL — {rec['label']}" if rec else "LOCAL — no model found"

    # -- availability --------------------------------------------------

    def _recommendation(self) -> dict | None:
        if not self._rec_checked:
            # An explicit Settings pick wins; fall back to the hardware
            # recommendation if that file has gone missing.
            explicit = hardware.pair_for_model(self.local_model)
            self._rec = explicit or hardware.recommend_model()
            self._rec_checked = True
        return self._rec

    @property
    def has_key(self) -> bool:
        """True when the active engine can actually think: an API key for the
        Claude path, or model files on disk for the local path. (Name kept for
        tbs_app compatibility.)"""
        if self._resolved == "claude":
            return bool(self.api_key)
        return self._recommendation() is not None

    # -- local engine ---------------------------------------------------

    def _ensure_local(self) -> str | None:
        """Make sure llama-server is up with the recommended model.
        Returns None when ready, else a spoken-word error string."""
        rec = self._recommendation()
        if rec is None:
            return NO_LOCAL_MODEL_MSG
        ok = self._server.ensure_running(rec["model_path"], rec["mmproj_path"],
                                         port=self._local.port)
        if ok and self._server.port is not None:
            # The server may have bound a different port if the preferred one
            # was already taken on this host — keep LocalBrain pointed at it.
            self._local.port = self._server.port
        return None if ok else LOCAL_START_FAIL_MSG

    # -- claude engine (unchanged original behavior) ------------------------

    def _require_client(self) -> anthropic.Anthropic:
        if self._client is None:
            raise NoAPIKeyError(NO_KEY_MSG)
        return self._client

    @staticmethod
    def _extract_text(response) -> str:
        if response.stop_reason == "refusal":
            return "I'd rather not comment on that frame, sir."
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def _claude_ask(self, command: str, jpeg_b64: str | None,
                    past: list[dict]) -> str:
        client = self._require_client()
        content: list[dict] = []
        if jpeg_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": jpeg_b64},
            })
        note = (command if jpeg_b64 is None else
                f"{command}\n\n(The attached image is your current camera view.)")
        content.append({"type": "text", "text": note})
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=TBS_SYSTEM,
            messages=[{"role": p["role"], "content": p["text"]} for p in past]
                     + [{"role": "user", "content": content}],
        )
        return self._extract_text(response)

    def _claude_narrate(self, jpeg_b64: str, previous: str | None) -> str:
        client = self._require_client()
        if previous:
            prompt = (f'Your previous observation was: "{previous}"\n'
                      "Here is the current frame.")
        else:
            prompt = "This is the first frame. Describe what you see."
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=NARRATE_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg",
                                "data": jpeg_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return self._extract_text(response)

    # -- public interface (same signatures tbs_app calls today) ----------

    def ask(self, command: str, jpeg_b64: str | None) -> str:
        """Answer a spoken command, attaching the current camera frame if given.

        Claude path: raises NoAPIKeyError / anthropic.* like it always has.
        Local path: never raises — returns spoken error strings instead.
        """
        with self._lock:
            past = list(self._history)

        if self._resolved == "claude":
            answer = self._claude_ask(command, jpeg_b64, past)
        else:
            err = self._ensure_local()
            if err:
                return err
            answer = self._local.ask(command, jpeg_b64, past)

        with self._lock:
            self._history.append({"role": "user", "text": command})
            self._history.append({"role": "assistant", "text": answer})
            excess = len(self._history) - self.history_turns * 2
            if excess > 0:
                del self._history[:excess]
        return answer

    def narrate(self, jpeg_b64: str, previous: str | None = None) -> str:
        """Watch mode: describe the frame, noting changes from the previous one.
        Stateless — does not touch conversation history."""
        if self._resolved == "claude":
            return self._claude_narrate(jpeg_b64, previous)
        err = self._ensure_local()
        if err:
            return err
        return self._local.narrate(jpeg_b64, previous)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def shutdown(self) -> None:
        """Stop the local llama-server if this brain started one."""
        self._server.stop()
