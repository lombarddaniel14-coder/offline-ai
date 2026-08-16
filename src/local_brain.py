"""LocalBrain — talks to the local llama-server over the OpenAI-compatible
chat completions API, with vision (image_url data-URI parts).

Mirrors the Claude brain's behavior: ask() answers a spoken command with the
current camera frame attached; narrate() is the stateless watch-mode call
using the same [NOTABLE] tag convention. Small models get a compact 2-3
sentence version of the TBS persona.

Every failure path returns a clear spoken-word error STRING — this class
never raises into the GUI.
"""

import json
import socket
import urllib.error
import urllib.request

NOTABLE_TAG = "[NOTABLE]"

# Compact persona — small local models drift with long system prompts.
LOCAL_TBS_SYSTEM = (
    "You are Buddy, a voice assistant that sees through a live camera; an "
    "attached image is your current camera view, live in the room. Reply in "
    "1-3 short spoken sentences — no markdown, no lists, no emoji — and "
    "occasionally address the user as 'Sir'. If the image is too dark or "
    "unclear, say so plainly."
)

LOCAL_NARRATE_SYSTEM = (
    "You are Buddy watching a live camera feed, one frame every few seconds. "
    "In 1-2 short spoken sentences, describe what changed since your previous "
    "observation; if nothing meaningful changed, say so in a few words. If you "
    "see something genuinely noteworthy (a person appearing or leaving, "
    "something unusual), start your reply with " + NOTABLE_TAG +
    " followed by a short proactive remark."
)

SERVER_DOWN_MSG = ("My local brain isn't running yet, sir — give me a moment "
                   "and try again.")
TIMEOUT_MSG = ("The local model is taking too long to answer, sir. Try again "
               "in a moment.")
MALFORMED_MSG = "The local model returned something I couldn't read, sir."

HISTORY_EXCHANGES = 8  # last N exchanges (text-only) sent with each ask


class LocalBrain:
    def __init__(self, port: int = 8090, timeout: float = 120.0,
                 max_tokens: int = 300):
        self.port = port
        self.timeout = timeout        # CPU inference is slow — be generous
        self.max_tokens = max_tokens

    # -- wire ------------------------------------------------------------

    def _post(self, messages: list[dict]) -> str:
        """POST /v1/chat/completions; always returns a string (answer or
        spoken error message)."""
        payload = {
            "model": "tbs-local",   # llama-server serves a single model
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            return f"The local model refused that request, sir (HTTP {e.code})."
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError)):
                return TIMEOUT_MSG
            return SERVER_DOWN_MSG
        except (TimeoutError, socket.timeout):
            return TIMEOUT_MSG
        except OSError:
            return SERVER_DOWN_MSG

        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
            text = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return MALFORMED_MSG
        if not isinstance(text, str) or not text.strip():
            return MALFORMED_MSG
        return text.strip()

    @staticmethod
    def _image_part(jpeg_b64: str) -> dict:
        return {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}

    # -- public ------------------------------------------------------------

    def ask(self, question: str, jpeg_b64: str | None,
            history: list[dict] | None = None) -> str:
        """Answer a spoken command, attaching the current frame if given.

        history: list of {"role": "user"/"assistant", "text": str} — the
        facade's rolling text-only history; the last 8 exchanges are sent.
        """
        messages: list[dict] = [{"role": "system", "content": LOCAL_TBS_SYSTEM}]
        for h in (history or [])[-HISTORY_EXCHANGES * 2:]:
            messages.append({"role": h["role"], "content": h["text"]})
        if jpeg_b64:
            messages.append({"role": "user", "content": [
                self._image_part(jpeg_b64),
                {"type": "text",
                 "text": f"{question}\n\n(The attached image is your current "
                         "camera view.)"},
            ]})
        else:
            messages.append({"role": "user", "content": question})
        return self._post(messages)

    def narrate(self, jpeg_b64: str, previous: str | None = None) -> str:
        """Watch mode: describe the frame, noting changes vs the previous
        observation. Stateless — same [NOTABLE] convention as the Claude brain."""
        if previous:
            prompt = (f'Your previous observation was: "{previous}"\n'
                      "Here is the current frame.")
        else:
            prompt = "This is the first frame. Describe what you see."
        messages = [
            {"role": "system", "content": LOCAL_NARRATE_SYSTEM},
            {"role": "user", "content": [
                self._image_part(jpeg_b64),
                {"type": "text", "text": prompt},
            ]},
        ]
        return self._post(messages)
