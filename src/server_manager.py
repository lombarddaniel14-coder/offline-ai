"""LocalServer — lifecycle manager for the bundled llama-server.exe.

Starts <root>/engine/llama-server.exe hidden (no console window), polls
GET /health until the model is loaded (first CPU load can take 60-120s,
so the timeout is 180s), reuses an already-healthy server when the model
is unchanged, and terminates the whole process tree on stop().

Crash safety / correctness:
  * The server is never spawned onto a port someone else already owns —
    the preferred port is bind-tested first and a free ephemeral port is
    used instead if needed. That guarantees the health poll only ever
    sees OUR child (a foreign server answering /health can no longer be
    mistaken for a successful start while our child dies on bind).
  * A pid breadcrumb (<root>/config/llama-server.pid) is written at spawn
    and removed on clean stop. If the app dies without stop() (task-kill,
    crash, power loss), the next launch reaps the orphaned llama-server —
    verified by owner-pid liveness and process image path, so PID reuse
    can never kill an innocent process.
  * The server log lives INSIDE the app root (<root>/_logs/), never in the
    host %TEMP% — zero traces on host PCs.

Everything is resolved relative to app_root() — no absolute paths, fully
portable (flash-drive safe).
"""

import atexit
import ctypes
import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from portable import app_root

CREATE_NO_WINDOW = 0x08000000  # Windows: don't flash a console window
HEALTH_TIMEOUT_S = 180         # first CPU model load can take 60-120s
POLL_INTERVAL_S = 1.0
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


# -- process inspection (Windows, ctypes — no extra deps) --------------------

def _pid_alive(pid: int) -> bool:
    """True if a process with this pid is currently running."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        if k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return code.value == _STILL_ACTIVE
        return False
    finally:
        k32.CloseHandle(h)


def _pid_image(pid: int) -> str:
    """Full executable path of a running process ('' if unavailable)."""
    if os.name != "nt":
        return ""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_ulong(len(buf))
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return ""
    finally:
        k32.CloseHandle(h)


# -- port selection -----------------------------------------------------------

def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pick_port(preferred: int) -> int:
    """The preferred port if free, else a free ephemeral port. A portable
    product lands on arbitrary host PCs — anything may already own 8090."""
    if _port_free(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalServer:
    """Owns exactly one llama-server child process at a time."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._model_path: str | None = None
        self._port: int | None = None
        self._lock = threading.RLock()
        self._log_path: Path | None = None
        self._pid_file = app_root() / "config" / "llama-server.pid"
        self._reaped = False
        self.last_error = ""
        # Never leave a llama-server orphaned if the app/tests exit cleanly.
        # (Unclean death — task-kill, power loss — is covered by the pid
        # breadcrumb: the next launch reaps the orphan in _reap_stale().)
        atexit.register(self.stop)

    @property
    def port(self) -> int | None:
        """The port the running server actually bound (may differ from the
        preferred port passed to ensure_running)."""
        return self._port

    # -- health --------------------------------------------------------

    @staticmethod
    def _health_ok(port: int) -> bool:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def is_running(self) -> bool:
        with self._lock:
            return (self._proc is not None
                    and self._proc.poll() is None
                    and self._port is not None
                    and self._health_ok(self._port))

    # -- start / reuse ---------------------------------------------------

    def ensure_running(self, model_path, mmproj_path, port: int = 8090,
                       ctx: int = 4096) -> bool:
        """Start (or reuse) llama-server serving model_path. `port` is the
        PREFERRED port — if another process owns it, a free ephemeral port is
        used instead (read the actual one from .port afterwards). Blocks until
        the server answers /health with 200, or returns False (see .last_error)."""
        model_path = str(model_path)
        mmproj_path = str(mmproj_path)
        with self._lock:
            # Reuse: same model, OUR child alive, health green on the port it
            # actually bound (the preferred port is irrelevant for reuse).
            if (self._proc is not None and self._proc.poll() is None
                    and self._model_path == model_path
                    and self._port is not None and self._health_ok(self._port)):
                return True

            self.stop()  # different model / dead process / bad health

            # One-time per process: kill a llama-server orphaned by a
            # previous crashed run (atexit never fires on task-kill).
            if not self._reaped:
                self._reaped = True
                self._reap_stale()

            exe = app_root() / "engine" / "llama-server.exe"
            if not exe.is_file():
                self.last_error = f"engine not found: {exe}"
                return False
            for p in (model_path, mmproj_path):
                if not Path(p).is_file():
                    self.last_error = f"model file not found: {p}"
                    return False

            # Never spawn onto an occupied port: the health poll below must
            # only ever see OUR child. (A foreign server already healthy on
            # the port would otherwise be mistaken for a successful start
            # while our child dies on bind.)
            port = _pick_port(port)

            cmd = [str(exe),
                   "-m", model_path,
                   "--mmproj", mmproj_path,
                   "--host", "127.0.0.1",
                   "--port", str(port),
                   "-c", str(ctx),
                   "--no-webui"]

            # Server log lives INSIDE the app root — zero traces on host PCs.
            log_dir = app_root() / "_logs"
            self._log_path = log_dir / f"llama-server-{port}.log"
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_f = open(self._log_path, "wb")
            except OSError:  # e.g. write-locked flash drive — run logless
                log_f = subprocess.DEVNULL
            flags = CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=log_f, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=flags,
                    cwd=str(exe.parent))
            except OSError as e:
                self.last_error = f"failed to launch llama-server: {e}"
                self._proc = None
                return False
            finally:
                if log_f is not subprocess.DEVNULL:
                    try:
                        log_f.close()  # child holds its own handle
                    except OSError:
                        pass

            self._model_path = model_path
            self._port = port
            self._write_pid_file()

            deadline = time.time() + HEALTH_TIMEOUT_S
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    self.last_error = (
                        f"llama-server exited with code {self._proc.returncode}"
                        f" during startup.{self._log_tail()}")
                    self._proc = None
                    self._model_path = None
                    self._port = None
                    self._clear_pid_file()
                    return False
                if self._health_ok(port):
                    self.last_error = ""
                    return True
                time.sleep(POLL_INTERVAL_S)

            self.last_error = (
                f"llama-server did not become healthy within {HEALTH_TIMEOUT_S}s."
                f"{self._log_tail()}")
            self.stop()
            return False

    # -- stop -----------------------------------------------------------

    def stop(self) -> None:
        """Terminate the server process tree. Safe to call repeatedly."""
        with self._lock:
            proc, self._proc = self._proc, None
            self._model_path = None
            self._port = None
            self._clear_pid_file()
            if proc is None or proc.poll() is not None:
                return
            if os.name == "nt":
                # /T kills the whole tree (llama-server spawns no children
                # today, but stay safe).
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, creationflags=CREATE_NO_WINDOW)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # -- crash-safety breadcrumb ------------------------------------------

    def _write_pid_file(self) -> None:
        """So the NEXT launch can reap this server if we die without stop()."""
        try:
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)
            self._pid_file.write_text(json.dumps({
                "server_pid": self._proc.pid,
                "owner_pid": os.getpid(),
                "exe": str(app_root() / "engine" / "llama-server.exe"),
            }), encoding="utf-8")
        except OSError:
            pass  # read-only media: no breadcrumb possible (and no trace risk)

    def _clear_pid_file(self) -> None:
        try:
            self._pid_file.unlink(missing_ok=True)
        except OSError:
            pass

    def _reap_stale(self) -> None:
        """Kill a llama-server orphaned by a previous crashed run.

        Only acts when (a) the breadcrumb's owner process is gone — a live
        owner (this process or a sibling app instance) keeps its server —
        and (b) the recorded pid is really a llama-server.exe right now
        (PID reuse can never kill an innocent process)."""
        if os.name != "nt":
            return
        try:
            info = json.loads(self._pid_file.read_text(encoding="utf-8"))
            server_pid = int(info["server_pid"])
            owner_pid = int(info["owner_pid"])
            exe = str(info.get("exe", ""))
        except (OSError, ValueError, KeyError, TypeError):
            return  # no/corrupt breadcrumb — nothing to reap
        if _pid_alive(owner_pid):
            return  # owner app still running — that server is not orphaned
        if _pid_alive(server_pid):
            img = _pid_image(server_pid)
            ours = str(app_root() / "engine" / "llama-server.exe")
            if (img and Path(img).name.lower() == "llama-server.exe"
                    and img.lower() in (exe.lower(), ours.lower())):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(server_pid)],
                    capture_output=True, creationflags=CREATE_NO_WINDOW)
        self._clear_pid_file()

    # -- diagnostics ------------------------------------------------------

    def _log_tail(self, lines: int = 12) -> str:
        if self._log_path is None:
            return ""
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        tail = "\n".join(text.splitlines()[-lines:]).strip()
        return f"\nlast log lines:\n{tail}" if tail else ""
