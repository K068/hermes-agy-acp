"""Duplex JSON-RPC (NDJSON) transport for one long-lived ``agy acp`` child.

Why this exists instead of reusing :mod:`agent.acp_client`: that client starts a **fresh** process
per request and replays the whole conversation, which is exactly the mode this provider replaces.
Here one process serves many turns of one conversation, so the transport must keep a request table
alive for the whole process lifetime, answer server→client requests from arbitrary threads, and
never block a turn on a full stderr pipe.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import proctree
from .settings import Settings

logger = logging.getLogger(__name__)

INITIALIZE_PARAMS: dict[str, Any] = {
    "protocolVersion": 1,
    # The provider lets agy use its own filesystem tooling; the bridge is offered anyway so a
    # future agy release that prefers client fs calls still works instead of failing the turn.
    "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}, "terminal": False},
    "clientInfo": {"name": "hermes-agy-acp", "title": "Hermes Agent (agy_acp)", "version": "0.1.0"},
}

_RATE_LIMIT_RE = re.compile(
    r"quota|resource[ _-]?exhausted|rate[ _-]?limit|too many requests|\b429\b|usage limit|"
    r"limit reached|exceeded your",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(r"not (?:logged|signed) in|authentication required|unauthorized|\b401\b|login", re.IGNORECASE)

#: JSON-RPC code ACP uses for "the client must authenticate first".
AUTH_REQUIRED_CODE = -32000


class AcpError(RuntimeError):
    """A JSON-RPC or transport failure, tagged with a machine-readable ``kind``."""

    def __init__(self, message: str, *, kind: str = "server", method: str = "", code: int | None = None,
                 data: Any = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.method = method
        self.code = code
        self.data = data

    @property
    def rate_limited(self) -> bool:
        return self.kind == "rate_limit"


def classify_error(message: str, code: int | None) -> str:
    # The message wins over the code: a quota refusal can arrive under several codes (including the
    # generic auth code on some relays), and misclassifying it as "not signed in" hides the real fix.
    if _RATE_LIMIT_RE.search(message or ""):
        return "rate_limit"
    if code == AUTH_REQUIRED_CODE or _AUTH_RE.search(message or ""):
        return "auth"
    return "server"


@dataclass
class _Pending:
    method: str
    event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class AgyConnection:
    """One ``agy acp`` child process and its JSON-RPC request table."""

    def __init__(
        self,
        settings: Settings,
        *,
        cwd: str,
        on_notification: Callable[[dict[str, Any]], None] | None = None,
        on_server_request: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        label: str = "agy",
        trace: Any = None,
    ) -> None:
        self.settings = settings
        self.cwd = cwd
        self.label = label
        self.trace = trace
        self.capabilities: dict[str, Any] = {}
        self.agent_info: dict[str, Any] = {}
        self.stderr_tail: list[str] = []
        self._on_notification = on_notification
        self._on_server_request = on_server_request
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 0
        self._closed = threading.Event()
        self._eof = threading.Event()
        self._stderr_path: Path | None = None
        self.started_at = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────
    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def alive(self) -> bool:
        """False as soon as stdout reaches EOF — a crashing child's ``poll()`` can lag that by a moment."""
        return (self._proc is not None and self._proc.poll() is None
                and not self._closed.is_set() and not self._eof.is_set())

    def start(self) -> dict[str, Any]:
        """Spawn the child and complete the ACP handshake. Returns ``initialize``'s result."""
        settings = self.settings
        argv = [settings.command, *settings.args]
        log_dir = Path(settings.log_dir or ".")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._stderr_path = log_dir / f"stderr-{self.label}.log"
        with contextlib.suppress(OSError):
            if self._stderr_path.exists() and self._stderr_path.stat().st_size > 8 * 1024 * 1024:
                self._stderr_path.write_text("", encoding="utf-8")
        stderr_sink = open(self._stderr_path, "a", encoding="utf-8", errors="replace")
        env = self._child_env()
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_sink,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                cwd=self.cwd, env=env, **proctree.spawn_kwargs(),
            )
        except FileNotFoundError as exc:
            stderr_sink.close()
            raise AcpError(
                f"agy_acp: cannot start '{settings.command}'. Install the Antigravity CLI or set AGY_ACP_BIN.",
                kind="missing_command",
            ) from exc
        except OSError as exc:
            stderr_sink.close()
            raise AcpError(f"agy_acp: cannot start '{settings.command}': {exc}", kind="transport") from exc
        self._proc = proc
        self.started_at = time.monotonic()
        proctree.attach(proc)
        if proc.stdin is None or proc.stdout is None:
            proctree.terminate_tree(proc)
            raise AcpError("agy_acp: child process exposed no stdio pipes.", kind="transport")
        self._reader = threading.Thread(target=self._read_stdout, name=f"agy-acp-out-{proc.pid}", daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, args=(stderr_sink,), name=f"agy-acp-err-{proc.pid}", daemon=True)
        self._stderr_reader.start()
        try:
            result = self.request("initialize", INITIALIZE_PARAMS, timeout=settings.connect_timeout)
        except Exception:
            self.close("initialize failed")
            raise
        self.capabilities = dict(result.get("agentCapabilities") or {})
        self.agent_info = dict(result.get("agentInfo") or {})
        return result

    def _child_env(self) -> dict[str, str]:
        """Child environment: inherited, plus the account HOME / proxy this provider was told to use.

        Hermes already strips Tier-1 secrets from subprocess environments; importing that helper
        inside Hermes keeps the same policy. Standalone (tests), a plain copy is used.
        """
        try:  # pragma: no cover - exercised inside Hermes only
            from tools.environments.local import hermes_subprocess_env  # type: ignore

            env = dict(hermes_subprocess_env(inherit_credentials=True))
        except Exception:
            env = dict(os.environ)
        env.setdefault("LANG", "C.UTF-8")
        env["PYTHONIOENCODING"] = "utf-8"
        if self.settings.child_home:
            env["HOME"] = self.settings.child_home
            env.setdefault("USERPROFILE", self.settings.child_home)
        proxy = self.settings.proxy
        if proxy:
            for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
                env[key] = proxy
        # The child must not inherit an outer agent's session markers as its own identity.
        for key in ("HERMES_SESSION_ID", "HERMES_CRON_SESSION"):
            env.pop(key, None)
        return env

    # ── io ────────────────────────────────────────────────────────────
    def _read_stderr(self, sink) -> None:
        try:
            for line in sink:  # type: ignore[union-attr]
                text = line.rstrip("\n")
                self.stderr_tail.append(text)
                if len(self.stderr_tail) > 80:
                    del self.stderr_tail[:-80]
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                sink.close()

    def _read_stdout(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in proc.stdout:
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    logger.debug("agy_acp: non-JSON stdout line: %s", line[:200])
                    continue
                if not isinstance(message, dict):
                    continue
                if self.trace is not None:
                    self.trace.frame("in", message, truncated_hint=len(line))
                try:
                    self._dispatch(message)
                except Exception:  # a handler bug must not kill the reader
                    logger.warning("agy_acp: dispatch failed", exc_info=True)
        finally:
            self._eof.set()
            self._fail_pending("agy acp process exited")

    def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if isinstance(method, str):
            if "id" in message:
                response = None
                if self._on_server_request is not None:
                    response = self._on_server_request(message)
                if response is None:
                    response = {
                        "jsonrpc": "2.0", "id": message.get("id"),
                        "error": {"code": -32601, "message": f"agy_acp does not implement '{method}'"},
                    }
                self._send(response)
            else:
                if self._on_notification is not None:
                    self._on_notification(message)
            return
        if "result" in message or "error" in message:
            with self._pending_lock:
                pending = self._pending.pop(message.get("id"), None)
            if pending is None:
                return
            if "error" in message:
                pending.error = dict(message.get("error") or {})
            else:
                pending.result = dict(message.get("result") or {})
            pending.event.set()

    def _fail_pending(self, reason: str) -> None:
        with self._pending_lock:
            pending, self._pending = dict(self._pending), {}
        for waiter in pending.values():
            waiter.error = {"code": -32099, "message": reason}
            waiter.event.set()
        if not self._closed.is_set():
            logger.info("agy_acp: %s", reason)

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            raise AcpError("agy_acp: child stdin is closed", kind="transport")
        data = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._write_lock:
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                raise AcpError(f"agy_acp: write to child failed: {exc}", kind="transport") from exc
        if self.trace is not None:
            self.trace.frame("out", payload, truncated_hint=len(data))

    # ── rpc ───────────────────────────────────────────────────────────
    def request(self, method: str, params: dict[str, Any], *, timeout: float | None = None,
                tick: Any = None, tick_interval: float = 0.2) -> dict[str, Any]:
        """Send a request and wait for its response.

        ``tick`` is called every ``tick_interval`` seconds while waiting; returning a truthy value
        means "this caller wants the request aborted" (the caller has already sent whatever cancel
        notification it needs) — the wait continues so the agent's own ``stopReason`` is still seen.
        """
        with self._pending_lock:
            self._next_id += 1
            request_id = self._next_id
            waiter = _Pending(method=method)
            self._pending[request_id] = waiter
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        except Exception:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            slice_timeout = tick_interval if tick is not None else None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with self._pending_lock:
                        self._pending.pop(request_id, None)
                    raise AcpError(f"agy_acp: {method} timed out after {timeout:.0f}s", kind="timeout", method=method)
                slice_timeout = remaining if slice_timeout is None else min(slice_timeout, remaining)
            if waiter.event.wait(slice_timeout):
                break
            if tick is not None:
                try:
                    tick()
                except Exception:
                    pass
            elif deadline is None:
                break
        if waiter.error is not None:
            error = waiter.error
            message = str(error.get("message") or error)
            code = error.get("code")
            data = error.get("data")
            if data:
                message = f"{message} | {json.dumps(data, ensure_ascii=False)[:400]}"
            raise AcpError(message, kind=classify_error(message, code if isinstance(code, int) else None),
                           method=method, code=code if isinstance(code, int) else None, data=data)
        return waiter.result or {}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def cancel(self, session_id: str) -> bool:
        """Send ``session/cancel``. Never raises: a dead child has nothing left to cancel."""
        if not session_id:
            return False
        try:
            self.notify("session/cancel", {"sessionId": session_id})
            return True
        except Exception:
            return False

    # ── teardown ──────────────────────────────────────────────────────
    def close(self, reason: str = "", *, session_id: str = "", kill_timeout: float = 3.0) -> None:
        if self._closed.is_set():
            return
        if session_id:
            self.cancel(session_id)
        self._closed.set()
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            proctree.terminate_tree(proc, timeout=kill_timeout)
        self._fail_pending(reason or "agy acp closed")
        if self.trace is not None:
            with contextlib.suppress(Exception):
                self.trace.close()
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=2.0)
        if self._stderr_reader is not None and self._stderr_reader is not threading.current_thread():
            self._stderr_reader.join(timeout=2.0)

    def __enter__(self) -> "AgyConnection":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close("context exit")


def drain_queue(q: "queue.Queue[Any]") -> None:
    """Empty a queue without blocking (used when a turn ends mid-stream)."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
