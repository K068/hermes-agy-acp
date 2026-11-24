"""OpenAI-client-shaped facade the Hermes agent loop talks to, backed by one stateful ACP session.

Hermes calls ``client.chat.completions.create(model=..., messages=[...], tools=[...])`` and reads a
completion back. Everything below that line is ours:

* ``messages`` is *not* forwarded. The newest user message is sliced out
  (:mod:`.prompt`) and sent as the turn; the agent's own history stays in its process;
* ``tools`` is ignored on purpose — the agent executes its own file/terminal tools, so forwarding
  Hermes' overlapping schemas would only add thousands of tokens per turn and two owners per action;
* a turn is serialized account-wide, and ``session/cancel`` is wired to the two stop paths Hermes
  actually has (``close()`` and the cancel sentinel file).
"""

from __future__ import annotations

import atexit
import hashlib
import contextlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .manager import AgyAcpSessionManager, LoadFailedError, NonlinearTurnError, TurnBusyError
from .prompt import build_turn
from .protocol import AcpError
from .settings import Settings

logger = logging.getLogger(__name__)

_MANAGERS: dict[str, AgyAcpSessionManager] = {}
_MANAGER_LOCK = threading.Lock()


#: Settings that decide *how* a child is launched. The manager captures these at construction, so a
#: change (account home, proxy, binary, bridge name) must rebuild it — otherwise the gateway keeps
#: running the old account until someone restarts it.
_LAUNCH_FIELDS = ("command", "args", "child_home", "proxy", "model", "bridge_name",
                  "max_concurrent_turns", "capability_mode", "allow_background_origins")


def _launch_fingerprint(settings: Settings) -> tuple:
    return tuple(getattr(settings, name, None) for name in _LAUNCH_FIELDS)


def get_manager(settings: Settings | None = None) -> AgyAcpSessionManager:
    """Process-wide manager (one child pool, one account gate) keyed by its state database."""
    settings = settings or Settings.from_env()
    key = str(settings.state_db)
    fingerprint = _launch_fingerprint(settings)
    with _MANAGER_LOCK:
        manager = _MANAGERS.get(key)
        if manager is not None and getattr(manager, "_launch_fingerprint", None) != fingerprint:
            if manager.busy():
                # a turn is running on the old settings; leave it alone, the next call rebuilds
                return manager
            logger.info("agy_acp: launch settings changed; rebuilding the session manager")
            with contextlib.suppress(Exception):
                manager.close_all(reason="launch settings changed")
            manager = None
        if manager is None:
            manager = AgyAcpSessionManager(settings)
            manager._launch_fingerprint = fingerprint  # type: ignore[attr-defined]
            _MANAGERS[key] = manager
        return manager


@atexit.register
def _close_managers() -> None:  # pragma: no cover - process teardown
    with _MANAGER_LOCK:
        managers = list(_MANAGERS.values())
        _MANAGERS.clear()
    for manager in managers:
        try:
            manager.close_all(reason="process exit")
        except Exception:
            pass


def current_session_id() -> str:
    """The Hermes session this call belongs to, with a stable per-cwd fallback."""
    sid = ""
    try:
        from gateway.session_context import get_session_env  # type: ignore

        sid = str(get_session_env("HERMES_SESSION_ID") or "")
    except Exception:
        sid = ""
    sid = sid or os.environ.get("HERMES_SESSION_ID", "")
    if not sid:
        sid = "anon-" + hashlib.sha1(str(Path.cwd()).encode("utf-8")).hexdigest()[:8]
    return sid


def background_origin() -> str:
    """``""`` for a human turn, else a short label naming the automation that called us."""
    try:
        from gateway.session_context import get_session_env  # type: ignore

        cron = str(get_session_env("HERMES_CRON_SESSION") or "")
        source = str(get_session_env("HERMES_SESSION_SOURCE") or "")
    except Exception:
        cron = os.environ.get("HERMES_CRON_SESSION", "")
        source = ""
    if cron == "1":
        return "cron"
    if source.lower() in ("cron", "kanban", "webhook", "background", "heartbeat"):
        return source.lower()
    try:
        from agent.delegation_context import is_delegated_child_context  # type: ignore

        if is_delegated_child_context():
            return "subagent"
    except Exception:
        pass
    return ""


def _auxiliary_call_frame() -> str:
    """Name of the auxiliary-call module on the stack, or ``""`` for a main-loop turn.

    Hermes runs compression, titling, vision and background-review calls through the same
    ``create_client`` seam as the main loop, and gives the provider no flag for it. Walking a few
    frames at client-construction/call time is the only in-process signal, so it is kept cheap and
    bounded. Set ``AGY_ACP_ALLOW_AUX=1`` to let auxiliary traffic through deliberately.
    """
    frame = sys._getframe(1)
    depth = 0
    while frame is not None and depth < 40:
        name = os.path.basename(frame.f_code.co_filename)
        if name in ("auxiliary_client.py", "auxiliary_hooks.py"):
            return name
        frame = frame.f_back
        depth += 1
    return ""


def _effective_timeout(timeout: Any, default: float) -> float:
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
    values = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(values, default=default) if values else default


def _usage(payload: dict[str, Any]) -> Any:
    def pick(*names: str) -> int:
        for name in names:
            value = payload.get(name)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    prompt = pick("inputTokens", "input_tokens", "prompt_tokens", "promptTokens")
    completion = pick("outputTokens", "output_tokens", "completion_tokens", "completionTokens")
    # agy 1.5.15 sends ``cachedReadTokens`` (measured: 8458 on a warm turn); the other spellings are
    # kept for older/newer builds. Missing this key made every turn report cache_read = 0.
    cached = pick("cachedReadTokens", "cached_read_tokens", "cachedInputTokens", "cached_tokens",
                  "cacheReadInputTokens", "cacheReadTokens")
    total = pick("totalTokens", "total_tokens") or (prompt + completion)
    cache_write = pick("cachedWriteTokens", "cached_write_tokens", "cacheWriteInputTokens")
    thought = pick("thoughtTokens", "thought_tokens", "reasoningTokens", "reasoning_tokens")
    details = SimpleNamespace(cached_tokens=cached, cache_write_tokens=cache_write, thought_tokens=thought)
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total,
                           prompt_tokens_details=details, thought_tokens=thought)


def _stream_chunks(completion: Any) -> Any:
    try:
        from agent.acp_openai_bridge import completion_to_stream_chunks  # type: ignore

        return completion_to_stream_chunks(completion)
    except Exception:  # standalone (tests) fallback: same two-chunk shape
        choice = completion.choices[0]
        delta = SimpleNamespace(role="assistant", content=choice.message.content or None, tool_calls=None,
                                reasoning_content=choice.message.reasoning_content,
                                reasoning=choice.message.reasoning)
        data = SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
                               model=completion.model, usage=None)
        usage = SimpleNamespace(choices=[], model=completion.model, usage=completion.usage)
        return [data, usage]


class AgyAcpClient:
    """One client per Hermes agent, one process per Hermes chat (pooled by the manager)."""

    #: Already a complete client for the agent loop: no transport/async re-wrap.
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    placeholder_model = "agy"

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None,
        default_headers: dict[str, str] | None = None, command: str | None = None,
        args: list[str] | None = None, acp_cwd: str | None = None, timeout: float | None = None,
        settings: Settings | None = None, **_: Any,
    ) -> None:
        self.settings = settings or Settings.from_env()
        if command:
            self.settings.command = command
        if args:
            self.settings.args = tuple(args)
        self.api_key = api_key or self.placeholder_model
        self.base_url = base_url or "acp://agy"
        self._default_headers = dict(default_headers or {})
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._default_timeout = float(timeout) if isinstance(timeout, (int, float)) else self.settings.turn_timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed = False
        self._manager = get_manager(self.settings)
        self._last_session = ""
        self._turn_started_at: float | None = None
        self._cancel_pending = False
        self._turn_lock = threading.Lock()

    # ── public facade ────────────────────────────────────────────────
    def _create_chat_completion(
        self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None,
        timeout: Any = None, tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
        stream: bool = False, **_: Any,
    ) -> Any:
        if self.is_closed:
            raise RuntimeError("agy_acp client is closed")
        if not self.settings.allow_auxiliary:
            aux = _auxiliary_call_frame()
            if aux:
                raise RuntimeError(
                    f"agy_acp refuses auxiliary calls (traffic from {aux}): compression, titling, vision and "
                    "background review must not spend this subscription. Point auxiliary work at another "
                    "provider (e.g. `hermes config set model.aux_provider opencode-go`) or set "
                    "AGY_ACP_ALLOW_AUX=1 deliberately."
                )
        if not self.settings.allow_background:
            origin = background_origin()
            if origin and origin in tuple(getattr(self.settings, "allow_background_origins", ()) or ()):
                origin = ""  # this automation is explicitly trusted with the account
            if origin:
                raise RuntimeError(
                    f"agy_acp is interactive-only: this call came from '{origin}'. The provider drives a single "
                    "consumer subscription and must only run on a human's direct input (spec 1.2). Use another "
                    "provider for cron/subagents, or set AGY_ACP_ALLOW_BACKGROUND=1 to override deliberately."
                )
        sid = current_session_id()
        self._last_session = sid
        with self._turn_lock:
            self._turn_started_at = time.monotonic()
            self._cancel_pending = False
        model = str(model or self.settings.model or "")
        already = self._manager.memory_keys(sid)
        row = self._manager.store.get(sid) or {}
        turn_number = int(row.get("turn_count") or 0) + 1  # survives restarts, unlike an in-process counter
        system_prompt = next((str(m.get("content")) for m in (messages or [])
                              if isinstance(m, dict) and str(m.get("role") or "").lower() == "system"
                              and isinstance(m.get("content"), str)), "")
        turn = build_turn(
            messages or [], already_injected=already, cwd=self._cwd, session_id=sid,
            turn_number=turn_number, max_chars=self.settings.max_prompt_chars,
            work_dir=self.settings.work_dir, settings=self.settings, system_prompt=system_prompt,
        )
        if not turn.blocks:
            raise ValueError("agy_acp: the newest user message is empty; nothing to send.")
        interrupted = self._manager.take_interrupted(sid)
        if interrupted:
            note = (f"[System note: your previous turn in this session was interrupted when the provider "
                    f"restarted {int(interrupted.get('age_seconds') or 0)}s ago; whatever it had produced was "
                    f"not saved.]")
            turn.text = (turn.text + "\n\n" + note).strip()
            turn.blocks = [{"type": "text", "text": turn.text}] + [b for b in turn.blocks if b.get("type") != "text"]
            turn.notes.append("interrupted-turn note added")
        verdict = self._manager.check_linearity(sid, turn.user_hash) if self.settings.enforce_linear else "ok"
        if verdict != "ok" and self.settings.nonlinear_action != "resync":
            self._manager.raise_nonlinear(sid, verdict)  # raises before any process is spawned
        try:
            pooled = self._manager.session_for(sid, cwd=self._cwd, model=model)
            if verdict != "ok":
                self._manager.resync(pooled, reason=verdict)
            budget = _effective_timeout(timeout, self._default_timeout)
            started = time.monotonic()
            text, reasoning, result = self._run_with_retry(pooled, turn, budget)
        finally:
            with self._turn_lock:
                self._turn_started_at = None
        elapsed = time.monotonic() - started
        content = text
        if result.loop_tripped:
            reason = next((str(n) for n in result.notes if str(n).startswith("tool loop")), "tool limit reached")
            content = (content + "\n\n" if content.strip() else "") + (
                f"[agy_acp] Tool-loop breaker stopped this turn: {reason}. "
                f"Events: {Path(self.settings.log_dir or '.')}/events-{sid}.jsonl")
        elif result.cancelled and not content.strip():
            content = "(turn cancelled before the agent produced any text)"
        elif not content.strip():
            content = f"(agy returned no text; stop_reason={result.stop_reason or 'unknown'})"
        message = SimpleNamespace(content=content, tool_calls=None, reasoning=reasoning or None,
                                  reasoning_content=reasoning or None, reasoning_details=None)
        completion = SimpleNamespace(
            id=f"agy-acp-{sid[:8]}-{pooled.turn_seq}",
            object="chat.completion",
            created=int(time.time()),
            choices=[SimpleNamespace(message=message, finish_reason="stop", index=0)],
            usage=_usage(result.usage),
            model=model or pooled.model or self.placeholder_model,
            hermes_elapsed_seconds=round(elapsed, 2),
        )
        logger.info("agy_acp: turn ok session=%s elapsed=%.1fs chars=%d tools=%d stop=%s", sid, elapsed,
                    len(content), result.tool_calls, result.stop_reason)
        return _stream_chunks(completion) if stream else completion

    def _next_turn_number(self, sid: str) -> int:
        return self._manager.turn_seq(sid) + 1

    def _run_with_retry(self, pooled: Any, turn: Any, budget: float) -> tuple[str, str, Any]:
        def cancelled() -> bool:
            with self._turn_lock:
                return self._cancel_pending

        try:
            result = self._manager.run_turn(pooled, turn, timeout=budget, abort_if=cancelled)
            return result.text, result.reasoning, result
        except AcpError as exc:
            if not (exc.rate_limited and self.settings.retry_429):
                raise
            logger.warning("agy_acp: rate limited; retrying once in %.0fs", self.settings.retry_429_delay)
            time.sleep(max(0.0, self.settings.retry_429_delay))
            result = self._manager.run_turn(pooled, turn, timeout=budget, abort_if=cancelled)
            return result.text, result.reasoning, result
        except (TurnBusyError, LoadFailedError, NonlinearTurnError):
            raise

    # ── stop paths ───────────────────────────────────────────────────
    def cancel(self) -> bool:
        """Ask the current turn of this chat to stop (millisecond path, no process teardown).

        Three windows are covered, because Hermes' interrupt loop calls this from a *stranger*
        thread at any moment: a live turn (``session/cancel`` sentinel), a turn that is still
        building its session (a pending flag that makes ``run_turn`` abort before sending), and no
        turn at all (``False``).
        """
        sid = self._last_session or current_session_id()
        if self._manager.cancel(sid):
            return True
        with self._turn_lock:
            started = self._turn_started_at
            if started is not None and (time.monotonic() - started) < self.settings.turn_timeout:
                self._cancel_pending = True
                return True
        return False

    def close(self) -> None:
        """Hermes' shutdown/teardown path: cancel the turn, then close this chat's child tree."""
        self.is_closed = True
        try:
            sid = self._last_session or current_session_id()
            self._manager.cancel(sid)
            self._manager.close_session(sid, reason="client close")
        except Exception:
            logger.debug("agy_acp: close failed", exc_info=True)

    def __enter__(self) -> "AgyAcpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _WithReason:
    """``TurnResult`` plus the breaker text, kept out of the dataclass's public shape."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)

    @property
    def counter_reason(self) -> str:
        for note in getattr(self._result, "notes", []):
            if str(note).startswith("tool loop"):
                return str(note)
        return ""
