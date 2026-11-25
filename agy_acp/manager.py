"""Process pool, turn lock and turn execution for the stateful ``agy_acp`` provider.

Three invariants decide every detail here:

1. **One live turn per account.** ``agy`` drives a single consumer subscription; parallel prompts are
   the fastest way to a 429. A ``Semaphore(1)`` gate serializes turns process-wide, and is released
   while a permission dialog waits for its 90 s answer so one user's dialog cannot starve a chat.
2. **No history replay, ever.** The pool maps ``hermes_session_id -> AgyProcessHandle`` with an idle
   timeout and an LRU cap; a restarted process re-attaches with ``session/load``. If ``load`` fails
   the provider raises and names the two ways forward — it never silently resends history.
3. **Every wait is bounded and every child is reaped.** ``stopReason``, permission timeouts, turn
   timeouts, tool-loop breaker and ``session/cancel`` all funnel into the same teardown path.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
import time
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import fs, proctree
from .display import EventLog, ToolCounter, summarize_update
from .mcp_mount import mount as mount_bridge
from .permissions import PermissionBroker, check_native_command
from .protocol import AgyConnection, AcpError
from .prompt import TurnPrompt
from .settings import Settings
from .store import SessionStore
from .trace import TraceWriter

logger = logging.getLogger(__name__)

MODEL_LEVELS = ("xhigh", "high", "medium", "low", "max")


class NonlinearTurnError(RuntimeError):
    """The transcript moved backwards (regenerate / history edit): the agent's state would fork."""


class LoadFailedError(RuntimeError):
    """``session/load`` could not restore the agent's context; explicit choice required."""


class TurnBusyError(RuntimeError):
    """Another turn held the account gate for the whole timeout."""


@dataclass
class TurnResult:
    text: str = ""
    reasoning: str = ""
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    loop_tripped: bool = False
    tool_calls: int = 0
    notes: list[str] = field(default_factory=list)
    elapsed: float = 0.0


class TurnGate:
    """Account-wide turn gate with ``capacity`` permits (``AGY_ACP_MAX_CONCURRENT``).

    With capacity 1 the old behaviour holds: a permission wait hands the permit back, so one stuck
    dialog cannot starve another chat, and ``resume`` takes it again. With capacity > 1 the spare
    permits already cover that case, so ``pause``/``resume`` are no-ops — a waiting approval simply
    occupies one of them.
    """

    def __init__(self, capacity: int = 1) -> None:
        self.capacity = max(1, int(capacity))
        self._sem = threading.BoundedSemaphore(self.capacity)
        self._lock = threading.Lock()
        self._holders: set[int] = set()

    @property
    def held(self) -> bool:
        with self._lock:
            return bool(self._holders)

    def acquire(self, timeout: float | None) -> bool:
        if not self._sem.acquire(timeout=timeout):
            return False
        with self._lock:
            self._holders.add(threading.get_ident())
        return True

    def release(self) -> None:
        with self._lock:
            ident = threading.get_ident()
            if ident not in self._holders:
                return
            self._holders.discard(ident)
        try:
            self._sem.release()
        except ValueError:
            pass

    def pause(self) -> None:
        if self.capacity > 1:
            return
        with self._lock:
            ident = threading.get_ident()
            if ident not in self._holders:
                return
            self._holders.discard(ident)
        try:
            self._sem.release()
        except ValueError:
            pass

    def resume(self, timeout: float = 120.0) -> bool:
        if self.capacity > 1:
            return True
        if not self._sem.acquire(timeout=timeout):
            return False
        with self._lock:
            self._holders.add(threading.get_ident())
        return True


@dataclass
class TurnResult:
    text: str = ""
    reasoning: str = ""
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    loop_tripped: bool = False
    tool_calls: int = 0
    notes: list[str] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class TurnState:
    hermes_session_id: str
    acp_session_id: str
    counter: ToolCounter
    approval_context: Context = field(default_factory=copy_context, repr=False)
    started_at: float = field(default_factory=time.monotonic)
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    cancelled: bool = False
    loop_tripped: bool = False
    cancel_sent: bool = False
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    tool_updates: int = 0
    notes: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def feed(self, update: dict[str, Any]) -> None:
        kind = str(update.get("sessionUpdate") or "").strip()
        content = update.get("content")
        chunk = str(content.get("text") or "") if isinstance(content, dict) else ""
        if kind == "agent_message_chunk" and chunk:
            self.text.append(chunk)
        elif kind == "agent_thought_chunk" and chunk:
            self.reasoning.append(chunk)
        elif kind in ("tool_call", "tool_call_update"):
            self.tool_updates += 1
            self.counter.observe(is_start=kind == "tool_call")


@dataclass
class PooledSession:
    hermes_session_id: str
    cwd: str
    model: str
    conn: AgyConnection
    acp_session_id: str
    store: SessionStore
    broker: PermissionBroker
    events: EventLog
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    turn_seq: int = 0
    active_turn: TurnState | None = None
    system_prompt_sent: bool = False
    config_options: list[dict[str, Any]] = field(default_factory=list)
    work_dir: str = ""
    log: Callable[[dict[str, Any]], None] = lambda event: None

    def busy(self) -> bool:
        return self.active_turn is not None


class AgyAcpSessionManager:
    """Owns every live ``agy acp`` child of this process."""

    def __init__(self, settings: Settings | None = None, *, store: SessionStore | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_dirs()
        self.store = store or SessionStore(self.settings.state_db)  # type: ignore[arg-type]
        self.gate = TurnGate(settings.max_concurrent_turns)
        self._pool: dict[str, PooledSession] = {}
        self._lock = threading.RLock()
        # Held while a session is being spawned/handshaked. Deliberately NOT ``_lock``: the child
        # streams notifications (session_info_update, available_commands_update) *before* answering
        # session/load, and the reader thread needs the pool lock to route them. Holding one lock for
        # both is a guaranteed 120 s deadlock against the real CLI.
        self._create_lock = threading.Lock()
        self._reaper: threading.Thread | None = None
        self._stop = threading.Event()
        self._log = logging.getLogger("agy_acp.events")

    # ── public API ───────────────────────────────────────────────────
    def session_for(self, hermes_session_id: str, *, cwd: str, model: str = "") -> PooledSession:
        """Return the live session for ``hermes_session_id``, creating or restoring it if needed."""
        model = model or self.settings.model
        with self._lock:
            pooled = self._pool.get(hermes_session_id)
            if pooled is not None and pooled.conn.alive and pooled.cwd == cwd:
                pooled.last_used = time.monotonic()
                if model and pooled.model != model:
                    # The user picked another model in Hermes: apply it to the live native session
                    # instead of silently keeping the one it was created with.
                    self._select_model(pooled, model)
                return pooled
            stale = pooled is not None
        if stale:
            self._discard(hermes_session_id, reason="stale process or cwd changed")
        with self._create_lock:
            self._ensure_capacity_locked()
            with self._lock:  # another thread may have created it while we waited
                pooled = self._pool.get(hermes_session_id)
                if pooled is not None and pooled.conn.alive and pooled.cwd == cwd:
                    pooled.last_used = time.monotonic()
                    if model and pooled.model != model:
                        self._select_model(pooled, model)
                    return pooled
            pooled = self._create(hermes_session_id, cwd=cwd, model=model)
            with self._lock:
                self._pool[hermes_session_id] = pooled
                self._start_reaper_locked()
            return pooled

    def run_turn(self, pooled: PooledSession, prompt: TurnPrompt, *, timeout: float | None = None,
                 abort_if: Callable[[], bool] | None = None) -> TurnResult:
        settings = self.settings
        budget = timeout if timeout is not None else settings.turn_timeout
        if not self.gate.acquire(timeout=budget):
            raise TurnBusyError(
                f"agy_acp: all {self.gate.capacity} turn slots on this account are busy; this request "
                f"waited {budget:.0f}s and gave up without sending anything. Raise "
                "AGY_ACP_MAX_CONCURRENT (or agy_acp.max_concurrent) if the account tolerates it."
            )
        if abort_if is not None and abort_if():
            # Someone pressed stop while the session was still spinning up: honour it *before* the
            # prompt reaches the agent, so a cancelled turn costs nothing.
            self.gate.release()
            pooled.events.write({"event": "turn.cancelled_before_send", "session": pooled.hermes_session_id})
            return TurnResult(cancelled=True, stop_reason="cancelled",
                              notes=["cancelled before the request was sent"])
        turn = TurnState(
            hermes_session_id=pooled.hermes_session_id,
            acp_session_id=pooled.acp_session_id,
            counter=ToolCounter(limit=settings.tool_loop_limit, seconds=settings.tool_loop_seconds),
        )
        pooled.turn_seq += 1
        cancel_path = Path(settings.cancel_dir) / f"{pooled.hermes_session_id}.cancel"
        with contextlib.suppress(OSError):
            cancel_path.unlink()
        pooled.active_turn = turn
        pooled.last_used = time.monotonic()
        started = time.monotonic()
        with contextlib.suppress(Exception):
            self.store.begin_inflight(pooled.hermes_session_id, user_hash=prompt.user_hash)
        try:
            params: dict[str, Any] = {"sessionId": pooled.acp_session_id, "prompt": prompt.blocks}
            self._log_event(pooled, {"event": "turn.start", "turn": pooled.turn_seq,
                                     "chars": len(prompt.text), "blocks": len(prompt.blocks),
                                     "notes": prompt.notes,
                                     "tool_loop_limit": settings.tool_loop_limit,
                                     "tool_loop_seconds": settings.tool_loop_seconds,
                                     "prompt_timeout_seconds": budget})
            for note in prompt.notes:
                if note not in turn.notes:
                    turn.notes.append(note)
            tick = self._make_tick(pooled, turn, cancel_path)
            result = pooled.conn.request("session/prompt", params, timeout=budget, tick=tick)
            turn.stop_reason = str(result.get("stopReason") or result.get("stop_reason") or "end_turn")
            if isinstance(result.get("usage"), dict):
                turn.usage = dict(result["usage"])
            if turn.counter.tripped:
                turn.loop_tripped = True
                turn.notes.append(turn.counter.tripped_reason)
            if turn.cancel_sent or turn.stop_reason == "cancelled":
                turn.cancelled = True
                turn.notes.append("turn cancelled by user")
            self.store.record_turn(pooled.hermes_session_id, prompt.user_hash)
            if prompt.new_memory_keys:
                self.store.add_injected_ids(pooled.hermes_session_id, prompt.new_memory_keys)
            self.store.touch(pooled.hermes_session_id)
            return TurnResult(
                text="".join(turn.text), reasoning="".join(turn.reasoning), stop_reason=turn.stop_reason,
                usage=turn.usage, cancelled=turn.cancelled, loop_tripped=turn.loop_tripped,
                tool_calls=turn.counter.count, notes=turn.notes, elapsed=time.monotonic() - started,
            )
        except AcpError as exc:
            self._handle_turn_error(pooled, exc)
            raise
        finally:
            pooled.active_turn = None
            with contextlib.suppress(Exception):
                self.store.end_inflight(pooled.hermes_session_id)
            with contextlib.suppress(OSError):
                cancel_path.unlink()
            self._log_event(pooled, {"event": "turn.end", "turn": pooled.turn_seq,
                                     "stop": turn.stop_reason, "cancelled": turn.cancelled,
                                     "tools": turn.counter.count,
                                     "approval_wait_seconds": round(turn.counter.approval_wait_seconds, 2),
                                     "elapsed": round(time.monotonic() - started, 2)})
            self.gate.release()

    def take_interrupted(self, hermes_session_id: str) -> dict[str, Any] | None:
        """A marker left by a turn that never finished (provider restart): read once, then cleared."""
        if str(self.settings.on_interrupted or "").lower() == "ignore":
            self.store.end_inflight(hermes_session_id)
            return None
        return self.store.take_interrupted(hermes_session_id)

    def busy(self) -> bool:
        """True while a turn is running (used before rebuilding a manager on new settings)."""
        if self.gate.held:
            return True
        with self._lock:
            return any(pooled.active_turn is not None for pooled in self._pool.values())

    def inflight(self) -> list[dict[str, Any]]:
        return self.store.inflight_sessions()

    def cancel(self, hermes_session_id: str) -> bool:
        """Ask the running turn of ``hermes_session_id`` to stop (used by the CLI / UI)."""
        with self._lock:
            pooled = self._pool.get(hermes_session_id)
        if pooled is None or pooled.active_turn is None:
            return False
        path = Path(self.settings.cancel_dir) / f"{hermes_session_id}.cancel"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(time.time()), encoding="utf-8")
        return True

    def close_session(self, hermes_session_id: str, *, reason: str = "") -> None:
        pooled = self._take(hermes_session_id)
        if pooled is not None:
            self._teardown(pooled, reason=reason or "closed")

    def close_all(self, *, reason: str = "shutdown") -> None:
        with self._lock:
            ids = list(self._pool)
        for sid in ids:
            self.close_session(sid, reason=reason)
        self._stop.set()

    def reap_idle(self) -> list[str]:
        """Close sessions idle longer than the configured timeout. Returns the closed ids."""
        cutoff = time.monotonic() - self.settings.idle_seconds
        with self._lock:
            victims = [sid for sid, pooled in self._pool.items()
                       if pooled.last_used < cutoff and not pooled.busy()]
        for sid in victims:
            self.close_session(sid, reason="idle timeout")
        if victims:
            logger.info("agy_acp: closed %d idle session(s)", len(victims))
        return victims

    def pool_state(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"session": pooled.hermes_session_id, "acp_session": pooled.acp_session_id,
                     "pid": pooled.conn.pid, "busy": pooled.busy(),
                     "idle_seconds": round(time.monotonic() - pooled.last_used, 1)}
                    for pooled in self._pool.values()]

    def memory_keys(self, hermes_session_id: str) -> set[str]:
        return self.store.injected_ids(hermes_session_id)

    def check_linearity(self, hermes_session_id: str, user_hash: str) -> str:
        """``"ok"`` | ``"regenerate"`` | ``"rewind"`` for the incoming turn."""
        row = self.store.get(hermes_session_id)
        if not row or int(row.get("turn_count") or 0) == 0:
            return "ok"
        if user_hash and user_hash == (row.get("last_user_hash") or ""):
            return "regenerate"
        if user_hash and user_hash == (row.get("prev_user_hash") or ""):
            return "rewind"
        return "ok"

    def raise_nonlinear(self, hermes_session_id: str, verdict: str) -> None:
        """Raise the explicit "cannot rewind a stateful session" error."""
        what = "a regenerated turn" if verdict == "regenerate" else "an edited earlier turn"
        raise NonlinearTurnError(
            f"agy_acp: this request re-sends {what} that the agent has already answered natively "
            f"(session {hermes_session_id}). A stateful session cannot rewind — replaying it would double-charge "
            "quota and desynchronize the two transcripts. Start a new chat to ask again, or set "
            "AGY_ACP_ON_NONLINEAR=resync to open a fresh native session automatically (the agent's in-context "
            "history is dropped, which is what makes it safe)."
        )

    def resync(self, pooled: PooledSession, *, reason: str) -> None:
        """Open a brand-new native session for this Hermes chat (explicit opt-in path)."""
        self._resync(pooled, reason=reason)

    def turn_seq(self, hermes_session_id: str) -> int:
        with self._lock:
            pooled = self._pool.get(hermes_session_id)
        return pooled.turn_seq if pooled is not None else 0

    # ── creation ─────────────────────────────────────────────────────
    def _create(self, hermes_session_id: str, *, cwd: str, model: str) -> PooledSession:
        settings = self.settings
        Path(cwd).mkdir(parents=True, exist_ok=True)
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", hermes_session_id)[:40] or "session"
        events = EventLog(Path(settings.log_dir or ".") / f"events-{label}.jsonl",
                          max_bytes=settings.event_log_max_bytes)
        holder: dict[str, PooledSession] = {}

        def on_notification(message: dict[str, Any]) -> None:
            self._route_update(message, holder)

        def on_request(message: dict[str, Any]) -> dict[str, Any] | None:
            return self._route_request(message, holder)

        conn = AgyConnection(settings, cwd=cwd, on_notification=on_notification,
                             on_server_request=on_request, label=label,
                             trace=(TraceWriter(Path(settings.trace_dir) / f"{label}.ndjson")
                                    if settings.enable_ndjson_trace and settings.trace_dir else None))
        conn.start()
        row = self.store.get(hermes_session_id)
        session_id = ""
        restored = False
        options: list[dict[str, Any]] = []
        if row and row.get("acp_session_id") and self._supports_load(conn):
            try:
                # The bridge must ride the restore path too, or a chat that survived a restart silently
                # loses hermes_memory while its prompt still describes the tool.
                loaded = conn.request("session/load",
                                      {"sessionId": row["acp_session_id"], "cwd": cwd,
                                       "mcpServers": mount_bridge(settings, session_id=hermes_session_id)},
                                      timeout=settings.connect_timeout)
                session_id = str(loaded.get("sessionId") or "") or row["acp_session_id"]
                options = [o for o in (loaded.get("configOptions") or []) if isinstance(o, dict)]
                restored = True
                events.write({"event": "session.loaded", "acp_session": session_id})
            except AcpError as exc:
                if settings.on_load_failure != "reset":
                    conn.close("load failed")
                    raise LoadFailedError(
                        f"agy_acp: the agent's native session {row['acp_session_id']} could not be restored "
                        f"({exc}). Hermes will not replay the conversation into a fresh session — that is exactly "
                        "the token burn this provider exists to avoid. Choose explicitly: start a new chat, or set "
                        "AGY_ACP_ON_LOAD_FAILURE=reset to start a fresh native session automatically."
                    ) from exc
                events.write({"event": "session.load_failed", "error": str(exc)[:300], "action": "reset"})
        if not session_id:
            params: dict[str, Any] = {"cwd": cwd, "mcpServers": mount_bridge(settings, session_id=hermes_session_id)}
            if settings.system_prompt:
                params["_meta"] = {"antigravity": {"systemPrompt": {"append": settings.system_prompt}}}
            created = conn.request("session/new", params, timeout=settings.connect_timeout)
            session_id = str(created.get("sessionId") or "")
            options = [o for o in (created.get("configOptions") or []) if isinstance(o, dict)]
            if not session_id:
                conn.close("session/new returned no id")
                raise AcpError("agy_acp: session/new returned no sessionId", kind="server")
            events.write({"event": "session.new", "acp_session": session_id,
                          "child_home": settings.child_home, "proxy": bool(settings.proxy),
                          "bin": settings.command, "home": str(settings.home),
                          "capability_mode": settings.capability_mode})
        native_gate = self._native_gate()
        broker = PermissionBroker(
            permission_dir=Path(settings.permission_dir),
            timeout=settings.permission_timeout,
            auto_allow_readonly=settings.auto_allow_readonly,
            auto_allow_bridge=settings.auto_allow_bridge,
            bridge_policy_name=getattr(settings, "bridge_approval", "all"),
            bridge_prefix=settings.bridge_prefix,
            capability_mode=settings.capability_mode,
            write_timeout=min(settings.approval_wait or settings.permission_timeout,
                              settings.turn_timeout),
            native_gate=native_gate,
            command_guard=check_native_command if native_gate is not None else None,
            on_wait_start=self.gate.pause,
            on_wait_end=self._resume_gate,
            log=events.write,
        )
        pooled = PooledSession(
            hermes_session_id=hermes_session_id, cwd=cwd, model=model, conn=conn,
            acp_session_id=session_id, store=self.store, broker=broker, events=events,
            system_prompt_sent=restored, config_options=options, work_dir=str(settings.work_dir),
        )
        pooled.log = events.write
        holder["session"] = pooled
        self.store.upsert(hermes_session_id, acp_session_id=session_id, cwd=cwd, model=model)
        if model:
            self._select_model(pooled, model)
        return pooled

    def _resume_gate(self) -> None:
        if not self.gate.resume(timeout=max(5.0, self.settings.permission_timeout)):
            logger.warning("agy_acp: could not re-acquire the turn gate after a permission wait")

    def _supports_load(self, conn: AgyConnection) -> bool:
        caps = conn.capabilities or {}
        session_caps = caps.get("sessionCapabilities") or caps.get("session_capabilities") or {}
        return bool(caps.get("loadSession") or caps.get("load_session") or "load" in session_caps)

    @staticmethod
    def _split_model_level(model: str) -> tuple[str, str]:
        for level in MODEL_LEVELS:
            suffix = f"-{level}"
            if model.endswith(suffix):
                return model[: -len(suffix)], level
        return model, ""

    @staticmethod
    def _option_by(options: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        """First option whose ``category`` or ``id`` mentions *key* (substring, not equality).

        ``"thought" in (category, id)`` — the natural-looking form — is a tuple membership test and
        silently returns None for ``("", "thought_level")``; that bug shipped a turn at the account's
        default thought level instead of the requested one.
        """
        for option in options:
            haystacks = (str(option.get("category") or "").lower(), str(option.get("id") or "").lower())
            if any(key in hay for hay in haystacks):
                return option
        return None

    def _select_model(self, pooled: PooledSession, model: str) -> None:
        """Apply the requested model through ``session/set_config_option`` (never a spawn flag)."""
        conn = pooled.conn
        options = pooled.config_options
        model_opt = self._option_by(options, "model")
        level_opt = self._option_by(options, "thought")
        base, level = self._split_model_level(model)
        values = [str(o.get("value")) for o in (model_opt or {}).get("options") or [] if isinstance(o, dict)]
        target = model if model in values or not values else (base if base in values else "")
        logger.debug("agy_acp: model selection requested=%r base=%r level=%r target=%r offered=%s",
                     model, base, level, target, values)
        if not target:
            logger.warning("agy_acp: model %r is not offered by this account; using the session default", model)
            return
        try:
            if model_opt and str(model_opt.get("currentValue") or "") != target:
                applied = conn.request("session/set_config_option",
                                       {"sessionId": pooled.acp_session_id,
                                        "configId": str(model_opt.get("id") or "model"), "value": target},
                                       timeout=60)
                fresh = [o for o in (applied.get("configOptions") or []) if isinstance(o, dict)]
                for option in fresh:  # keep the cached view honest for the next comparison
                    for cached in pooled.config_options or []:
                        if str(cached.get("id")) == str(option.get("id")):
                            cached["currentValue"] = option.get("currentValue")
                reported = [o for o in fresh if str(o.get("id")) == str(model_opt.get("id") or "model")]
                logger.debug("agy_acp: model set -> %s", [o.get("currentValue") for o in reported])
            if level and level_opt:
                levels = [str(o.get("value")) for o in level_opt.get("options") or [] if isinstance(o, dict)]
                if level in levels and str(level_opt.get("currentValue") or "") != level:
                    applied = conn.request("session/set_config_option",
                                           {"sessionId": pooled.acp_session_id,
                                            "configId": str(level_opt.get("id") or "thought_level"),
                                            "value": level}, timeout=60)
                    fresh = [o for o in (applied.get("configOptions") or []) if isinstance(o, dict)]
                    for option in fresh:
                        for cached in pooled.config_options or []:
                            if str(cached.get("id")) == str(option.get("id")):
                                cached["currentValue"] = option.get("currentValue")
                    reported = [o for o in fresh
                                if str(o.get("id")) == str(level_opt.get("id") or "thought_level")]
                    logger.debug("agy_acp: thought level %r -> %s", level,
                                 [o.get("currentValue") for o in reported])
                    if reported and str(reported[0].get("currentValue")) != level:
                        logger.warning("agy_acp: the agent did not accept thought level %r (it reports %r)",
                                       level, reported[0].get("currentValue"))
                elif level not in levels:
                    logger.info("agy_acp: %r offers no thought level %r (has %s); using its default",
                                target, level, levels)
            pooled.model = target + (f"-{level}" if level else "")
            self.store.touch(pooled.hermes_session_id, model=pooled.model)
        except (AcpError, KeyError, TypeError) as exc:
            logger.warning("agy_acp: model selection for %r failed; using the session default: %s", model, exc)

    # ── routing ──────────────────────────────────────────────────────
    def _find_by_acp_id(self, holder: dict[str, PooledSession], acp_session_id: str) -> PooledSession | None:
        pooled = holder.get("session")
        if pooled is not None and pooled.acp_session_id == acp_session_id:
            return pooled
        with self._lock:
            for candidate in self._pool.values():
                if candidate.acp_session_id == acp_session_id:
                    return candidate
        return None

    def _route_update(self, message: dict[str, Any], holder: dict[str, PooledSession]) -> None:
        params = message.get("params") or {}
        pooled = self._find_by_acp_id(holder, str(params.get("sessionId") or ""))
        if pooled is None:
            return
        update = params.get("update") or {}
        if not isinstance(update, dict):
            return
        turn = pooled.active_turn
        if turn is not None:
            turn.feed(update)
        kind = str(update.get("sessionUpdate") or "")
        if kind in ("tool_call", "tool_call_update", "plan") or turn is None:
            pooled.events.write({
                "event": "session.update",
                "tool": summarize_update(update, head=self.settings.display_head, tail=self.settings.display_tail),
            })

    def _route_request(self, message: dict[str, Any], holder: dict[str, PooledSession]) -> dict[str, Any] | None:
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        pooled = self._find_by_acp_id(holder, str(params.get("sessionId") or ""))
        if method == "session/request_permission":
            turn = pooled.active_turn if pooled is not None else None
            if turn is None:
                return {"jsonrpc": "2.0", "id": message.get("id"),
                        "result": {"outcome": {"outcome": "cancelled"}}}
            # The ACP reader is separate from the Hermes turn that owns approval routing.
            started = time.monotonic()
            try:
                return turn.approval_context.run(pooled.broker.handle, message,
                                                 session_id=pooled.hermes_session_id)
            finally:
                turn.counter.approval_wait_seconds += time.monotonic() - started
        handler = fs.HANDLERS.get(method)
        if handler is not None and pooled is not None:
            try:
                return {"jsonrpc": "2.0", "id": message.get("id"), "result": handler(params, pooled.cwd)}
            except Exception as exc:
                return {"jsonrpc": "2.0", "id": message.get("id"),
                        "error": {"code": -32602, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": message.get("id"),
                "error": {"code": -32601, "message": f"agy_acp does not implement '{method}'"}}

    # ── turn internals ───────────────────────────────────────────────
    def _make_tick(self, pooled: PooledSession, turn: TurnState, cancel_path: Path) -> Callable[[], bool]:
        def tick() -> bool:
            if turn.cancel_sent:
                return True
            if cancel_path.exists():
                with contextlib.suppress(OSError):
                    cancel_path.unlink()
                if pooled.conn.cancel(pooled.acp_session_id):
                    turn.cancel_sent = True
                    turn.notes.append("session/cancel sent")
                    pooled.log({"event": "turn.cancel", "session": pooled.hermes_session_id})
                return True
            if turn.counter.tripped and self.settings.tool_loop_action == "cancel" and not turn.loop_tripped:
                turn.loop_tripped = True
                turn.cancel_sent = True
                pooled.conn.cancel(pooled.acp_session_id)
                pooled.log({"event": "turn.loop_breaker", "reason": turn.counter.tripped_reason})
                return True
            return False

        return tick

    def _handle_turn_error(self, pooled: PooledSession, exc: AcpError) -> None:
        pooled.log({"event": "turn.error", "kind": exc.kind, "method": exc.method, "error": str(exc)[:1000]})
        if exc.kind in ("transport", "timeout") or not pooled.conn.alive:
            self.close_session(pooled.hermes_session_id, reason=f"transport: {exc.kind}")

    def _resync(self, pooled: PooledSession, *, reason: str) -> None:
        """Open a brand-new native session for the same Hermes chat (explicit user opt-in)."""
        params: dict[str, Any] = {"cwd": pooled.cwd,
                                  "mcpServers": mount_bridge(self.settings, session_id=pooled.hermes_session_id)}
        if self.settings.system_prompt:
            params["_meta"] = {"antigravity": {"systemPrompt": {"append": self.settings.system_prompt}}}
        created = pooled.conn.request("session/new", params, timeout=self.settings.connect_timeout)
        session_id = str(created.get("sessionId") or "")
        if not session_id:
            raise AcpError("agy_acp: session/new returned no sessionId during resync", kind="server")
        pooled.acp_session_id = session_id
        pooled.system_prompt_sent = True
        pooled.store.rebind(pooled.hermes_session_id, session_id)
        pooled.log({"event": "session.resync", "reason": reason, "acp_session": session_id})

    @staticmethod
    def _native_gate():
        """Hermes' own approval gate, or ``None`` when this install cannot offer one.

        ``tools.approval.request_tool_approval`` is the documented entry for plugins: it asks the same
        human gate as Tier-2 shell commands (CLI prompt / gateway approval card), keeps session and
        permanent answers under ``plugin_rule:<rule_key>`` so a repeat is not asked twice, and fails
        closed when nobody is there. Imported lazily: the plugin also runs outside Hermes.
        """
        mode = str(os.environ.get("AGY_ACP_APPROVAL_BACKEND") or "auto").strip().lower()
        if mode in ("file", "off", "none"):
            return None
        try:
            from tools.approval import request_tool_approval
        except Exception:  # noqa: BLE001 - standalone use has no Hermes core on sys.path
            if mode == "hermes":
                logger.warning("agy_acp: AGY_ACP_APPROVAL_BACKEND=hermes but tools.approval is not "
                               "importable; falling back to the file inbox")
            return None

        def gate(tool_display: str, reason: str, rule_key: str) -> dict[str, Any]:
            return request_tool_approval(tool_display, reason, rule_key=rule_key)
        return gate

    def _log_event(self, pooled: PooledSession, event: dict[str, Any]) -> None:
        pooled.events.write(event)

    # ── teardown / capacity ──────────────────────────────────────────
    def _take(self, hermes_session_id: str) -> PooledSession | None:
        with self._lock:
            return self._pool.pop(hermes_session_id, None)

    def _discard(self, hermes_session_id: str, *, reason: str) -> None:
        pooled = self._take(hermes_session_id)
        if pooled is not None:
            self._teardown(pooled, reason=reason)

    def _teardown(self, pooled: PooledSession, *, reason: str) -> None:
        """Close one child's process tree. Runs WITHOUT the pool lock (killing a tree takes seconds)."""
        if pooled.active_turn is not None:
            pooled.conn.cancel(pooled.acp_session_id)
        pooled.events.write({"event": "session.closed", "reason": reason})
        pooled.conn.close(reason, session_id="")  # terminate the tree; the transcript stays loadable
        self._cleanup_spill(pooled)
        self.store.touch(pooled.hermes_session_id)

    @staticmethod
    def _cleanup_spill(pooled: PooledSession) -> None:
        """Remove this session's spilled turn files (workspace copy first, then the home copy)."""
        import shutil

        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", pooled.hermes_session_id or "session")[:64]
        candidates = [Path(pooled.cwd) / ".hermes_tmp" / "agy_acp" / safe]
        if pooled.work_dir:
            candidates.append(Path(pooled.work_dir) / safe)
        for folder in candidates:
            shutil.rmtree(folder, ignore_errors=True)
            parent = folder.parent
            with contextlib.suppress(OSError):
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()

    def _ensure_capacity_locked(self) -> None:
        """Evict the least-recently-used idle child **without** holding the pool lock while killing it."""
        while True:
            with self._lock:
                if len(self._pool) < self.settings.max_sessions:
                    return
                candidate = next((p for p in sorted(self._pool.values(), key=lambda item: item.last_used)
                                  if not p.busy()), None)
            if candidate is None:  # everything is mid-turn: over capacity for now, never kill a live turn
                logger.warning("agy_acp: all %d sessions are busy; allowing one extra child",
                               len(self._pool))
                return
            logger.info("agy_acp: LRU eviction of %s", candidate.hermes_session_id)
            self._discard(candidate.hermes_session_id, reason="LRU eviction")

    def _start_reaper_locked(self) -> None:
        if self._reaper is not None and self._reaper.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(30.0):
                try:
                    self.reap_idle()
                except Exception:
                    logger.debug("agy_acp: idle reaper failed", exc_info=True)

        self._reaper = threading.Thread(target=loop, name="agy-acp-reaper", daemon=True)
        self._reaper.start()

    def __enter__(self) -> "AgyAcpSessionManager":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close_all()
