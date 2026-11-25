"""Permission handling that cannot deadlock a turn.

ACP lets the agent ask the client before it runs a tool (``session/request_permission``). A provider
with no human attached has two failure modes: answering "allow" (an unattended agent edits files)
or never answering (the turn hangs until the transport timeout, holding the account's concurrency
slot the whole time). This module implements the third option:

* read-only tools are auto-allowed (opt out with ``AGY_ACP_AUTO_ALLOW_READONLY=0``);
* supervised shell commands use Hermes' terminal guard when available;
* other escalations use the Hermes tool gate or a **request file** under
  ``$HERMES_HOME/agy_acp/permissions/``, answerable by the bundled CLI or an in-process callback;
* the global turn lock is released while the user is being asked, so one stuck dialog cannot starve
  the other chats;
* after ``AGY_ACP_PERMISSION_TIMEOUT`` (90 s default) the provider answers **deny** with
  ``"User permission timed out"``, releases its bookkeeping, and lets the turn fail cleanly.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TIMEOUT_REASON = "User permission timed out"

READONLY_KINDS = frozenset({
    "read", "read_file", "grep", "search", "glob", "list", "ls", "list_directory", "inspect",
    "fetch", "web_search", "web_fetch", "think",
})
READONLY_TOOLS = frozenset({
    "read_file", "grep", "ls", "glob", "search_files", "list_directory", "web_search", "web_fetch",
    "read_text_file", "read_multiple_files", "codebase_search",
})
ALLOW_KINDS = ("allow_once", "allow_always", "allow")
REJECT_KINDS = ("reject_once", "reject_always", "reject", "deny")


@dataclass
class Decision:
    allow: bool
    option_id: str | None = None
    reason: str = ""
    source: str = "policy"  # policy | user | timeout | callback


@dataclass
class PendingPermission:
    permission_id: str
    created_at: float
    request: dict[str, Any]
    session_id: str
    path: Path
    answer_path: Path
    event: threading.Event = field(default_factory=threading.Event)
    decision: Decision | None = None


def _option_by_kind(options: list[dict[str, Any]], wanted: tuple[str, ...]) -> dict[str, Any] | None:
    for option in options or []:
        if not isinstance(option, dict):
            continue
        kind = str(option.get("kind") or "").lower()
        if kind in wanted and option.get("optionId"):
            return option
    return None


#: The bridge's tools arrive as ``mcp_hermes-bridge_<tool>`` with ``kind: "other"``, so kind alone
#: cannot tell a search from a forget. Inspect the requested action instead.
BRIDGE_PREFIX = "mcp_hermes-bridge_"  # default server name; see broker.bridge_prefix
LEGACY_BRIDGE_PREFIXES = (BRIDGE_PREFIX,)
BRIDGE_READ_ACTIONS = frozenset({"search", "show", "list", "view", "status"})


#: How much of the bridge runs without asking. Hermes' own memory tools (scope-recall's
#: ``revise``/``forget``) have **no approval gate** — the model may change memory unattended — so
#: gating the bridge's memory actions would be stricter than the host it borrows the capability from.
#: ``all`` (default): the bridge never prompts. ``reads``: writes ask. ``none``: every action asks.
BRIDGE_POLICIES = ("all", "reads", "none")


def bridge_policy(explicit: str = "") -> str:
    """Effective policy: ``AGY_ACP_BRIDGE_APPROVAL``, with the older boolean honoured as ``none``."""
    value = (explicit or os.environ.get("AGY_ACP_BRIDGE_APPROVAL", "")).strip().lower()
    if value in BRIDGE_POLICIES:
        return value
    if os.environ.get("AGY_ACP_AUTO_ALLOW_BRIDGE", "").strip().lower() in ("0", "false", "no", "off"):
        return "none"
    return "all"


def bridge_read(tool: dict[str, Any], prefix: str = BRIDGE_PREFIX) -> bool | None:
    """``True`` for a bridge read, ``False`` for a bridge write, ``None`` when it is not a bridge call."""
    name = str(tool.get("name") or tool.get("toolName") or "").lower()
    prefixes = tuple({(prefix or BRIDGE_PREFIX).lower(), *LEGACY_BRIDGE_PREFIXES})
    if not any(name.startswith(p) for p in prefixes):
        return None
    return str(_tool_input(tool).get("action") or "").strip().lower() in BRIDGE_READ_ACTIONS


def is_readonly(tool: dict[str, Any]) -> bool:
    kind = str(tool.get("kind") or "").lower()
    name = str(tool.get("name") or tool.get("toolName") or "").lower()
    if name == "run_command":
        return False
    if kind and kind in READONLY_KINDS:
        return True
    return bool(name) and name in READONLY_TOOLS


def _tool_input(tool: dict[str, Any]) -> dict[str, Any]:
    raw = tool.get("rawInput")
    if raw is None:
        raw = tool.get("input")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    return raw if isinstance(raw, dict) else {}


def check_native_command(command: str) -> dict[str, Any] | None:
    """Use the host terminal guard without executing or rewriting the ACP command."""
    try:
        from tools.approval import (_presence, _unattended_contexts, check_all_command_guards,
                                    detect_dangerous_command, is_approval_bypass_active)
    except ImportError:
        return None

    verdict = check_all_command_guards(command, env_type="local")
    if (verdict.get("approved") is True and not verdict.get("pre_approved")
            and not is_approval_bypass_active()):
        # The terminal guard allows unbound headless callers by default. Keep ACP's
        # fail-closed behavior for flagged commands when there is no turn or policy.
        _, cli, gateway, ask = _presence()
        if not (cli or gateway or ask or _unattended_contexts()):
            flagged, _, description = detect_dangerous_command(command)
            if flagged:
                return {"approved": False, "description": f"No Hermes approval surface: {description}"}
    return verdict


def describe_for_human(tool: dict[str, Any], prefix: str = BRIDGE_PREFIX) -> tuple[str, str]:
    """(one sentence for the approval card, rule key for a session/always answer)."""
    name = str(tool.get("name") or tool.get("toolName") or "tool")
    raw = _tool_input(tool)
    action = str(raw.get("action") or "").strip().lower()
    kind = str(tool.get("kind") or "").strip().lower()
    matched = next((p for p in ((prefix or BRIDGE_PREFIX).lower(), *LEGACY_BRIDGE_PREFIXES)
                    if name.lower().startswith(p)), "")
    if matched:
        short = name[len(matched):]
        what = f"Hermes bridge: {short}" + (f" action={action}" if action else "")
        return f"agy wants to change Hermes memory ({what}). Allow it?", f"agy:{short}:{action or kind}"
    target = str(raw.get("CommandLine") or raw.get("Path") or "").strip()
    what = f"{name}" + (f" {target[:120]}" if target else "")
    rule_key = f"agy:{name}:{kind}"
    if name.lower() == "run_command":
        payload = json.dumps(raw, sort_keys=True, ensure_ascii=True)
        rule_key += ":" + hashlib.sha256(payload.encode("ascii")).hexdigest()
    return f"agy wants to run {what}. Allow it?", rule_key


class PermissionBroker:
    """Answers ``session/request_permission`` for one session."""

    def __init__(
        self,
        *,
        permission_dir: Path,
        timeout: float = 90.0,
        auto_allow_readonly: bool = True,
        auto_allow_bridge: bool = True,
        bridge_policy_name: str = "",
        bridge_prefix: str = BRIDGE_PREFIX,
        capability_mode: str = "supervised",
        write_timeout: float | None = None,
        on_wait_start: Callable[[], None] | None = None,
        on_wait_end: Callable[[], None] | None = None,
        handler: Callable[[dict[str, Any]], Decision | None] | None = None,
        native_gate: Callable[[str, str, str], dict[str, Any]] | None = None,
        command_guard: Callable[[str], dict[str, Any] | None] | None = None,
        poll_interval: float = 0.2,
        log: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.permission_dir = Path(permission_dir)
        self.auto_allow_readonly = auto_allow_readonly
        self.auto_allow_bridge = auto_allow_bridge
        self.bridge_policy = bridge_policy(bridge_policy_name)
        self.bridge_prefix = (bridge_prefix or BRIDGE_PREFIX).lower()
        #: T3 Code's idea, scaled down: read-only / supervised / full. Read once, never per turn,
        #: and nothing derived from it can escalate (a request either is or is not in the mode).
        self.capability_mode = (capability_mode or "supervised").strip().lower()
        #: Writes and command execution may wait longer than a read prompt: nobody has to be at the
        #: keyboard for the exact second the agent asks.
        self.timeout = timeout
        self.write_timeout = float(write_timeout) if write_timeout else timeout
        self.on_wait_start = on_wait_start
        self.on_wait_end = on_wait_end
        self.handler = handler
        #: Hermes' own approval gate (tools.approval.request_tool_approval) when reachable: it shows
        #: the desktop/CLI prompt, keeps session and permanent answers, and fails closed with no human.
        self.native_gate = native_gate
        self.command_guard = command_guard
        self.poll_interval = poll_interval
        self.log = log or (lambda event: None)
        self._pending: dict[str, PendingPermission] = {}
        self._lock = threading.Lock()

    # ── main entry ───────────────────────────────────────────────────
    def handle(self, message: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        """Build the JSON-RPC response for one permission request (never raises)."""
        params = message.get("params") or {}
        tool = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
        tool = tool or {}
        options = [o for o in (params.get("options") or []) if isinstance(o, dict)]
        name = str(tool.get("name") or tool.get("toolName") or "")
        kind = str(tool.get("kind") or "")
        bridge_kind = bridge_read(tool, self.bridge_prefix)
        if bridge_kind is not None and self.capability_mode == "readonly":
            # Read-only means "nothing changes without a human saying so" — not "changes are
            # impossible". A read runs; a write goes to the same gate/inbox as everything else.
            if bridge_kind is True:
                decision = Decision(True, self._pick(options, ALLOW_KINDS),
                                    "capability_mode=readonly allows bridge reads", "policy")
                self._note("permission.auto_allow", session_id, name, kind, decision, extra={"bridge": "read"})
                return self._respond(message, decision, options)
        if self.capability_mode == "full" and bridge_kind is not None:
            decision = Decision(True, self._pick(options, ALLOW_KINDS),
                                "capability_mode=full allows every bridge action", "policy")
            self._note("permission.auto_allow", session_id, name, kind, decision,
                       extra={"bridge": "read" if bridge_kind else "write"})
            return self._respond(message, decision, options)
        if self.auto_allow_bridge and self.capability_mode != "readonly":
            allowed_by_policy = (bridge_kind is True and self.bridge_policy in ("all", "reads")) or \
                                (bridge_kind is False and self.bridge_policy == "all")
            if allowed_by_policy:
                decision = Decision(True, self._pick(options, ALLOW_KINDS),
                                    f"hermes-bridge action auto-allowed (policy={self.bridge_policy})", "policy")
                self._note("permission.auto_allow", session_id, name, kind, decision,
                           extra={"bridge": "read" if bridge_kind else "write"})
                return self._respond(message, decision, options)
        if self.capability_mode == "full" and bridge_kind is None:
            decision = Decision(True, self._pick(options, ALLOW_KINDS),
                                "capability_mode=full allows agy's own actions", "policy")
            self._note("permission.auto_allow", session_id, name, kind, decision)
            return self._respond(message, decision, options)
        if self.auto_allow_readonly and is_readonly(tool):
            decision = Decision(True, self._pick(options, ALLOW_KINDS), "read-only tool auto-allowed", "policy")
            self._note("permission.auto_allow", session_id, name, kind, decision)
            return self._respond(message, decision, options)
        if (self.capability_mode == "supervised" and self.command_guard is not None
                and name == "run_command" and kind == "execute"):
            command = _tool_input(tool).get("CommandLine")
            if isinstance(command, str) and command.strip():
                try:
                    verdict = self.command_guard(command)
                except Exception:  # Guard failure is a denial, so it cannot be overridden by the inbox.
                    verdict = {"approved": False, "description": "Hermes command guard failed"}
                if verdict is not None:
                    allowed = verdict.get("approved") is True
                    # Hermes owns remembered approvals. Answer only for this ACP tool call.
                    current_options = [option for option in options
                                       if str(option.get("kind") or "").lower() not in
                                       ("allow_always", "reject_always")]
                    option_id = self._pick(current_options, ALLOW_KINDS if allowed else REJECT_KINDS)
                    reason = str(verdict.get("user_summary") or verdict.get("description") or "")[:300]
                    decision = Decision(allowed, option_id, reason, "hermes-command")
                    self._note("permission.decision", session_id, name, kind, decision,
                               extra={"outcome": verdict.get("outcome") or verdict.get("status"),
                                      "pattern_key": verdict.get("pattern_key"),
                                      "user_approved": verdict.get("user_approved", False),
                                      "smart_approved": verdict.get("smart_approved", False)})
                    return self._respond(message, decision, current_options)
        if self.native_gate is not None:
            reason, rule_key = describe_for_human(tool, self.bridge_prefix)
            self._note("permission.gate_request", session_id, name, kind, None,
                       extra={"rule_key": rule_key, "prompt": reason})
            verdict: dict[str, Any] | None
            try:
                verdict = self.native_gate(name, reason, rule_key)
            except Exception as exc:  # noqa: BLE001 - never lose the turn to a gate failure
                verdict = None
                self._note("permission.gate_failed", session_id, name, kind, None,
                           extra={"error": f"{type(exc).__name__}: {exc}"[:300]})
            # "approval_required"/"blocked" mean the gate found no human surface here (it fails closed
            # instead of prompting): that is not a denial — this plugin's own inbox is the surface then.
            if isinstance(verdict, dict) and (verdict.get("status") == "approval_required"
                                              or verdict.get("outcome") in ("blocked", "no_human")):
                self._note("permission.gate_no_surface", session_id, name, kind, None,
                           extra={"fallback": "file inbox"})
                verdict = None
            if isinstance(verdict, dict) and "approved" in verdict:
                allowed = bool(verdict.get("approved"))
                decision = Decision(allowed, self._pick(options, ALLOW_KINDS if allowed else REJECT_KINDS),
                                    str(verdict.get("user_summary") or verdict.get("message") or "")[:300],
                                    "hermes-gate")
                self._note("permission.decision", session_id, name, kind, decision,
                           extra={"outcome": verdict.get("outcome"), "rule_key": rule_key})
                return self._respond(message, decision, options)
        wait = self.write_timeout if not is_readonly(tool) else self.timeout
        pending = self._register(params, tool, session_id, timeout=wait)
        self._note("permission.ask", session_id, name, kind, None, extra={"id": pending.permission_id})
        if self.on_wait_start is not None:
            try:
                self.on_wait_start()
            except Exception:
                pass
        decision: Decision | None = None
        try:
            decision = self._await(pending, timeout=wait)
        finally:
            if self.on_wait_end is not None:
                try:
                    self.on_wait_end()
                except Exception:
                    pass
            self._cleanup(pending)
        self._note("permission.decision", session_id, name, kind, decision)
        return self._respond(message, decision, options)

    # ── internals ────────────────────────────────────────────────────
    def _register(self, params: dict[str, Any], tool: dict[str, Any], session_id: str,
                  *, timeout: float | None = None) -> PendingPermission:
        permission_id = f"perm-{uuid.uuid4().hex[:12]}"
        try:
            self.permission_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        path = self.permission_dir / f"{permission_id}.json"
        answer_path = self.permission_dir / f"{permission_id}.answer.json"
        request = {
            "id": permission_id,
            "session_id": session_id,
            "created_at": time.time(),
            "timeout_seconds": self.write_timeout if timeout is None else timeout,
            "tool": {
                "name": tool.get("name") or tool.get("toolName") or "",
                "kind": tool.get("kind") or "",
                "title": tool.get("title") or "",
                "toolCallId": tool.get("toolCallId") or "",
                "rawInput": tool.get("rawInput"),
            },
            "options": [
                {"optionId": o.get("optionId"), "name": o.get("name"), "kind": o.get("kind")}
                for o in (params.get("options") or []) if isinstance(o, dict)
            ],
        }
        pending = PendingPermission(permission_id, time.time(), request, session_id, path, answer_path)
        try:
            path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        with self._lock:
            self._pending[permission_id] = pending
        if self.handler is not None:
            def _run() -> None:
                try:
                    decision = self.handler(dict(request))
                except Exception:
                    decision = None
                if decision is not None:
                    self.answer(permission_id, decision)
            threading.Thread(target=_run, name=f"agy-acp-perm-{permission_id}", daemon=True).start()
        return pending

    def _await(self, pending: PendingPermission, *, timeout: float | None = None) -> Decision:
        budget = self.timeout if timeout is None else timeout
        deadline = pending.created_at + max(0.0, budget)
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return Decision(False, None, TIMEOUT_REASON, "timeout")
            if pending.event.wait(min(self.poll_interval, remaining)):
                return pending.decision or Decision(False, None, TIMEOUT_REASON, "timeout")
            answer = self._read_answer(pending)
            if answer is not None:
                return answer

    def _read_answer(self, pending: PendingPermission) -> Decision | None:
        try:
            raw = pending.answer_path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        allow = bool(data.get("allow")) and not str(data.get("decision") or "").lower() in ("deny", "reject")
        return Decision(allow, data.get("optionId"), str(data.get("reason") or "answered by user"), "user")

    def answer(self, permission_id: str, decision: Decision | bool, *, reason: str = "") -> bool:
        """Resolve a pending request from any thread (CLI, UI plugin, callback)."""
        if isinstance(decision, bool):
            decision = Decision(decision, None, reason or ("approved" if decision else "denied"), "user")
        with self._lock:
            pending = self._pending.get(permission_id)
        if pending is None:
            return False
        pending.decision = decision
        pending.event.set()
        return True

    def pending_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(p.request) for p in self._pending.values()]

    def _cleanup(self, pending: PendingPermission) -> None:
        with self._lock:
            self._pending.pop(pending.permission_id, None)
        for path in (pending.path, pending.answer_path):
            try:
                path.unlink()
            except OSError:
                pass

    def _pick(self, options: list[dict[str, Any]], wanted: tuple[str, ...]) -> str | None:
        option = _option_by_kind(options, wanted)
        return str(option["optionId"]) if option else None

    def _respond(self, message: dict[str, Any], decision: Decision | None, options: list[dict[str, Any]]) -> dict[str, Any]:
        outcome: dict[str, Any]
        if decision is not None and decision.allow:
            option_id = decision.option_id or self._pick(options, ALLOW_KINDS)
            outcome = ({"outcome": "selected", "optionId": option_id} if option_id
                       else {"outcome": "cancelled"})
        else:
            option_id = (decision.option_id if decision else None) or self._pick(options, REJECT_KINDS)
            outcome = ({"outcome": "selected", "optionId": option_id} if option_id
                       else {"outcome": "cancelled"})
        return {"jsonrpc": "2.0", "id": message.get("id"), "result": {"outcome": outcome}}

    def _note(self, event: str, session_id: str, name: str, kind: str, decision: Decision | None,
              extra: dict[str, Any] | None = None) -> None:
        payload = {"event": event, "session": session_id, "tool": name, "kind": kind,
                   "allow": None if decision is None else decision.allow,
                   "reason": None if decision is None else decision.reason,
                   "source": None if decision is None else decision.source}
        if extra:
            payload.update(extra)
        try:
            self.log(payload)
        except Exception:
            pass


def _pending_files(permission_dir: Path) -> list[Path]:
    """Requests only — never a ``.answer.json`` (a watcher that globs ``perm-*.json`` would otherwise
    answer its own answers and grow ``perm-x.answer.answer.json`` without bound)."""
    return [p for p in Path(permission_dir).glob("perm-*.json") if not p.name.endswith(".answer.json")]


def answer_from_cli(permission_dir: Path, permission_id: str, allow: bool, *, reason: str = "",
                    option_id: str | None = None) -> bool:
    """Write an answer file for ``permission_id`` (used by the bundled CLI)."""
    if permission_id.endswith(".answer") or ".answer." in permission_id:
        return False  # refuse to answer an answer file
    target = Path(permission_dir) / f"{permission_id}.answer.json"
    if not (Path(permission_dir) / f"{permission_id}.json").exists():
        return False
    payload = {"allow": bool(allow), "reason": reason or ("approved" if allow else "denied")}
    if option_id:
        payload["optionId"] = option_id
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
    return True
