#!/usr/bin/env python3
"""A stand-in ``agy acp`` server for the provider's tests.

Speaks the same JSON-RPC/NDJSON dialect the real CLI does (verified against agy 1.5.15 on
2026-11-23): ``initialize`` → ``session/new|load`` → ``session/set_config_option`` →
``session/prompt`` with ``session/update`` notifications, ``session/request_permission`` as a
server→client request, ``session/cancel`` as a notification, ``session/close``.

Running the acceptance tests against this costs no quota and cannot 429. Scenarios are selected
with ``MOCK_AGY_SCENARIO`` and every received prompt is appended to ``MOCK_AGY_RECORD`` so a test
can prove what actually crossed the pipe.

Run directly to smoke-test:  MOCK_AGY_SCENARIO=default python mock_agy_acp.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid

SCENARIO = os.environ.get("MOCK_AGY_SCENARIO", "default")
RECORD = os.environ.get("MOCK_AGY_RECORD", "")
SESSION_FILE = os.environ.get("MOCK_AGY_SESSIONS", "")
CHUNK_DELAY = float(os.environ.get("MOCK_AGY_CHUNK_DELAY", "0.01"))
TURN_SECONDS = float(os.environ.get("MOCK_AGY_TURN_SECONDS", "0"))
PERMISSION_GRACE = float(os.environ.get("MOCK_AGY_PERMISSION_GRACE", "30"))
REPLY = os.environ.get("MOCK_AGY_REPLY", "mock reply: {n}")
MODELS = [m for m in os.environ.get(
    "MOCK_AGY_MODELS",
    "gemini-4-neon,gemini-3.8-flash,claude-sonnet-5-5").split(",") if m]
LEVELS = ["low", "medium", "high", "xhigh"]

_out_lock = threading.Lock()
_state_lock = threading.Lock()
_sessions: dict[str, dict] = {}
_cancel: dict[str, threading.Event] = {}
_permission_waiters: dict[int, dict] = {}
_prompt_counts: dict[str, int] = {}
_rate_limited_once = False
_next_server_id = 1000


def log(message: str) -> None:
    sys.stderr.write(f"[mock-agy] {message}\n")
    sys.stderr.flush()


def send(payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def notify(method: str, params: dict) -> None:
    send({"jsonrpc": "2.0", "method": method, "params": params})


def record(entry: dict) -> None:
    if not RECORD:
        return
    with _out_lock:
        try:
            with open(RECORD, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass


# ── session persistence (so session/load can cross process restarts) ─────────
def load_sessions() -> None:
    global _sessions
    if not SESSION_FILE or not os.path.exists(SESSION_FILE):
        return
    try:
        with open(SESSION_FILE, encoding="utf-8") as handle:
            _sessions = json.load(handle).get("sessions") or {}
    except (OSError, ValueError):
        _sessions = {}


def save_sessions() -> None:
    if not SESSION_FILE:
        return
    tmp = SESSION_FILE + ".tmp"
    with _state_lock:
        payload = {"sessions": _sessions}
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp, SESSION_FILE)


# ── protocol bits ───────────────────────────────────────────────────────────
def config_options(model: str = "gemini-4-neon", level: str = "high") -> list[dict]:
    return [
        {"id": "model", "category": "model", "type": "select", "currentValue": model,
         "options": [{"value": m, "name": m} for m in MODELS]},
        {"id": "thought_level", "category": "thought_level", "type": "select", "currentValue": level,
         "options": [{"value": lv, "name": lv} for lv in LEVELS]},
    ]


def session_payload(session_id: str, model: str = "gemini-4-neon", level: str = "high") -> dict:
    return {
        "sessionId": session_id,
        "models": {"availableModels": [{"modelId": m, "name": m} for m in MODELS]},
        "configOptions": config_options(model, level),
        "_meta": {"antigravity": {"tier": {"name": "Mock Tier", "id": "mock"}}},
    }


def handle_initialize(msg: dict) -> None:
    record({"event": "initialize", "params": msg.get("params")})
    send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {
        "protocolVersion": 1,
        "agentCapabilities": {
            "loadSession": True,
            "promptCapabilities": {"image": True, "audio": False, "embeddedContext": True},
            "sessionCapabilities": {"load": {}, "close": {}, "resume": {}, "list": {}},
            "_meta": {"antigravity": {"systemPrompt": {"append": True}}},
        },
        "agentInfo": {"name": "Mock Antigravity ACP", "version": "0.1.0"},
        "authMethods": [{"id": "oauth-personal", "name": "Log in with Google"}],
    }})


def handle_new(msg: dict) -> None:
    params = msg.get("params") or {}
    if SCENARIO == "silent_new":
        return  # deliberately never answered: exercises request timeouts
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"cwd": params.get("cwd"), "model": "gemini-4-neon", "level": "high",
                             "system_prompt": ((params.get("_meta") or {}).get("antigravity") or {})
                             .get("systemPrompt", {}).get("append", ""),
                             "turns": []}
    save_sessions()
    record({"event": "session/new", "session": session_id, "params": {k: v for k, v in params.items() if k != "_meta"},
            "system_prompt": _sessions[session_id]["system_prompt"]})
    send({"jsonrpc": "2.0", "id": msg.get("id"), "result": session_payload(session_id)})


def handle_load(msg: dict) -> None:
    params = msg.get("params") or {}
    session_id = str(params.get("sessionId") or "")
    if SCENARIO == "load_fail" or session_id not in _sessions:
        send({"jsonrpc": "2.0", "id": msg.get("id"),
              "error": {"code": -32002, "message": f"Session not found: {session_id}"}})
        return
    record({"event": "session/load", "session": session_id,
            "mcpServers": (msg.get("params") or {}).get("mcpServers")})
    session = _sessions[session_id]
    # A real agy answers session/load only after broadcasting session state; those notifications land
    # on the client *while* it is still waiting for the load result, which is exactly where a naive
    # "hold one lock across the handshake" implementation deadlocks.
    notify("session/update", {"sessionId": session_id, "update": {
        "sessionUpdate": "session_info_update", "title": session.get("title") or "Mock session"}})
    notify("session/update", {"sessionId": session_id, "update": {
        "sessionUpdate": "available_commands_update", "availableCommands": [{"name": "init"}]}})
    send({"jsonrpc": "2.0", "id": msg.get("id"), "result": session_payload(
        session_id, session.get("model", "gemini-4-neon"), session.get("level", "high"))})


def handle_set_config(msg: dict) -> None:
    params = msg.get("params") or {}
    session_id = str(params.get("sessionId") or "")
    session = _sessions.get(session_id) or {}
    if not session_id or not session:
        send({"jsonrpc": "2.0", "id": msg.get("id"),
              "error": {"code": -32602, "message": "unknown session"}})
        return
    if params.get("configId") == "model":
        session["model"] = str(params.get("value"))
    elif params.get("configId") == "thought_level":
        session["level"] = str(params.get("value"))
    save_sessions()
    record({"event": "session/set_config_option", "session": session_id, "configId": params.get("configId"),
            "value": params.get("value")})
    send({"jsonrpc": "2.0", "id": msg.get("id"),
          "result": {"configOptions": config_options(session.get("model", "gemini-4-neon"),
                                                     session.get("level", "high"))}})


def permission_request(session_id: str, tool: dict) -> dict:
    """Ask the client for permission; answer with the returned outcome (or timeout)."""
    global _next_server_id
    _next_server_id += 1
    request_id = _next_server_id
    waiter = {"event": threading.Event(), "response": None}
    _permission_waiters[request_id] = waiter
    send({"jsonrpc": "2.0", "id": request_id, "method": "session/request_permission", "params": {
        "sessionId": session_id,
        "toolCall": tool,
        "options": [
            {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
            {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
        ],
    }})
    waiter["event"].wait(PERMISSION_GRACE)
    _permission_waiters.pop(request_id, None)
    return waiter["response"] or {"timeout": True}


def emit_prompt(session_id: str, prompt: list, request_id: int) -> None:
    """Run one turn and record its end timestamp (used to prove turns never overlap)."""
    started = time.time()
    try:
        _emit_prompt(session_id, prompt, request_id)
    finally:
        record({"event": "session/prompt.end", "session": session_id, "ended": time.time(),
                "started": started, "elapsed": round(time.time() - started, 3)})


def _emit_prompt(session_id: str, prompt: list, request_id: int) -> None:
    """Run one turn: stream updates, honour cancel, then answer the request."""
    global _rate_limited_once
    session = _sessions.get(session_id) or {}
    with _state_lock:
        _prompt_counts[session_id] = _prompt_counts.get(session_id, 0) + 1
        # Numbered from the persisted turn list, so a session/load in a new process keeps counting —
        # exactly what the provider's restore path is supposed to buy.
        count = len(session.get("turns") or []) + 1
    text = "".join(str(block.get("text") or "") for block in prompt if isinstance(block, dict))
    record({"event": "session/prompt", "session": session_id, "prompt": prompt, "text": text,
            "chars": len(text), "n": count, "started": time.time()})
    cancel = _cancel.setdefault(session_id, threading.Event())
    cancel.clear()

    if SCENARIO == "ratelimit" and not _rate_limited_once:
        _rate_limited_once = True
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32603, "message": "429 RESOURCE_EXHAUSTED: quota exceeded"}})
        return
    if SCENARIO == "crash":
        log("crashing as requested")
        os._exit(7)

    def thought():
        notify("session/update", {"sessionId": session_id, "update": {
            "sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "thinking..."}}})

    def chunk(text_piece: str) -> bool:
        notify("session/update", {"sessionId": session_id, "update": {
            "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text_piece}}})
        if cancel.wait(CHUNK_DELAY):
            return False
        return True

    thought()
    if TURN_SECONDS:
        deadline = time.monotonic() + TURN_SECONDS
        while time.monotonic() < deadline:
            if cancel.is_set():
                break
            time.sleep(min(CHUNK_DELAY, 0.05))

    if SCENARIO.startswith("permission") and (SCENARIO != "permission" or "permission" in text.lower()):
        readonly = SCENARIO == "permission_read"
        command = text if SCENARIO == "permission_native" else "rm -rf build"
        tool = {"toolCallId": "call-1", "name": "read_file" if readonly else "run_command",
                "kind": "read" if readonly else "execute",
                "title": "read config.yaml" if readonly else "run command",
                "rawInput": {"path": "config.yaml"} if readonly else {"CommandLine": command}}
        notify("session/update", {"sessionId": session_id, "update": {
            "sessionUpdate": "tool_call", "toolCallId": "call-1", "kind": "execute",
            "title": tool["title"], "rawInput": tool["rawInput"], "status": "pending"}})
        outcome = permission_request(session_id, tool)
        record({"event": "permission.outcome", "session": session_id, "outcome": outcome})
        if isinstance(outcome, dict) and outcome.get("timeout"):
            notify("session/update", {"sessionId": session_id, "update": {
                "sessionUpdate": "tool_call_update", "toolCallId": "call-1", "status": "failed",
                "rawOutput": {"text": "permission timed out"}}})
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"stopReason": "refusal", "usage": {"inputTokens": len(text), "outputTokens": 1}}})
            return
        selected = ((outcome.get("result") or {}).get("outcome") or {})
        if selected.get("optionId") == "reject-once" or selected.get("outcome") == "cancelled":
            notify("session/update", {"sessionId": session_id, "update": {
                "sessionUpdate": "tool_call_update", "toolCallId": "call-1", "status": "failed",
                "rawOutput": {"text": "user rejected"}}})
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"stopReason": "refusal", "usage": {"inputTokens": len(text), "outputTokens": 1}}})
            return

    if SCENARIO == "big_output":
        payload = "log line %d\n" % 0 * 1
        big = "".join(f"log line {i:06d} {'x' * 100}\n" for i in range(9000))
        notify("session/update", {"sessionId": session_id, "update": {
            "sessionUpdate": "tool_call", "toolCallId": "call-big", "kind": "execute",
            "title": "npm test", "rawInput": {"command": "npm test"}}})
        notify("session/update", {"sessionId": session_id, "update": {
            "sessionUpdate": "tool_call_update", "toolCallId": "call-big", "status": "completed",
            "rawOutput": {"text": big}}})
        del payload

    if SCENARIO == "loop":
        for index in range(40):
            notify("session/update", {"sessionId": session_id, "update": {
                "sessionUpdate": "tool_call", "toolCallId": f"call-{index}", "kind": "execute",
                "title": f"step {index}"}})
            if cancel.wait(0.02):
                break

    reply = REPLY.format(n=count)
    pieces = [reply[i:i + 7] for i in range(0, len(reply), 7)] or [""]
    for piece in pieces:
        if not chunk(piece):
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"stopReason": "cancelled", "usage": {"inputTokens": len(text), "outputTokens": 0}}})
            return
    session = _sessions.get(session_id)
    if session is not None:
        session.setdefault("turns", []).append({"user": text, "reply": reply})
        save_sessions()
    send({"jsonrpc": "2.0", "id": request_id,
          "result": {"stopReason": "end_turn",
                     "usage": {"inputTokens": len(text), "outputTokens": len(reply)}}})


def handle_prompt(msg: dict) -> None:
    params = msg.get("params") or {}
    session_id = str(params.get("sessionId") or "")
    prompt = params.get("prompt") or []
    if session_id not in _sessions:
        send({"jsonrpc": "2.0", "id": msg.get("id"),
              "error": {"code": -32002, "message": f"Session not found: {session_id}"}})
        return
    threading.Thread(target=emit_prompt, args=(session_id, prompt, msg.get("id")), daemon=True).start()


def handle_cancel(msg: dict) -> None:
    params = msg.get("params") or {}
    session_id = str(params.get("sessionId") or "")
    record({"event": "session/cancel", "session": session_id, "at": time.time()})
    _cancel.setdefault(session_id, threading.Event()).set()


def dispatch(msg: dict) -> None:
    if "method" not in msg:
        waiter = _permission_waiters.get(msg.get("id"))
        if waiter is not None:
            waiter["response"] = msg
            waiter["event"].set()
        return
    method = msg.get("method")
    if method == "session/cancel":
        handle_cancel(msg)
        return
    workers = {
        "initialize": handle_initialize,
        "session/new": handle_new,
        "session/load": handle_load,
        "session/set_config_option": handle_set_config,
        "session/prompt": handle_prompt,
    }
    if method == "session/close":
        record({"event": "session/close", "session": (msg.get("params") or {}).get("sessionId")})
        send({"jsonrpc": "2.0", "id": msg.get("id"), "result": {}})
        return
    handler = workers.get(str(method))
    if handler is None:
        send({"jsonrpc": "2.0", "id": msg.get("id"),
              "error": {"code": -32601, "message": f"mock: unknown method {method}"}})
        return
    if method == "session/prompt":
        handler(msg)
    else:
        handler(msg)


def main() -> None:
    load_sessions()
    if SCENARIO == "stderr_flood":
        for _ in range(20_000):
            sys.stderr.write("stderr flood line " + "y" * 80 + "\n")
        sys.stderr.flush()
    log(f"up scenario={SCENARIO} pid={os.getpid()} sessions={len(_sessions)}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            log(f"bad json: {line[:120]}")
            continue
        try:
            dispatch(msg)
        except Exception as exc:  # noqa: BLE001 - a mock must not die on one bad message
            log(f"dispatch error: {exc!r}")


if __name__ == "__main__":
    main()
