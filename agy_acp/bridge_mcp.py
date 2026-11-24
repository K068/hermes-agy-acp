#!/usr/bin/env python3
"""``hermes-bridge`` — the single, local, stdio MCP server the Antigravity agent may call.

Why a *bridge* and not more tools: everything local (read/write files, search code, run commands)
already exists inside ``agy``. Re-declaring those in Hermes would double the tool schema, put two
owners on every action, and push the fixed per-turn cost up — the exact waste this provider exists to
avoid. So the bridge exposes only what Hermes has and agy does not:

* ``hermes_memory`` — one polymorphic tool for the memory plane: offline substring ``search`` over
  scope-recall's store, ``show`` one ref, ``revise``/``forget`` a memory. Fewer tools, one
  schema, no per-action boilerplate.
* ``hermes_skill`` — one polymorphic tool to list/inspect Hermes skills (the persona/domain
  knowledge that lives in ``$HERMES_HOME/skills``), because agy knows nothing about them.

Hard rule: **every** result is capped at ``AGY_ACP_BRIDGE_PAGE_BYTES`` (4 KB by default) with
``has_more``/``next_page``, so one call can never flood the agent's context.

Transport: MCP stdio = newline-delimited JSON-RPC (no Content-Length framing). No third-party
dependency, nothing listens on a port.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SERVER_NAME = os.environ.get("AGY_ACP_BRIDGE_NAME", "").strip() or "hermes-bridge"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-06-18"
#: Newer MCP clients (Antigravity ACK 2026-07-28) probe these before ``initialize``.
SUPPORTED_VERSIONS = ("2026-07-28", "2025-11-25", "2025-06-18", "2024-11-05")
MAX_TOOLS = 5
DEFAULT_PAGE_BYTES = int(os.environ.get("AGY_ACP_BRIDGE_PAGE_BYTES", "4096") or 4096)

#: Inline cap for one character column before paging kicks in (keeps a single hit readable).
INLINE_CHARS = 600


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


# ── tool schemas ────────────────────────────────────────────────────────
def _memory_tool() -> dict[str, Any]:
    return {
        "name": "hermes_memory",
        "description": (
            "Hermes long-term memory. action=search finds past facts by literal substring (offline, "
            "no model call); action=show reads one ref; action=revise corrects an existing memory "
            "(ref + revision + value, or retract=true); action=forget retracts whole refs. Nothing "
            "creates new memory here: new memories come from the turn capture pipeline, so say the "
            "fact in your answer and it is recorded. Results are paged: read has_more/next_page."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "show", "revise", "forget"]},
                "query": {"type": "string", "description": "search: literal substring"},
                "ref": {"type": "string", "description": "show/revise/forget: memory ref (e.g. event-…)"},
                "revision": {"type": "integer", "minimum": 1, "description": "show/revise: revision (default 1)"},
                "value": {"type": "string", "description": "revise: the corrected statement"},
                "refs": {"type": "array", "items": {"type": "string"}, "description": "forget: refs"},
                "retract": {"type": "boolean", "description": "revise: withdraw the memory instead of setting a value"},
                "page": {"type": "integer", "minimum": 1, "description": "1-based page (default 1)"},
                "page_size": {"type": "integer", "minimum": 1, "description": "items per page (default 8)"},
            },
            "required": ["action"],
        },
    }


def _skill_tool() -> dict[str, Any]:
    return {
        "name": "hermes_skill",
        "description": (
            "Hermes skills (persona/domain instructions the Antigravity agent cannot see). "
            "action=list shows names+summaries; action=view reads one skill body. Paged."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "view"]},
                "name": {"type": "string", "description": "view: skill name or directory name"},
                "query": {"type": "string", "description": "list: optional filter substring"},
                "page": {"type": "integer", "minimum": 1},
                "page_size": {"type": "integer", "minimum": 1},
            },
            "required": ["action"],
        },
    }


def tool_schemas(extra_skills: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """At most ``MAX_TOOLS`` tools: the two polymorphic ones, plus one per configured skill."""
    tools = [_memory_tool(), _skill_tool()]
    seen = {"hermes_memory", "hermes_skill"}
    for name in extra_skills:
        if len(tools) >= MAX_TOOLS:
            break
        slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")[:24] or "skill"
        tool_name = f"hermes_skill_note__{slug}"
        if tool_name in seen:  # two skills that slug identically must not collide on the wire
            continue
        seen.add(tool_name)
        tools.append({
            "name": tool_name,
            "description": f"Read the configured Hermes skill '{name}' (paged).",
            "inputSchema": {"type": "object", "properties": {
                "name": {"type": "string", "const": name},
                "page": {"type": "integer", "minimum": 1},
                "page_size": {"type": "integer", "minimum": 1},
            }, "required": []},
        })
    return tools[:MAX_TOOLS]


# ── paging ──────────────────────────────────────────────────────────────
def page_items(items: list[str], *, page: int, page_size: int, budget: int) -> dict[str, Any]:
    """Pack rendered items into one page under ``budget`` bytes; report where to continue."""
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 8), 50))
    start = (page - 1) * page_size
    chunk = items[start:start + page_size]
    kept: list[str] = []
    used = 0
    truncated_item = False
    for item in chunk:
        size = len(item.encode("utf-8")) + 1
        if used + size > budget:
            if not kept:  # a single item larger than the budget: hard-slice it
                raw = item.encode("utf-8")[: max(0, budget - 60)].decode("utf-8", "ignore")
                kept.append(raw + "\n…[item truncated by bridge page limit]")
                truncated_item = True
            break
        kept.append(item)
        used += size
    consumed = len(kept) if not truncated_item else len(kept)
    has_more = start + consumed < len(items)
    out = {
        "page": page,
        "page_size": page_size,
        "items": kept,
        "has_more": has_more,
        "next_page": page + 1 if has_more else None,
        "total_items": len(items),
        "bytes": used,
    }
    return out


def render(payload: dict[str, Any], *, budget: int) -> str:
    """Serialise a page and keep the whole MCP result under ``budget`` bytes."""
    text = json.dumps(payload, ensure_ascii=False)
    if len(text.encode("utf-8")) <= budget:
        return text
    # Drop the tail of the item list until it fits; never emit an unbounded string.
    items = list(payload.get("items") or [])
    while items and len(json.dumps({**payload, "items": items + ["…"], "truncated": True},
                                   ensure_ascii=False).encode("utf-8")) > budget:
        items.pop()
    payload = {**payload, "items": items + (["…[more items on the next page]"] if payload.get("has_more") else []),
               "truncated": True}
    text = json.dumps(payload, ensure_ascii=False)
    if len(text.encode("utf-8")) > budget:  # last resort: hard slice on a character boundary
        text = text.encode("utf-8")[:budget].decode("utf-8", "ignore") + "…[truncated]"
    return text


# ── scope-recall (CLI, offline) ─────────────────────────────────────────
def _sr_cli() -> list[str] | None:
    override = os.environ.get("AGY_ACP_SR_CLI", "").strip()
    if override and Path(override).exists():
        return [override]
    # the interpreter running this bridge is the gateway's python; its bin/ usually holds the CLI
    sibling = Path(sys.executable).with_name("scope-recall")
    if sibling.exists():
        return [str(sibling)]
    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        path = Path(venv) / "bin" / "scope-recall"
        if path.exists():
            return [str(path)]
    import shutil

    found = shutil.which("scope-recall")
    return [found] if found else None


_LOCATOR: list[str] | None = None
_LOCATOR_RESOLVED = False


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    explicit = os.environ.get("AGY_ACP_SR_ROOT", "").strip()
    if explicit:
        roots.append(Path(explicit))
    pointer = hermes_home() / "scope-recall" / "shared-root"
    try:
        if pointer.is_file():
            value = pointer.read_text(encoding="utf-8").strip()
            if value:
                roots.append(Path(value))
    except OSError:
        pass
    roots.append(hermes_home() / ".scope-recall" / "shared")
    root = Path.home() / ".scope-recall" / "shared"
    roots.append(root)
    return roots


def _sr_locator(*, refresh: bool = False) -> list[str]:
    """How to name the store: an explicit root for a shared store, else this home as an entry."""
    global _LOCATOR, _LOCATOR_RESOLVED
    if _LOCATOR_RESOLVED and not refresh:
        return list(_LOCATOR or [])
    explicit = os.environ.get("AGY_ACP_SR_ROOT", "").strip()
    if explicit:
        _LOCATOR, _LOCATOR_RESOLVED = ["--root", explicit], True
        return list(_LOCATOR)
    for candidate in _candidate_roots():
        if candidate.is_dir():
            _LOCATOR, _LOCATOR_RESOLVED = ["--root", str(candidate)], True
            return list(_LOCATOR)
    _LOCATOR, _LOCATOR_RESOLVED = ["--host", "hermes", "--instance-root", str(hermes_home())], True
    return list(_LOCATOR)


def _run_sr(args: list[str], *, timeout: float, retry_locator: bool = True) -> tuple[bool, Any]:
    cli = _sr_cli()
    if cli is None:
        return False, "scope-recall CLI not found (set AGY_ACP_SR_CLI); bridge notes still work"
    cmd = [*cli, "memory", *args, "--json", *_sr_locator()]
    env = dict(os.environ)
    env.setdefault("HERMES_HOME", str(hermes_home()))
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"scope-recall failed to run: {exc}"
    payload: Any
    try:
        payload = json.loads((done.stdout or "").strip() or "{}")
    except ValueError:
        payload = (done.stderr or done.stdout or "").strip()
    # A shared store answers "name the store: --root" when asked as an entry: switch the locator once.
    if (isinstance(payload, dict) and payload.get("code") in ("store_unnamed", "shared_store_entry")
            and retry_locator):
        for candidate in _candidate_roots():
            if not candidate.is_dir():
                continue
            _LOCATOR = ["--root", str(candidate)]  # type: ignore[assignment]
            globals()["_LOCATOR_RESOLVED"] = True
            return _run_sr(args, timeout=timeout, retry_locator=False)
    if done.returncode != 0 or (isinstance(payload, dict) and payload.get("status") == "error"):
        if isinstance(payload, dict):
            detail = payload.get("field") or payload.get("code") or "error"
            message = payload.get("message") or payload.get("detail") or ""
        else:
            lines = str(payload).strip().splitlines()
            detail, message = (lines[-1][:200] if lines else "error"), ""
        return False, f"scope-recall: {detail} {message}".strip()
    return True, payload


def _items_from_sr(payload: Any) -> list[dict[str, Any]]:
    """The management contract nests results under ``data.items``; tolerate the older shapes too."""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("items", "results", "entries", "records"):
                if isinstance(data.get(key), list):
                    return [v for v in data[key] if isinstance(v, dict)]
        for key in ("items", "results", "hits", "entries", "records"):
            if isinstance(payload.get(key), list):
                return [v for v in payload[key] if isinstance(v, dict)]
    if isinstance(payload, list):
        return [v for v in payload if isinstance(v, dict)]
    return []


def _short(value: Any, limit: int = INLINE_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def _legacy_notes() -> Path:
    """``bridge_notes.jsonl`` from versions that had ``action=save``. Read-only now: the file was a
    dead end (not a scope-recall claim), and new facts come from the capture pipeline instead."""
    return hermes_home() / "agy_acp" / "bridge_notes.jsonl"


def memory_search(query: str, *, page: int, page_size: int, budget: int) -> str:
    if not query.strip():
        return render({"ok": False, "error": "query is required for action=search"}, budget=budget)
    ok, payload = _run_sr(["search", query, "--limit", str(max(20, page_size * 3))], timeout=15.0)
    lines: list[str] = []
    source = "scope-recall"
    detail: dict[str, Any] = {}
    if ok:
        for item in _items_from_sr(payload):
            ref = item.get("ref") or item.get("id") or item.get("entry_ref") or ""
            content = item.get("content") or item.get("text") or item.get("claim") or item
            stamp = item.get("occurred_at") or item.get("created_at") or ""
            lines.append(f"[{ref}] {stamp} {_short(content)}".strip())
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            for key in ("examined", "gaps", "answerability", "coverage"):
                if data.get(key) is not None:
                    detail[key] = data[key]
    else:
        lines.append(f"scope-recall unavailable: {_short(payload, 200)}")
        source = "unavailable"
    # legacy bridge notes from versions that had action=save: off unless explicitly enabled
    notes = _legacy_notes()
    if _env_flag("AGY_ACP_LEGACY_NOTES") and notes.exists():
        for line in notes.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                note = json.loads(line)
            except ValueError:
                continue
            if query.lower() in str(note.get("text", "")).lower():
                lines.append(f"[bridge-note:{note.get('id', '')}] {_short(note.get('text'))}")
    rendered = page_items(lines, page=page, page_size=page_size, budget=max(512, budget - 300))
    return render({"ok": ok, "source": source, "query": query, **detail, **rendered}, budget=budget)


def memory_show(ref: str, *, revision: int, budget: int) -> str:
    if not ref.strip():
        return render({"ok": False, "error": "ref is required for action=show"}, budget=budget)
    ok, payload = _run_sr(["show", ref, "--revision", str(max(1, int(revision or 1)))], timeout=15.0)
    if not ok:
        return render({"ok": False, "ref": ref, "error": _short(payload, 400)}, budget=budget)
    data = payload.get("data") if isinstance(payload, dict) else None
    value = data.get("value") if isinstance(data, dict) else None
    event = value.get("event") if isinstance(value, dict) else None
    content = event.get("content") if isinstance(event, dict) else None
    text = _short(content if content is not None else (value if value is not None else data), INLINE_CHARS)
    return render({"ok": True, "ref": ref, "revision": revision, "content": text,
                   "has_more": len(text) >= INLINE_CHARS, "next_page": None,
                   "note": "show returns one memory; use action=search to find others"},
                  budget=budget)


def memory_revise(ref: str, *, revision: int, value: str, retract: bool, budget: int) -> str:
    """Change an existing memory's value, or withdraw it. The only write scope-recall exposes."""
    if not ref.strip():
        return render({"ok": False, "error": "ref is required for action=revise"}, budget=budget)
    if not value.strip() and not retract:
        return render({"ok": False, "error": "action=revise needs value, or retract=true"},
                      budget=budget)
    args = ["revise", ref.strip(), "--expected-revision", str(max(1, int(revision or 1)))]
    args += ["--retract"] if retract else ["--value", value]
    ok, payload = _run_sr(args, timeout=20.0)
    if not ok:
        return render({"ok": False, "ref": ref, "error": _short(payload, 400),
                       "hint": "re-read with action=show; --expected-revision must match"},
                      budget=budget)
    data = payload.get("data") if isinstance(payload, dict) else None
    new_revision = None
    if isinstance(data, dict):
        new_revision = data.get("revision")
        value_obj = data.get("value")
        if new_revision is None and isinstance(value_obj, dict):
            new_revision = value_obj.get("revision")
    return render({"ok": True, "ref": ref, "action": "retract" if retract else "revise",
                   "revision": new_revision, "detail": _short(data if data is not None else payload, 300)},
                  budget=budget)


def memory_forget(refs: list[str], *, budget: int) -> str:
    refs = [r for r in refs if r.strip()]
    if not refs:
        return render({"ok": False, "error": "refs is required for action=forget"}, budget=budget)
    ok, payload = _run_sr(["forget", *refs, "--yes"], timeout=20.0)
    return render({"ok": ok, "forgot": refs if ok else [], "detail": _short(payload, 300)},
                  budget=budget)


# ── skills ──────────────────────────────────────────────────────────────
def _skill_dirs() -> list[Path]:
    root = hermes_home() / "skills"
    if not root.is_dir():
        return []
    return sorted({p.parent for p in root.rglob("SKILL.md")})


def _frontmatter(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.parent.name, ""
    name, desc = path.parent.name, ""
    if text.startswith("---"):
        head = text.split("---", 2)[1] if text.count("---") >= 2 else ""
        for line in head.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "name":
                name = value.strip().strip("\"'") or name
            elif key.strip() == "description":
                desc = value.strip().strip("\"'")
    return name, desc


def skill_list(query: str, *, page: int, page_size: int, budget: int) -> str:
    lines: list[str] = []
    for folder in _skill_dirs():
        name, desc = _frontmatter(folder / "SKILL.md")
        if query and query.lower() not in f"{name} {desc} {folder}".lower():
            continue
        lines.append(f"{name} ({folder.name}): {_short(desc, 160) or '(no description)'}")
    rendered = page_items(lines, page=page, page_size=page_size, budget=max(512, budget - 300))
    return render({"ok": True, **rendered}, budget=budget)


def skill_view(name: str, *, page: int, page_size: int, budget: int) -> str:
    wanted = name.strip().lower()
    for folder in _skill_dirs():
        skill_name, desc = _frontmatter(folder / "SKILL.md")
        if wanted in (skill_name.lower(), folder.name.lower()):
            body = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")
            return render({"ok": True, "name": skill_name, "path": str(folder / "SKILL.md"),
                           "page": max(1, page), "page_size": max(1, page_size),
                           "content": _short(body, max(200, budget - 400)),
                           "has_more": len(body) > budget - 400,
                           "next_page": max(1, page) + 1 if len(body) > budget - 400 else None},
                          budget=budget)
    return render({"ok": False, "error": f"no skill named {name!r}", "hint": "action=list"}, budget=budget)


# ── dispatch ────────────────────────────────────────────────────────────
def call_tool(name: str, arguments: dict[str, Any], *, budget: int) -> tuple[str, bool]:
    action = str(arguments.get("action") or "").strip().lower()
    page = int(arguments.get("page") or 1)
    page_size = int(arguments.get("page_size") or 8)
    if name == "hermes_memory":
        if action == "search":
            return memory_search(str(arguments.get("query") or ""), page=page, page_size=page_size, budget=budget), False
        if action == "show":
            return memory_show(str(arguments.get("ref") or ""),
                               revision=int(arguments.get("revision") or 1), budget=budget), False
        if action == "forget":
            refs = arguments.get("refs") or ([arguments["ref"]] if arguments.get("ref") else [])
            return memory_forget([str(r) for r in refs], budget=budget), False
        if action == "revise":
            return memory_revise(str(arguments.get("ref") or ""),
                                 revision=int(arguments.get("revision") or 1),
                                 value=str(arguments.get("value") or ""),
                                 retract=bool(arguments.get("retract")), budget=budget), False
        return render({"ok": False, "error": f"unknown action {action!r}",
                       "allowed": ["search", "show", "revise", "forget"]}, budget=budget), True
    if name in ("hermes_skill", "hermes_skill_note") or name.startswith(("hermes_skill_note__",
                                                                             "hermes_skill__")):
        if action == "view" or name.startswith(("hermes_skill_note", "hermes_skill__")):
            target = str(arguments.get("name") or "")
            return skill_view(target, page=page, page_size=page_size, budget=budget), False
        return skill_list(str(arguments.get("query") or ""), page=page, page_size=page_size, budget=budget), False
    return render({"ok": False, "error": f"unknown tool {name!r}"}, budget=budget), True


# ── JSON-RPC loop ───────────────────────────────────────────────────────
def _trace_line(direction: str, payload: Any) -> None:
    """Raw stdio trace for diagnosing an MCP client that refuses the handshake (opt-in)."""
    path = os.environ.get("AGY_ACP_BRIDGE_TRACE", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 3), "dir": direction, "payload": payload},
                                ensure_ascii=False) + "\n")
    except OSError:
        pass


def send(message: dict[str, Any]) -> None:
    _trace_line("out", message)
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def result(msg_id: Any, payload: dict[str, Any]) -> None:
    send({"jsonrpc": "2.0", "id": msg_id, "result": payload})


def error(msg_id: Any, code: int, message: str) -> None:
    send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})


def handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    msg_id = message.get("id")
    if not isinstance(method, str):
        return  # a response to something we never sent: ignore
    if method == "server/discover":
        # A strict client drops the whole server when this errors, so answer the probe.
        result(msg_id, {
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {"tools": {"listChanged": False}},
            "protocolVersion": PROTOCOL_VERSION,
            "protocolVersions": list(SUPPORTED_VERSIONS),
        })
        return
    if method == "initialize":
        wanted = str((message.get("params") or {}).get("protocolVersion") or "").strip()
        result(msg_id, {
            "protocolVersion": wanted if wanted in SUPPORTED_VERSIONS else PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
        return
    if method in ("notifications/initialized", "initialized"):
        return
    if method == "ping":
        result(msg_id, {})
        return
    if method == "tools/list":
        extra = tuple(t for t in (os.environ.get("AGY_ACP_BRIDGE_SKILLS") or "").split(",") if t.strip())
        result(msg_id, {"tools": tool_schemas(extra)})
        return
    # Optional MCP surfaces: an error here makes a strict client drop the whole server, so answer empty.
    if method == "resources/list":
        result(msg_id, {"resources": []})
        return
    if method == "resources/templates/list":
        result(msg_id, {"resourceTemplates": []})
        return
    if method == "prompts/list":
        result(msg_id, {"prompts": []})
        return
    if method == "logging/setLevel":
        result(msg_id, {})
        return
    if method.startswith("notifications/"):
        return  # notifications never get a reply
    if method == "tools/call":
        params = message.get("params") or {}
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        budget = int(os.environ.get("AGY_ACP_BRIDGE_PAGE_BYTES", "") or DEFAULT_PAGE_BYTES)
        try:
            text, is_error = call_tool(name, arguments, budget=budget)
        except Exception as exc:  # noqa: BLE001 - the bridge must answer, never crash the agent's turn
            text, is_error = render({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, budget=budget), True
        result(msg_id, {"content": [{"type": "text", "text": text}], "isError": is_error})
        return
    error(msg_id, -32601, f"{SERVER_NAME}: unsupported method {method}")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        _trace_line("in", line[:2000])
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if isinstance(message, dict):
            try:
                handle(message)
            except Exception as exc:  # noqa: BLE001
                error(message.get("id"), -32603, f"{SERVER_NAME}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
