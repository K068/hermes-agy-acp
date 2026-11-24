"""The Hermes control plane: one short block, injected once, on turn 1 only.

Constraints that shape this module:

* **≤ 500 tokens**, because every byte here is permanent prefix for the whole conversation.
* **No tool instructions.** ``agy`` owns file/terminal/search and documents its own tools; repeating
  Hermes' tool rules here would fight the agent's own prompt and invite duplicate work.
* **Turn 1 only.** The block rides in the first ``session/prompt`` user turn, so it becomes part of
  the cached prefix instead of mutating it on every turn.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .memory import estimate_tokens

OPEN_TAG = "<hermes_control_plane>"
CLOSE_TAG = "</hermes_control_plane>"

#: Lines that carry Hermes' own tool/format rules — the part agy must not be taught twice.
_TOOL_LINE_MARKERS = (
    "<tool_call", "</tool_call", "tool_calls", "function call", "function schema",
    "available tools", "tools:", "read_file", "write_file", "apply_patch", "patch(",
    "terminal(", "bash", "shell", "run_command", "search_files", "grep", "ls ",
    "execute_code", "browser_exec", "delegate_task", "JSON object with", "arguments must be",
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")

BRIDGE_SECTION = """\
Hermes memory and skills are reachable through the `hermes-bridge` MCP server, which this session has:
- `hermes_memory` action=search|show|save|forget — the user's long-term memory from other sessions.
  Call it when a request depends on something you were never told (past decisions, environment
  details, earlier mistakes). Do not search for facts already in this conversation.
- `hermes_skill` action=list|view — Hermes' domain playbooks. Check it before improvising a
  multi-step procedure the user has clearly done before.
Every bridge result is paged: respect `has_more`/`next_page` instead of assuming truncation means absence.
Never route file edits, shell commands or code search through the bridge — use your own tools."""


def _strip_tool_lines(text: str) -> str:
    kept: list[str] = []
    for line in (text or "").splitlines():
        low = line.lower()
        if any(marker in low for marker in _TOOL_LINE_MARKERS):
            continue
        kept.append(line)
    # collapse the holes left behind by stripped blocks
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _persona_source(settings: Any) -> str:
    """SOUL.md is the persona proper; the built system prompt is the fallback."""
    home = getattr(settings, "home", None)
    if home:
        soul = Path(home) / "SOUL.md"
        try:
            if soul.is_file():
                return soul.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return ""


def build_control_plane(
    settings: Any, *, system_prompt: str = "", persona: str = "", budget_tokens: int | None = None,
) -> str:
    """Return the block *without* the wrapper tags (empty string when nothing is worth sending)."""
    budget = int(budget_tokens or getattr(settings, "control_plane_max_tokens", 500) or 500)
    override = (persona or "").strip()
    if not override:
        override = _strip_tool_lines(_persona_source(settings) or system_prompt)
    else:
        override = _strip_tool_lines(override)

    bridge = BRIDGE_SECTION if getattr(settings, "inject_hermes_layer", True) else ""
    # Reserve room for the bridge section first: it is the part with no substitute.
    bridge_tokens = estimate_tokens(bridge)
    persona_budget = max(0, budget - bridge_tokens - 8)
    if persona_budget and override:
        # character budget from the token estimate (CJK ~1 char/token, ASCII ~3 chars/token)
        cut = override
        while cut and estimate_tokens(cut) > persona_budget:
            cut = cut[: max(0, int(len(cut) * 0.8))]
        override = cut.rstrip()
    parts = [part for part in (override, bridge) if part.strip()]
    text = "\n\n".join(parts).strip()
    if estimate_tokens(text) > budget:
        text = text[: max(0, int(len(text) * 0.85))].rstrip()
    return text


def wrap_control_plane(text: str) -> str:
    if not text.strip():
        return ""
    return f"{OPEN_TAG}\n{text.strip()}\n{CLOSE_TAG}"


def control_plane_turn_text(settings: Any, user_text: str, *, turn_number: int,
                            system_prompt: str = "") -> str:
    """The text actually sent for one turn: the block on turn 1 only, in front of the user's words."""
    if turn_number != 1 or not getattr(settings, "inject_hermes_layer", True):
        return user_text
    block = wrap_control_plane(build_control_plane(settings, system_prompt=system_prompt))
    return f"{block}\n\n{user_text}".strip() if block else user_text
