"""Turn slicing: the only thing Hermes sends a stateful agent is *this* turn.

A stateful session owns its history. So instead of ``_format_messages_as_prompt`` (the whole
transcript, replayed every request), this module extracts exactly:

* the newest user message, minus the ``<memory-context>`` block Hermes appended to it (re-added as a
  deduped, budgeted ``<recalled_memory>`` suffix);
* its ``data:`` image parts, as ACP image blocks;
* a static system prompt on the session's **first** turn only (frozen prefix — never re-sent);
* and, when the text is enormous, a spilled file plus a path reference.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .memory import CONTENT_START_CHARS, MAX_INJECTED_TOKENS, adapt_user_text

_DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
#: Kept inline when a prompt is spilled, so a model whose file tool is unavailable is not blind.
SPILL_HEAD_CHARS = 4000
SPILL_NOTE = (
    "[agy_acp] This turn's input was too large to send through the ACP pipe ({chars} characters). "
    "The full text is at {path}. Read that file before answering; the beginning is repeated below.\n"
)


@dataclass
class TurnPrompt:
    text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)
    new_memory_keys: list[str] = field(default_factory=list)
    user_hash: str = ""
    notes: list[str] = field(default_factory=list)
    spilled_path: str = ""
    images_dropped: int = 0
    layered: bool = False


def _content_parts(content: Any) -> tuple[list[str], list[dict[str, Any]], int]:
    """Split OpenAI-style message content into (text parts, ACP image blocks, dropped images)."""
    if isinstance(content, str):
        return [content], [], 0
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    dropped = 0
    for part in content or []:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type") or "")
        if kind in ("text", "input_text") and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind in ("image_url", "input_image"):
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str):
                dropped += 1
                continue
            match = _DATA_URL_RE.match(url.strip())
            if not match:
                dropped += 1  # remote URL: fetching it here would be an unrequested network cost
                continue
            mime, payload = match.group(1), match.group(2).strip()
            try:
                base64.b64decode(payload[:64] + "===", validate=False)
            except (binascii.Error, ValueError):
                dropped += 1
                continue
            images.append({"type": "image", "mimeType": mime, "data": payload})
        elif isinstance(part.get("text"), str):
            texts.append(part["text"])
    return texts, images, dropped


def last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages or []):
        if isinstance(message, dict) and str(message.get("role") or "").lower() == "user":
            return message
    return None


def message_text(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    texts, _images, _dropped = _content_parts(message.get("content"))
    return "\n".join(part for part in texts if part).strip()


def user_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def spill_text(text: str, *, cwd: str, session_id: str, turn_number: int, fallback_dir: Path,
               limit: int, head_bytes: int = 0) -> tuple[str, str]:
    """Write an oversized turn next to the workspace (so the agent's own read_file can open it).

    Returns ``(replacement_text, path)``. On any IO error the text is returned unchanged — a
    blocked turn is worse than a long one.
    """
    safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "session")[:64]
    name = f"turn-{turn_number}.txt"
    candidates = [Path(cwd) / ".hermes_tmp" / "agy_acp" / safe_session, Path(fallback_dir) / safe_session]
    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / name
            path.write_text(text, encoding="utf-8")
        except OSError:
            continue
        note = SPILL_NOTE.format(chars=len(text), path=str(path))
        head = text[:SPILL_HEAD_CHARS]
        if head_bytes:  # the head is what stays inline, and agy counts bytes, not characters
            while head and len(head.encode("utf-8")) > head_bytes:
                head = head[: int(len(head) * 0.8)]
        return f"{note}\n--- begin ---\n{head}\n--- end ---", str(path)
    return text, ""


def build_turn(
    messages: list[dict[str, Any]], *,
    already_injected: set[str] | list[str],
    cwd: str,
    session_id: str,
    turn_number: int,
    max_chars: int,
    work_dir: Path,
    settings: Any = None,
    system_prompt: str = "",
) -> TurnPrompt:
    message = last_user_message(messages)
    texts, images, dropped = _content_parts(message.get("content") if message else "")
    raw = "\n".join(part for part in texts if part).strip()
    adapted, new_keys, note = adapt_user_text(
        raw, already_injected,
        max_tokens=int(getattr(settings, "memory_tokens", MAX_INJECTED_TOKENS) or MAX_INJECTED_TOKENS),
        item_chars=int(getattr(settings, "memory_item_chars", CONTENT_START_CHARS) or CONTENT_START_CHARS),
    ) if settings is not None else adapt_user_text(raw, already_injected)
    notes: list[str] = []
    if note:
        notes.append(note)
    if dropped:
        notes.append(f"{dropped} image part(s) skipped (only inline data: images are forwarded)")
    spilled = ""
    max_bytes = int(getattr(settings, "max_prompt_bytes", 0) or 0) if settings is not None else 0
    over_bytes = bool(max_bytes) and len(adapted.encode("utf-8")) > max_bytes
    if len(adapted) > max_chars or over_bytes:
        adapted, spilled = spill_text(
            adapted, cwd=cwd, session_id=session_id, turn_number=turn_number,
            fallback_dir=work_dir, limit=max_chars,
            head_bytes=max(512, int(max_bytes * 0.4)) if max_bytes else 0)
        if spilled:
            notes.append(f"turn spilled to {spilled} ({len(raw)} chars / "
                         f"{len(raw.encode('utf-8'))} bytes, budget {max_chars} chars"
                         + (f" / {max_bytes} bytes" if max_bytes else "") + ")")
    layered = False
    if settings is not None and getattr(settings, "inject_hermes_layer", True):
        # Turn 1 only, and *after* the spill: the inline head must stay the user's own words, so the
        # control plane is prepended to whatever text finally goes out.
        from .control_plane import control_plane_turn_text

        with_layer = control_plane_turn_text(settings, adapted, turn_number=turn_number,
                                             system_prompt=system_prompt)
        if with_layer != adapted:
            layered = True
            notes.append("hermes control plane injected (turn 1)")
        adapted = with_layer
    blocks: list[dict[str, Any]] = [{"type": "text", "text": adapted}] if adapted else []
    blocks.extend(images)
    return TurnPrompt(
        text=adapted, blocks=blocks, new_memory_keys=new_keys, user_hash=user_hash(raw),
        notes=notes, spilled_path=spilled, images_dropped=dropped, layered=layered,
    )
