"""``scope-recall`` adaptation for a stateful agent: tail injection with per-session dedupe.

Hermes already puts recalled memory where a prefix cache can survive it: it appends the provider's
``prefetch()`` text to the **current user message** (``agent.turn_context.compose_user_api_content``
→ ``build_memory_context_block``), never into the system prompt. That is exactly the "turn-level
suffix" this provider needs, so nothing has to be intercepted or re-implemented.

What is left is the second half of the problem: a stateful agent keeps everything it was sent, so a
memory that was injected on turn 3 is already in the native context on turn 4. Sending it again
spends tokens and pushes the agent toward reciting its own memory. This module therefore:

* splits the injected ``<memory-context>`` block off the user text;
* parses the canonical JSON items inside it;
* drops items whose identity was already sent in this ACP session (tracked in SQLite);
* re-renders the survivors as a compact ``<recalled_memory>`` block; and
* enforces the recall budget (``top_k``-sized item list, ``max_injected_tokens``).

It is deliberately tolerant: anything it cannot parse is passed through untouched, because losing
a user's own words is far worse than injecting a memory twice.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

MEMORY_OPEN = "<memory-context>"
MEMORY_CLOSE = "</memory-context>"
SYSTEM_NOTE_PREFIX = "[System note:"
ID_FIELDS = ("id", "ref", "memory_id", "memory_ref", "source_ref", "entry_ref", "entry_id")

#: Budget from the spec: at most this many items and this many tokens of recalled context.
MAX_ITEMS = 5
MAX_INJECTED_TOKENS = 1500
#: Keys kept from the recall packet's envelope; they tell the model how complete the recall was
#: (``gaps``/``unmet_needs`` describe retrieval limits, not user facts) and cost almost nothing.
META_KEYS = ("answerability", "coverage", "gaps", "unmet_needs", "status")
#: Per-item projection: what a model needs to judge an item, not the whole evidence frame.
ITEM_KEYS = ("ref", "revision", "occurred_at", "kind")
#: Start here and shrink until the block fits; a first-item-too-big case must never drop everything.
CONTENT_START_CHARS = 1200
CONTENT_MIN_CHARS = 120

_CJK_RE = re.compile(r"[\u2e80-\ua4cf\uac00-\ud7af\uf900-\ufaff\U00020000-\U000323af]")


def estimate_tokens(text: str) -> int:
    """Conservative estimate: CJK one token/char, ASCII four chars/token, symbols by bytes."""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return cjk + (rest // 3) + 1


@dataclass
class ParsedMemory:
    guidance: str = ""
    items: list[Any] = field(default_factory=list)
    kind: str = "list"  # "list" | "object"
    meta: dict[str, Any] = field(default_factory=dict)

    def keys(self) -> list[str]:
        return [item_key(item) for item in self.items]


def split_memory_block(content: str) -> tuple[str, str | None]:
    """Return ``(user_text, memory_block)``; ``memory_block`` keeps its ``<memory-context>`` tags."""
    if not isinstance(content, str) or MEMORY_OPEN not in content or MEMORY_CLOSE not in content:
        return content, None
    start = content.find(MEMORY_OPEN)
    end = content.find(MEMORY_CLOSE, start)
    if end == -1:
        return content, None
    end += len(MEMORY_CLOSE)
    block = content[start:end]
    user_text = (content[:start] + content[end:]).strip()
    return user_text, block


def parse_memory_block(block: str) -> ParsedMemory | None:
    """Parse the block Hermes builds; ``None`` when the payload is not the canonical JSON shape."""
    if not block:
        return None
    body = block.strip()
    if body.startswith(MEMORY_OPEN):
        body = body[len(MEMORY_OPEN):]
    if body.endswith(MEMORY_CLOSE):
        body = body[: -len(MEMORY_CLOSE)]
    # The payload is the trailing JSON value. The probe note itself starts with "[System note: …",
    # so "first brace" is wrong — take the first position whose remainder parses as JSON.
    start = -1
    payload: Any = None
    for index, char in enumerate(body):
        if char not in "{[":
            continue
        try:
            payload = json.loads(body[index:].strip())
        except ValueError:
            continue
        start = index
        break
    if start == -1:
        return None
    prefix = body[:start]
    guidance_lines = [
        line.strip() for line in prefix.splitlines()
        if line.strip() and not line.strip().startswith(SYSTEM_NOTE_PREFIX)
    ]
    guidance = " ".join(guidance_lines).strip()
    if isinstance(payload, list):
        return ParsedMemory(guidance=guidance, items=list(payload), kind="list")
    if isinstance(payload, dict):
        meta = {key: payload[key] for key in META_KEYS if key in payload}
        items = payload.get("items")
        if isinstance(items, list):
            return ParsedMemory(guidance=guidance, items=list(items), kind="object", meta=meta)
        return ParsedMemory(guidance=guidance, items=[payload], kind="object", meta=meta)
    return None


def item_key(item: Any) -> str:
    """Stable identity for one memory item: a declared id when present, else a content hash."""
    if isinstance(item, dict):
        for name in ID_FIELDS:
            value = item.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
    blob = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "h:" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _compact(item: Any, *, chars: int) -> Any:
    """One item as a model-facing projection: identity + time + a bounded content excerpt."""
    if not isinstance(item, dict):
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return text if len(text) <= chars else text[:chars] + "…"
    out: dict[str, Any] = {key: item[key] for key in ITEM_KEYS if key in item and item[key] is not None}
    content = item.get("content")
    if isinstance(content, str):
        out["content"] = content if len(content) <= chars else content[:chars] + "…"
        if len(content) > chars:
            out["truncated"] = True
        return out
    # no plain content: keep a bounded view of whatever carries the meaning
    blob = json.dumps(item, ensure_ascii=False, sort_keys=True)
    out["content"] = blob if len(blob) <= chars else blob[:chars] + "…"
    if len(blob) > chars:
        out["truncated"] = True
    return out


def _render(payload: Any, guidance: str, kind: str) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    parts = ['<recalled_memory scope="agy-acp">',
             "[System note: recalled memory context from earlier sessions; not new user input.]"]
    if guidance:
        parts.append(guidance)
    parts.append(text)
    parts.append("</recalled_memory>")
    return "\n".join(parts)


def dedupe_and_render(
    parsed: ParsedMemory, already_injected: Iterable[str], *,
    max_items: int = MAX_ITEMS, max_tokens: int = MAX_INJECTED_TOKENS,
    item_chars: int = CONTENT_START_CHARS,
) -> tuple[str | None, list[str]]:
    """Drop already-sent items, fit the rest into the budget, render. Returns ``(block|None, new_keys)``.

    Real packets carry 2–20 KB per item, so "measure the whole item, break if it does not fit" drops
    *everything* on the first item and silently removes recall from the turn. Instead: project each
    item to identity + a bounded excerpt, start generous and shrink until the block fits, and only
    fall back to dropping trailing items once the excerpt floor is reached.
    """
    seen = {str(k) for k in already_injected}
    fresh: list[tuple[str, Any]] = []
    for item in parsed.items:
        key = item_key(item)
        if key in seen:
            continue
        fresh.append((key, item))
        if len(fresh) >= max_items:
            break
    if not fresh:
        return None, []

    guidance = parsed.guidance
    while guidance and estimate_tokens(guidance) > max(60, max_tokens // 3):
        guidance = guidance[: int(len(guidance) * 0.7)].rstrip()

    payload_meta: dict[str, Any] = dict(parsed.meta)
    chars = max(CONTENT_MIN_CHARS, int(item_chars))
    while True:
        items = [_compact(item, chars=chars) for _key, item in fresh]
        payload: Any = {"items": items} if parsed.kind == "object" else items
        if parsed.kind == "object" and payload_meta:
            payload = {**payload_meta, "items": items}
        rendered = _render(payload, guidance, parsed.kind)
        if estimate_tokens(rendered) <= max_tokens:
            return rendered, [key for key, _item in fresh]
        if chars > CONTENT_MIN_CHARS:
            chars = max(CONTENT_MIN_CHARS, int(chars * 0.7))
            continue
        if len(fresh) > 1:  # the floor is reached: give up the least relevant (last) item
            fresh.pop()
            continue
        # one item, floor reached: hard-trim the rendered text rather than lose the recall entirely
        cut = int(len(rendered) * 0.85)
        trimmed = _render({"items": [_compact(fresh[0][1], chars=max(40, cut // 2))],
                           **({"…": "trimmed to fit the recall budget"} if parsed.kind == "object" else {})},
                          guidance[: max(0, len(guidance) // 2)], parsed.kind)
        return trimmed, [fresh[0][0]]


def adapt_user_text(content: str, already_injected: Iterable[str], *,
                    max_tokens: int = MAX_INJECTED_TOKENS,
                    item_chars: int = CONTENT_START_CHARS) -> tuple[str, list[str], str | None]:
    """Full turn-level adaptation.

    Returns ``(text_to_send, new_memory_keys, dropped_note)``. ``dropped_note`` is a one-line
    diagnostic (``None`` when nothing was removed) — it is written to the event log, never into the
    prompt, because mentioning memory bookkeeping to the model wastes tokens and invites it to
    re-derive what it already knows.
    """
    user_text, block = split_memory_block(content)
    if block is None:
        return content, [], None
    parsed = parse_memory_block(block)
    if parsed is None:
        return content, [], "memory block present but unparsable; left untouched"
    rendered, new_keys = dedupe_and_render(parsed, already_injected)
    if rendered is None:
        return user_text, [], f"all {len(parsed.items)} recalled item(s) already in context; block dropped"
    dropped = len(parsed.items) - len(new_keys)
    note = f"{dropped} recalled item(s) dropped as already-sent" if dropped else None
    return f"{rendered}\n{user_text}".strip(), new_keys, note
