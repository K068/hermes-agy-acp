"""Display-layer handling for agent-side tool output.

The agent's own context is never truncated here — nothing in this module feeds tokens back into
``agy``. It only shapes what Hermes shows and stores:

* tool titles/outputs are folded to ``head + tail`` with an explicit "omitted" marker, so a 4 MB
  ``npm test`` dump cannot freeze a UI or bloat the chat database;
* every event is appended as one NDJSON line to a rotated per-session log, which is where the full
  (already truncated for display) text lives for debugging;
* the per-turn tool counter is the circuit breaker for runaway tool loops.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OMITTED = "\n...[已省略 {n} 字节，完整输出保留在 agy 上下文中]...\n"


def truncate(text: str, *, head: int = 2048, tail: int = 2048) -> str:
    """Head + tail folding with an explicit marker. Short text is returned untouched."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if head < 0:
        head = 0
    if tail < 0:
        tail = 0
    if len(text) <= head + tail + 32:
        return text
    omitted = len(text) - head - tail
    return text[:head] + OMITTED.format(n=omitted) + (text[-tail:] if tail else "")


@dataclass
class ToolCounter:
    """Circuit breaker state for one turn."""

    limit: int = 15
    seconds: float = 180.0
    started_at: float = field(default_factory=time.monotonic)
    count: int = 0
    tripped_reason: str = ""

    def observe(self, *, is_start: bool) -> None:
        if is_start:
            self.count += 1
        if self.tripped_reason:
            return
        if self.count > self.limit:
            self.tripped_reason = f"tool loop: {self.count} tool calls in one turn (limit {self.limit})"
            return
        elapsed = time.monotonic() - self.started_at
        if elapsed > self.seconds:
            self.tripped_reason = f"tool loop: turn has run {elapsed:.0f}s (limit {self.seconds:.0f}s)"

    @property
    def tripped(self) -> bool:
        return bool(self.tripped_reason)


class EventLog:
    """Append-only NDJSON log of agent activity, rotated by size. Never raises."""

    def __init__(self, path: Path, *, max_bytes: int = 5 * 1024 * 1024) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def write(self, event: dict[str, Any]) -> None:
        payload = {"ts": round(time.time(), 3), **event}
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            try:
                if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                    rotated = self.path.with_suffix(self.path.suffix + ".1")
                    try:
                        rotated.unlink()
                    except OSError:
                        pass
                    self.path.rename(rotated)
            except OSError:
                pass
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                pass


def summarize_update(update: dict[str, Any], *, head: int, tail: int) -> dict[str, Any]:
    """Compact, display-safe view of one ``session/update`` payload."""
    kind = str(update.get("sessionUpdate") or "").replace("_update", "") or "unknown"
    summary: dict[str, Any] = {"update": kind}
    content = update.get("content")
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        summary["chars"] = len(content["text"])
        summary["preview"] = truncate(content["text"], head=head, tail=tail)
    for key in ("title", "status", "toolCallId", "kind", "name", "toolName"):
        value = update.get(key)
        if isinstance(value, str) and value:
            summary[key] = truncate(value, head=head, tail=tail)
    raw = update.get("rawInput")
    if raw is not None:
        try:
            summary["input"] = truncate(json.dumps(raw, ensure_ascii=False)[:4000], head=head, tail=tail)
        except (TypeError, ValueError):
            summary["input"] = truncate(str(raw)[:4000], head=head, tail=tail)
    for key, label in (("rawOutput", "output"), ("content", "output")):
        value = update.get(key) if key == "rawOutput" else None
        if isinstance(value, dict):
            value = value.get("text")
        if isinstance(value, str) and value:
            summary[label] = truncate(value, head=head, tail=tail)
            break
    if isinstance(update.get("locations"), list):
        summary["locations"] = update["locations"][:4]
    return summary
