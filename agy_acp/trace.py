"""Frame-level NDJSON tracing for one child process.

Enabled with ``AGY_ACP_NDJSON_TRACE=1``; writes one line per frame to
``<HERMES_HOME>/.hermes_traces/<session>.ndjson``:

    {"ts":1795485198.881,"dir":"out","bytes":214,"frame":{...}}
    {"ts":...,"dir":"in","method":"session/update","bytes":91234,"frame":"...(truncated)", "truncated":true}

Writes go through a small queue and a single writer thread: a trace must never add latency to the
turn (or to the reader that keeps the agent's stdout drained). If the queue backs up the frame is
dropped with a counter, not blocked on.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

#: One line of a tool result can be megabytes; keep the trace readable and bounded.
MAX_FRAME_CHARS = 4000
MAX_FIELD_CHARS = 600


def _shrink(value: Any, depth: int = 0) -> Any:
    """Copy of ``value`` with long strings cut and depth bounded."""
    if depth > 6:
        return "…"
    if isinstance(value, str):
        return value if len(value) <= MAX_FIELD_CHARS else value[:MAX_FIELD_CHARS] + f"…[{len(value)} chars]"
    if isinstance(value, list):
        items = [_shrink(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            items.append(f"…[{len(value)} items]")
        return items
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 20:
                out["…"] = f"[{len(value)} keys]"
                break
            out[str(key)] = _shrink(item, depth + 1)
        return out
    return value


class TraceWriter:
    """Append-only writer with a bounded queue and a daemon thread."""

    def __init__(self, path: Path, *, max_queue: int = 2000) -> None:
        self.path = Path(path)
        self.dropped = 0
        self.written = 0
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._started = False
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _start(self) -> None:
        if self._started:
            return
        self._started = True

        def pump() -> None:
            with self.path.open("a", encoding="utf-8") as handle:
                while True:
                    line = self._queue.get()
                    if line is None:
                        handle.flush()
                        return
                    try:
                        handle.write(line + "\n")
                        handle.flush()  # a trace is read while the turn is still running
                        self.written += 1
                    except OSError:
                        return

        self._thread = threading.Thread(target=pump, name=f"agy-acp-trace-{self.path.stem}", daemon=True)
        self._thread.start()

    def frame(self, direction: str, frame: Any, *, truncated_hint: int | None = None) -> None:
        """Record one frame. ``direction`` is ``"in"`` (agent→client) or ``"out"`` (client→agent)."""
        payload: Any = frame
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        cut = len(encoded) > MAX_FRAME_CHARS
        if cut:
            payload = _shrink(payload)
            encoded = json.dumps(payload, ensure_ascii=False, default=str)
            if len(encoded) > MAX_FRAME_CHARS:
                payload = encoded[:MAX_FRAME_CHARS] + "…[frame truncated]"
                encoded = json.dumps(payload, ensure_ascii=False)
        entry = {
            "ts": round(time.time(), 3),
            "dir": direction,
            "bytes": truncated_hint if truncated_hint is not None else len(encoded),
            "method": (frame or {}).get("method") if isinstance(frame, dict) else None,
            "id": (frame or {}).get("id") if isinstance(frame, dict) else None,
            "frame": payload,
        }
        if cut:
            entry["truncated"] = True
        self._emit(entry)

    def _emit(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._start()
        line = json.dumps(entry, ensure_ascii=False, default=str)
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        if not self._started:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
