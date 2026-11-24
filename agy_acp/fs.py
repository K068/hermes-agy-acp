"""Minimal ``fs/*`` bridge for agents that delegate file access to the client.

The bundled Antigravity agent has its own file tools, so this path is normally unused. It exists
because ACP clients are allowed to advertise the capability, and a release that starts preferring
client-side reads should degrade into working reads instead of a failed turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _within(path_text: str, cwd: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = path.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


def read_text_file(params: dict[str, Any], cwd: str) -> dict[str, str]:
    path = _within(str(params.get("path") or ""), cwd)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        content = ""
    line, limit = params.get("line"), params.get("limit")
    if isinstance(line, int) and line > 1:
        end = line - 1 + limit if isinstance(limit, int) and limit > 0 else None
        content = "".join(content.splitlines(keepends=True)[line - 1:end])
    return {"content": content}


def write_text_file(params: dict[str, Any], cwd: str) -> None:
    path = _within(str(params.get("path") or ""), cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(params.get("content") or ""), encoding="utf-8")


HANDLERS = {"fs/read_text_file": read_text_file, "fs/write_text_file": write_text_file}
