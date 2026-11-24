"""Mounting ``hermes-bridge`` into the ACP session.

ACP carries MCP servers as ``{name, command, args, env:[{name,value}]}`` and the *agent* spawns them,
so the bridge inherits the agent's cwd and process lifetime. The command is this interpreter plus the
bridge script (absolute paths only): a pickup from ``PATH`` would resolve differently inside the
gateway's sanitised child environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
BRIDGE_SCRIPT = PACKAGE_DIR / "bridge_mcp.py"
BRIDGE_NAME = "hermes-bridge"  # default; overridable with settings.bridge_name


def _sr_cli() -> str:
    candidate = os.environ.get("AGY_ACP_SR_CLI", "").strip()
    if candidate and Path(candidate).exists():
        return candidate
    venv = os.environ.get("VIRTUAL_ENV", "")
    if venv:
        path = Path(venv) / "bin" / "scope-recall"
        if path.exists():
            return str(path)
    sibling = Path(sys.executable).with_name("scope-recall")  # the interpreter Hermes runs on
    if sibling.exists():
        return str(sibling)
    import shutil

    return shutil.which("scope-recall") or ""


def _store_roots(home: Path | str) -> list[Path]:
    """Where this user's scope-recall store may live, most specific first."""
    roots: list[Path] = []
    explicit = os.environ.get("AGY_ACP_SR_ROOT", "").strip()
    if explicit:
        roots.append(Path(explicit))
    base = Path(home) if str(home) else Path.home()
    pointer = base / "scope-recall" / "shared-root"
    try:
        if pointer.is_file():
            value = pointer.read_text(encoding="utf-8").strip()
            if value:
                roots.append(Path(value))
    except OSError:
        pass
    roots.append(base / ".scope-recall" / "shared")
    roots.append(Path.home() / ".scope-recall" / "shared")
    return roots


def bridge_env(settings: Any, *, session_id: str = "") -> dict[str, str]:
    name = str(getattr(settings, "bridge_name", "") or BRIDGE_NAME)
    env = {
        "HERMES_HOME": str(getattr(settings, "home", "")),
        "AGY_ACP_BRIDGE_NAME": name,
        "AGY_ACP_BRIDGE_PAGE_BYTES": str(int(getattr(settings, "bridge_page_bytes", 4096) or 4096)),
    }
    skills = ",".join(getattr(settings, "bridge_tools", ()) or ())
    if skills:
        env["AGY_ACP_BRIDGE_SKILLS"] = skills
    if session_id:
        env["AGY_ACP_SESSION_ID"] = session_id
    sr = _sr_cli()
    if sr:
        env["AGY_ACP_SR_CLI"] = sr
    # agy runs the bridge with HOME=its own account home, so a store root must be resolved here,
    # where HOME is still the Hermes user's, and handed over explicitly.
    shared_root = os.environ.get("AGY_ACP_SR_ROOT", "").strip()
    if not shared_root:
        for candidate in _store_roots(getattr(settings, "home", "") or ""):
            if candidate.is_dir():
                shared_root = str(candidate)
                break
    if shared_root:
        env["AGY_ACP_SR_ROOT"] = shared_root
    trace_path = os.environ.get("AGY_ACP_BRIDGE_TRACE", "").strip()
    if trace_path:
        env["AGY_ACP_BRIDGE_TRACE"] = trace_path
    return env


def server_entry(settings: Any, *, session_id: str = "") -> dict[str, Any]:
    command = str(getattr(settings, "bridge_command", "") or "").strip() or sys.executable
    args = list(getattr(settings, "bridge_args", ()) or ()) or [str(BRIDGE_SCRIPT)]
    env = bridge_env(settings, session_id=session_id)
    return {
        "name": str(getattr(settings, "bridge_name", "") or BRIDGE_NAME),
        "command": command,
        "args": args,
        "env": [{"name": key, "value": value} for key, value in sorted(env.items())],
    }


def mount(settings: Any, *, session_id: str = "") -> list[dict[str, Any]]:
    """The ``mcpServers`` value for ``session/new``. Empty when the layer is switched off."""
    if not getattr(settings, "inject_hermes_layer", True):
        return []
    return [server_entry(settings, session_id=session_id)]
