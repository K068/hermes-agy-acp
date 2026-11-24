"""Tiny MCP stdio client used by the tests to drive ``hermes-bridge`` the way agy would."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

def _bridge_path() -> Path:
    """Both layouts: tests inside the plugin dir, or ``tests/`` beside an ``agy_acp/`` package."""
    here = Path(__file__).resolve().parent.parent
    for candidate in (here / "bridge_mcp.py", here / "agy_acp" / "bridge_mcp.py"):
        if candidate.exists():
            return candidate
    return here / "bridge_mcp.py"


BRIDGE = _bridge_path()


class McpClient:
    """Spawn the bridge, do the handshake, call tools, read results."""

    def __init__(self, *, env: dict[str, str] | None = None, timeout: float = 20.0,
                 handshake: str = "plain") -> None:
        self.timeout = timeout
        self.handshake = handshake  # "strict" mimics agy: server/discover first, newest version
        self.last_protocol_version = ""
        child_env = dict(os.environ)
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env.pop("AGY_ACP_SESSION_ID", None)
        if env:
            child_env.update(env)
        self.proc = subprocess.Popen(
            [sys.executable, str(BRIDGE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=child_env, cwd=str(BRIDGE.parent),
        )
        self._inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr: list[str] = []
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()
        self._next_id = 0

    def _pump_out(self) -> None:
        for line in self.proc.stdout or ():
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except ValueError:
                continue

    def _pump_err(self) -> None:
        for line in self.proc.stderr or ():
            self.stderr.append(line.rstrip())

    def send(self, payload: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                message = self._inbox.get(timeout=0.2)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"bridge exited: {self.stderr[-3:]}")
                continue
            if message.get("id") == request_id:
                return message
        raise TimeoutError(f"no answer to {method}")

    def initialize(self) -> dict[str, Any]:
        if self.handshake == "strict":
            probe = self.request("server/discover", {
                "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                          "io.modelcontextprotocol/clientInfo": {"name": "antigravity-client", "version": "v1.0.0"}}})
            if "error" in probe:
                raise RuntimeError(f"server/discover refused: {probe['error']}")
        result = self.request("initialize", {
            "protocolVersion": "2025-11-25" if self.handshake == "strict" else "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agy-mock", "version": "0"},
        })["result"]
        self.last_protocol_version = str(result.get("protocolVersion") or "")
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list")["result"]["tools"]

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        payload = self.request("tools/call", {"name": name, "arguments": arguments})["result"]
        text = "".join(part.get("text", "") for part in payload.get("content") or []
                       if isinstance(part, dict))
        return text, bool(payload.get("isError"))

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
