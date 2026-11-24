#!/usr/bin/env python3
"""Free probe: what this account's `agy acp` really answers, without spending a single prompt.

    python3 -m agy_acp.probe            # inside ~/.hermes/plugins
    AGY_ACP_BIN=/opt/acp-ant/bin/agy AGY_ACP_HOME=/opt/acp-ant/accounts/g1 \
      AGY_ACP_PROXY=http://127.0.0.1:7890 python3 -m agy_acp.probe

Sends ``initialize``, ``session/new``, ``session/load`` (with a bogus id, to prove the method is
there) and ``session/close``. No ``session/prompt``, so no model is called and no quota is used.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from .protocol import AcpError, AgyConnection
from .settings import Settings


def main() -> int:
    settings = Settings.from_env()
    settings.ensure_dirs()
    workdir = os.environ.get("AGY_ACP_PROBE_CWD") or tempfile.mkdtemp(prefix="agy-probe-")
    out: dict[str, object] = {"command": settings.command, "args": list(settings.args),
                              "home": settings.child_home or os.environ.get("HOME", ""),
                              "proxy": settings.proxy, "cwd": workdir}
    conn = AgyConnection(settings, cwd=workdir, label="probe")
    try:
        init = conn.start()
        out["agentInfo"] = conn.agent_info
        caps = conn.capabilities
        out["capabilities"] = {
            "loadSession": bool(caps.get("loadSession") or caps.get("load_session")),
            "sessionCapabilities": sorted((caps.get("sessionCapabilities") or {})),
        }
        session = conn.request("session/new", {"cwd": workdir, "mcpServers": []}, timeout=settings.connect_timeout)
        session_id = str(session.get("sessionId") or "")
        out["sessionId"] = session_id
        out["models"] = [m.get("modelId") for m in ((session.get("models") or {}).get("availableModels") or [])]
        out["configOptions"] = [
            {"id": o.get("id"), "category": o.get("category"), "currentValue": o.get("currentValue"),
             "values": [v.get("value") for v in (o.get("options") or [])][:12]}
            for o in (session.get("configOptions") or []) if isinstance(o, dict)
        ]
        out["tier"] = ((session.get("_meta") or {}).get("antigravity") or {}).get("tier")
        try:
            conn.request("session/load", {"sessionId": "probe-does-not-exist", "cwd": workdir,
                                          "mcpServers": []}, timeout=30)
            out["session/load"] = "accepted a bogus id (unexpected)"
        except AcpError as exc:
            out["session/load"] = f"{exc.code}: {str(exc)[:120]}"
        if session_id:
            try:
                conn.request("session/close", {"sessionId": session_id}, timeout=15)
                out["session/close"] = "ok"
            except AcpError as exc:
                out["session/close"] = str(exc)[:120]
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the probe's job is to report, not to raise
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["stderr_tail"] = conn.stderr_tail[-8:]
    finally:
        conn.close("probe done")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
