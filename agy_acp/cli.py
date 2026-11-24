"""Small operator CLI for the provider: status, cancel, and the permission inbox.

    python -m agy_acp.cli status
    python -m agy_acp.cli cancel <hermes-session-id>
    python -m agy_acp.cli permissions
    python -m agy_acp.cli answer <permission-id> allow|deny
    python -m agy_acp.cli sessions

``cancel`` writes the sentinel the running turn's tick watches (milliseconds, no process teardown);
``answer`` writes the response file the permission broker polls. Both work from any shell while the
gateway keeps running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .permissions import answer_from_cli
from .settings import Settings
from .store import SessionStore


def _store(settings: Settings) -> SessionStore:
    return SessionStore(settings.state_db)  # type: ignore[arg-type]


def cmd_status(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    from .permissions import _pending_files

    store = _store(settings)
    try:
        rows = store.all_sessions()
    finally:
        store.close()
    payload = {
        "state_db": str(settings.state_db),
        "log_dir": str(settings.log_dir),
        "max_concurrent_turns": settings.max_concurrent_turns,
        "capability_mode": settings.capability_mode,
        "bridge_approval": settings.bridge_approval,
        "inflight": store.inflight_sessions() if hasattr(store, "inflight_sessions") else [],
        "sessions": [
            {"hermes_session": row["hermes_session_id"], "acp_session": row["acp_session_id"],
             "model": row["model"], "turns": row["turn_count"],
             "idle_seconds": int(time.time()) - int(row["last_active_at"]),
             "injected_memory": len(json.loads(row["injected_memory_ids"] or "[]"))}
            for row in rows
        ],
        "pending_permissions": [p.name for p in _pending_files(Path(settings.permission_dir))],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    path = Path(settings.cancel_dir) / f"{args.session}.cancel"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time()), encoding="utf-8")
    print(f"cancel requested for {args.session} ({path})")
    return 0


def cmd_permissions(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    from .permissions import _pending_files

    for path in sorted(_pending_files(Path(settings.permission_dir))):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tool = payload.get("tool") or {}
        print(f"{payload.get('id')}  {tool.get('name')}  {tool.get('title')}  session={payload.get('session_id')}")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    ok = answer_from_cli(Path(settings.permission_dir), args.permission_id, args.verdict == "allow",
                         reason=f"answered from CLI ({args.verdict})")
    print("answered" if ok else "no such pending permission")
    return 0 if ok else 1


def cmd_sessions(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = _store(settings)
    try:
        for row in store.all_sessions():
            print(json.dumps({k: row[k] for k in ("hermes_session_id", "acp_session_id", "model", "turn_count",
                                                  "last_active_at")}, ensure_ascii=False))
    finally:
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agy_acp.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("sessions").set_defaults(func=cmd_sessions)
    sub.add_parser("permissions").set_defaults(func=cmd_permissions)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("session")
    cancel.set_defaults(func=cmd_cancel)
    answer = sub.add_parser("answer")
    answer.add_argument("permission_id")
    answer.add_argument("verdict", choices=("allow", "deny"))
    answer.set_defaults(func=cmd_answer)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
