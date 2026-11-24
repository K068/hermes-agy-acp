"""Test harness: a temp Hermes home, the mock agent, and a ready-to-drive client."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PACKAGE_DIR = HERE.parent
def _mock_path() -> Path:
    """Works in both layouts: tests inside the plugin dir, or ``tests/`` beside an ``agy_acp/`` package."""
    here = Path(__file__).resolve().parent.parent
    for candidate in (here / "mock_agy_acp.py", here / "agy_acp" / "mock_agy_acp.py"):
        if candidate.exists():
            return candidate
    return here / "mock_agy_acp.py"


MOCK = _mock_path()
sys.path.insert(0, str(PACKAGE_DIR.parent))  # import the plugin package by name

from agy_acp.client import AgyAcpClient  # noqa: E402
from agy_acp.settings import Settings  # noqa: E402

_ENV_KEYS = (
    "HERMES_SESSION_ID", "HERMES_CRON_SESSION", "HERMES_SESSION_SOURCE", "MOCK_AGY_SCENARIO",
    "AGY_ACP_APPROVAL_BACKEND",
    "MOCK_AGY_RECORD", "MOCK_AGY_SESSIONS", "MOCK_AGY_TURN_SECONDS", "MOCK_AGY_PERMISSION_GRACE",
    "MOCK_AGY_CHUNK_DELAY", "MOCK_AGY_REPLY", "MOCK_AGY_MODELS",
)


class Harness:
    """One isolated provider world: own home, own state db, own mock process(es)."""

    def __init__(self, *, scenario: str = "default", session: str = "test-session", **settings: Any) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="agy-acp-test-"))
        self.work = self.root / "work"
        self.work.mkdir(parents=True)
        self.record = self.root / "records.jsonl"
        self.sessions_file = self.root / "sessions.json"
        self._saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
        os.environ["HERMES_SESSION_ID"] = session
        os.environ.pop("HERMES_CRON_SESSION", None)
        os.environ.pop("HERMES_SESSION_SOURCE", None)
        os.environ["MOCK_AGY_SCENARIO"] = scenario
        os.environ.setdefault("AGY_ACP_APPROVAL_BACKEND", "file")  # deterministic: inbox only
        os.environ["MOCK_AGY_RECORD"] = str(self.record)
        os.environ["MOCK_AGY_SESSIONS"] = str(self.sessions_file)
        self.session = session
        base = dict(
            command=sys.executable,
            args=(str(MOCK),),
            home=self.root,
            state_db=self.root / "state" / "sessions.db",
            log_dir=self.root / "logs",
            connect_timeout=20.0,
            turn_timeout=30.0,
            permission_timeout=3.0,
            retry_429_delay=0.05,
            max_sessions=4,
            idle_seconds=1800.0,
        )
        base.update(settings)
        self.settings = Settings(**base)
        self.settings.ensure_dirs()
        self.client = AgyAcpClient(settings=self.settings, acp_cwd=str(self.work))
        self.manager = self.client._manager

    # ── driving ──────────────────────────────────────────────────────
    def turn(self, text: str, *, model: str = "", stream: bool = False, timeout: float | None = 30.0) -> Any:
        messages = [{"role": "user", "content": text}]
        return self.client.chat.completions.create(model=model, messages=messages, stream=stream, timeout=timeout)

    def turn_messages(self, messages: list[dict[str, Any]], *, model: str = "") -> Any:
        return self.client.chat.completions.create(model=model, messages=messages, timeout=30.0)

    # ── observation ──────────────────────────────────────────────────
    def records(self) -> list[dict[str, Any]]:
        if not self.record.exists():
            return []
        out = []
        for line in self.record.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def prompts(self) -> list[dict[str, Any]]:
        return [entry for entry in self.records() if entry.get("event") == "session/prompt"]

    def events(self) -> list[str]:
        names = sorted(p.name for p in (self.root / "logs").glob("events-*.jsonl"))
        lines: list[str] = []
        for name in names:
            lines.extend((self.root / "logs" / name).read_text(encoding="utf-8").splitlines())
        return lines

    def event_dicts(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.events() if line.strip()]

    def close(self) -> None:
        try:
            self.manager.close_all(reason="test teardown")
        finally:
            for key, value in self._saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            shutil.rmtree(self.root, ignore_errors=True)


class HarnessTestCase(unittest.TestCase):
    scenario = "default"
    settings: dict[str, Any] = {}

    def setUp(self) -> None:
        self.h = Harness(scenario=self.scenario, **self.settings)

    def tearDown(self) -> None:
        self.h.close()
