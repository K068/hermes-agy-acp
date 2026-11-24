"""SQLite map ``hermes_session_id -> agy acp session`` plus the state the tail-injection needs.

The table is the only durable state the provider owns. It exists so that a restarted process can
``session/load`` the agent's own native context instead of silently replaying history (which would
both waste quota and defeat the prefix cache).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agy_acp_sessions (
    hermes_session_id TEXT PRIMARY KEY,
    acp_session_id    TEXT NOT NULL,
    cwd               TEXT NOT NULL,
    model             TEXT NOT NULL DEFAULT '',
    injected_memory_ids TEXT NOT NULL DEFAULT '[]',
    last_user_hash    TEXT NOT NULL DEFAULT '',
    prev_user_hash    TEXT NOT NULL DEFAULT '',
    turn_count        INTEGER NOT NULL DEFAULT 0,
    created_at        INTEGER NOT NULL,
    last_active_at    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS agy_acp_inflight (
    hermes_session_id TEXT PRIMARY KEY,
    started_at        INTEGER NOT NULL,
    user_hash         TEXT NOT NULL DEFAULT '',
    turn_count        INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agy_acp_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SessionStore:
    """Thread-safe wrapper around the mapping database (WAL, short transactions)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA busy_timeout=10000")
            self._db.executescript(_SCHEMA)
            self._db.execute("INSERT OR REPLACE INTO agy_acp_meta(key, value) VALUES('schema_version', ?)",
                             (str(SCHEMA_VERSION),))
            self._db.commit()

    # ── rows ──────────────────────────────────────────────────────────
    def get(self, hermes_session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM agy_acp_sessions WHERE hermes_session_id = ?", (hermes_session_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def upsert(self, hermes_session_id: str, *, acp_session_id: str, cwd: str, model: str) -> None:
        now = int(time.time())
        with self._lock:
            self._db.execute(
                """
                INSERT INTO agy_acp_sessions
                    (hermes_session_id, acp_session_id, cwd, model, injected_memory_ids,
                     last_user_hash, prev_user_hash, turn_count, created_at, last_active_at)
                VALUES (?, ?, ?, ?, '[]', '', '', 0, ?, ?)
                ON CONFLICT(hermes_session_id) DO UPDATE SET
                    acp_session_id = excluded.acp_session_id,
                    cwd            = excluded.cwd,
                    model          = excluded.model,
                    last_active_at = excluded.last_active_at
                """,
                (hermes_session_id, acp_session_id, cwd, model, now, now),
            )
            self._db.commit()

    def rebind(self, hermes_session_id: str, acp_session_id: str) -> None:
        """Point the Hermes session at a new native session (resync / branch). Clears per-session memory state."""
        with self._lock:
            self._db.execute(
                "UPDATE agy_acp_sessions SET acp_session_id = ?, injected_memory_ids = '[]', "
                "last_user_hash = '', prev_user_hash = '', turn_count = 0, last_active_at = ? "
                "WHERE hermes_session_id = ?",
                (acp_session_id, int(time.time()), hermes_session_id),
            )
            self._db.commit()

    def touch(self, hermes_session_id: str, *, model: str | None = None) -> None:
        with self._lock:
            if model is None:
                self._db.execute("UPDATE agy_acp_sessions SET last_active_at = ? WHERE hermes_session_id = ?",
                                 (int(time.time()), hermes_session_id))
            else:
                self._db.execute(
                    "UPDATE agy_acp_sessions SET last_active_at = ?, model = ? WHERE hermes_session_id = ?",
                    (int(time.time()), model, hermes_session_id))
            self._db.commit()

    def delete(self, hermes_session_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM agy_acp_sessions WHERE hermes_session_id = ?", (hermes_session_id,))
            self._db.commit()

    # ── memory dedupe ────────────────────────────────────────────────
    def injected_ids(self, hermes_session_id: str) -> set[str]:
        row = self.get(hermes_session_id)
        if not row:
            return set()
        try:
            values = json.loads(row.get("injected_memory_ids") or "[]")
        except (TypeError, ValueError):
            return set()
        return {str(v) for v in values if isinstance(v, str) and v}

    def add_injected_ids(self, hermes_session_id: str, ids: Iterable[str]) -> set[str]:
        new_ids = {str(i) for i in ids if str(i)}
        with self._lock:
            row = self._db.execute(
                "SELECT injected_memory_ids FROM agy_acp_sessions WHERE hermes_session_id = ?",
                (hermes_session_id,)).fetchone()
            if row is None:
                return new_ids
            try:
                current = {str(v) for v in json.loads(row[0] or "[]")}
            except (TypeError, ValueError):
                current = set()
            merged = current | new_ids
            self._db.execute(
                "UPDATE agy_acp_sessions SET injected_memory_ids = ? WHERE hermes_session_id = ?",
                (json.dumps(sorted(merged)), hermes_session_id))
            self._db.commit()
        return merged

    # ── linearity bookkeeping ────────────────────────────────────────
    def record_turn(self, hermes_session_id: str, user_hash: str) -> dict[str, Any]:
        """Fold ``user_hash`` into the row and return the state *before* it was folded."""
        with self._lock:
            row = self._db.execute(
                "SELECT last_user_hash, prev_user_hash, turn_count FROM agy_acp_sessions "
                "WHERE hermes_session_id = ?", (hermes_session_id,)).fetchone()
            before = {"last_user_hash": "", "prev_user_hash": "", "turn_count": 0}
            if row is not None:
                before = {"last_user_hash": row["last_user_hash"] or "",
                          "prev_user_hash": row["prev_user_hash"] or "",
                          "turn_count": int(row["turn_count"] or 0)}
            self._db.execute(
                "UPDATE agy_acp_sessions SET prev_user_hash = last_user_hash, last_user_hash = ?, "
                "turn_count = turn_count + 1, last_active_at = ? WHERE hermes_session_id = ?",
                (user_hash, int(time.time()), hermes_session_id))
            self._db.commit()
        return before

    # ── in-flight turn marker ────────────────────────────────────────
    def begin_inflight(self, hermes_session_id: str, *, user_hash: str = "") -> None:
        """Mark a turn as running so a restart can say the previous turn was cut off."""
        with self._lock:
            self._db.execute(
                "INSERT INTO agy_acp_inflight(hermes_session_id, started_at, user_hash, turn_count) "
                "VALUES(?, ?, ?, COALESCE((SELECT turn_count FROM agy_acp_sessions WHERE "
                "hermes_session_id = ?), 0)) ON CONFLICT(hermes_session_id) DO UPDATE SET "
                "started_at = excluded.started_at, user_hash = excluded.user_hash",
                (hermes_session_id, int(time.time()), user_hash, hermes_session_id))
            self._db.commit()

    def end_inflight(self, hermes_session_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM agy_acp_inflight WHERE hermes_session_id = ?",
                             (hermes_session_id,))
            self._db.commit()

    def take_interrupted(self, hermes_session_id: str) -> dict[str, Any] | None:
        """Return (and clear) the marker left by a turn that never finished."""
        with self._lock:
            row = self._db.execute("SELECT * FROM agy_acp_inflight WHERE hermes_session_id = ?",
                                   (hermes_session_id,)).fetchone()
            if row is None:
                return None
            self._db.execute("DELETE FROM agy_acp_inflight WHERE hermes_session_id = ?",
                             (hermes_session_id,))
            self._db.commit()
            record = dict(row)
        record["age_seconds"] = max(0, int(time.time()) - int(record.get("started_at") or 0))
        return record

    def inflight_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM agy_acp_inflight ORDER BY started_at").fetchall()
        return [dict(r) for r in rows]

    # ── housekeeping ─────────────────────────────────────────────────
    def stale_sessions(self, idle_seconds: float) -> list[str]:
        cutoff = int(time.time() - max(1.0, idle_seconds))
        with self._lock:
            rows = self._db.execute(
                "SELECT hermes_session_id FROM agy_acp_sessions WHERE last_active_at < ?", (cutoff,)).fetchall()
        return [str(r[0]) for r in rows]

    def all_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM agy_acp_sessions ORDER BY last_active_at DESC").fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
