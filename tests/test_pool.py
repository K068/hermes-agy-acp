"""Phase 1/2: pool reuse, LRU + idle eviction, ``session/load`` recovery."""

from __future__ import annotations

import os
import time
import unittest

from agy_acp.manager import LoadFailedError

from .support import Harness, HarnessTestCase


class PoolTests(HarnessTestCase):
    def test_one_process_serves_many_turns(self) -> None:
        first = self.h.manager.session_for(self.h.session, cwd=str(self.h.work))
        pid = first.conn.pid
        self.h.turn("one")
        self.h.turn_messages([{"role": "user", "content": "one"},
                              {"role": "assistant", "content": "mock reply: 1"},
                              {"role": "user", "content": "two"}])
        second = self.h.manager.session_for(self.h.session, cwd=str(self.h.work))
        self.assertEqual(second.conn.pid, pid)
        self.assertEqual(len([e for e in self.h.records() if e.get("event") == "session/new"]), 1)
        self.assertEqual(second.acp_session_id, first.acp_session_id)

    def test_restart_loads_the_native_session_instead_of_replaying(self) -> None:
        self.h.turn("first")
        old_pid = self.h.manager.pool_state()[0]["pid"]
        old_acp = self.h.manager.pool_state()[0]["acp_session"]
        self.h.manager.close_session(self.h.session, reason="simulated restart")
        time.sleep(0.2)
        with self.assertRaises(ProcessLookupError):
            os.kill(old_pid, 0)
        completion = self.h.turn_messages([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "mock reply: 1"},
            {"role": "user", "content": "after restart"},
        ])
        self.assertEqual(completion.choices[0].message.content, "mock reply: 2")
        events = [e for e in self.h.records() if e.get("event") in ("session/load", "session/new")]
        self.assertEqual([e["event"] for e in events], ["session/new", "session/load"])
        # a restored chat must keep the control plane: agy honours mcpServers on session/load
        for record in events:
            servers = record.get("mcpServers") or (record.get("params") or {}).get("mcpServers") or []
            self.assertEqual(len(servers), 1, f"{record['event']} lost the hermes-bridge mount")
            self.assertEqual(servers[0]["name"], "hermes-bridge")
        self.assertEqual(events[-1]["session"], old_acp)
        self.assertEqual(self.h.prompts()[-1]["text"], "after restart")

    def test_load_failure_raises_instead_of_replaying(self) -> None:
        self.h.manager.store.upsert(self.h.session, acp_session_id="gone-forever",
                                    cwd=str(self.h.work), model="")
        with self.assertRaises(LoadFailedError) as ctx:
            self.h.manager.session_for(self.h.session, cwd=str(self.h.work))
        self.assertIn("will not replay", str(ctx.exception))
        self.assertEqual(self.h.manager.pool_state(), [])

    def test_load_failure_can_be_opted_into_a_fresh_session(self) -> None:
        h = Harness(on_load_failure="reset")
        try:
            h.manager.store.upsert(h.session, acp_session_id="gone-forever", cwd=str(h.work), model="")
            completion = h.turn("fresh start")
            self.assertEqual(completion.choices[0].message.content, "mock reply: 1")
            events = [e.get("event") for e in h.event_dicts()]
            self.assertIn("session.load_failed", events)
            self.assertIn("session.new", events)
        finally:
            h.close()

    def test_lru_eviction_keeps_the_mapping(self) -> None:
        h = Harness(max_sessions=1)
        try:
            first = h.manager.session_for("chat-a", cwd=str(h.work))
            first_pid = first.conn.pid
            os.environ["HERMES_SESSION_ID"] = "chat-b"
            h.client.chat.completions.create(model="", messages=[{"role": "user", "content": "b"}],
                                             timeout=20.0)
            self.assertEqual(len(h.manager.pool_state()), 1)
            time.sleep(0.2)
            with self.assertRaises(ProcessLookupError):
                os.kill(first_pid, 0)
            self.assertTrue(h.manager.store.get("chat-a")["acp_session_id"])
            os.environ["HERMES_SESSION_ID"] = "chat-a"
            h.client.chat.completions.create(model="", messages=[{"role": "user", "content": "back"}],
                                             timeout=20.0)
            loads = [e for e in h.records() if e.get("event") == "session/load"]
            self.assertEqual(len(loads), 1)
        finally:
            h.close()

    def test_idle_session_is_reaped(self) -> None:
        h = Harness(idle_seconds=0.1)
        try:
            pooled = h.manager.session_for(h.session, cwd=str(h.work))
            pid = pooled.conn.pid
            time.sleep(0.3)
            reaped = h.manager.reap_idle()
            self.assertEqual(reaped, [h.session])
            time.sleep(0.2)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            h.close()

    def test_busy_session_is_not_reaped(self) -> None:
        h = Harness(idle_seconds=0.05, turn_timeout=20.0)
        try:
            pooled = h.manager.session_for(h.session, cwd=str(h.work))
            pooled.last_used = 0.0  # pretend idle
            pooled.active_turn = object()  # type: ignore[assignment]
            self.assertEqual(h.manager.reap_idle(), [])
            pooled.active_turn = None
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
