"""Phase 1: JSON-RPC duplex base, process hygiene, timeouts."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest

from agy_acp import proctree
from agy_acp.protocol import AcpError

from .support import Harness, HarnessTestCase


class HandshakeTests(HarnessTestCase):
    def test_initialize_capabilities_and_session_new(self) -> None:
        pooled = self.h.manager.session_for(self.h.session, cwd=str(self.h.work))
        caps = pooled.conn.capabilities
        self.assertTrue(caps.get("loadSession") or "load" in (caps.get("sessionCapabilities") or {}))
        self.assertTrue(pooled.acp_session_id)
        self.assertEqual(pooled.conn.agent_info.get("name"), "Mock Antigravity ACP")
        events = [entry.get("event") for entry in self.h.records()]
        self.assertIn("session/new", events)

    def test_unknown_server_method_is_refused_not_fatal(self) -> None:
        pooled = self.h.manager.session_for(self.h.session, cwd=str(self.h.work))
        with self.assertRaises(AcpError) as ctx:
            pooled.conn.request("session/does_not_exist", {})
        self.assertEqual(ctx.exception.code, -32601)
        self.assertTrue(pooled.conn.alive)  # the transport survived a protocol-level refusal

    def test_request_timeout_is_bounded(self) -> None:
        h = Harness(scenario="silent_new", connect_timeout=0.6)
        try:
            started = time.monotonic()
            with self.assertRaises(AcpError) as ctx:
                h.manager.session_for(h.session, cwd=str(h.work))
            elapsed = time.monotonic() - started
            self.assertEqual(ctx.exception.kind, "timeout")
            self.assertLess(elapsed, 5.0)
            self.assertEqual(h.manager.pool_state(), [])
        finally:
            h.close()


class StderrTests(HarnessTestCase):
    scenario = "stderr_flood"

    def test_a_full_stderr_pipe_cannot_deadlock_the_turn(self) -> None:
        # 20k lines (~1.8 MB) on stderr before the first response: without a stderr pump this hangs.
        started = time.monotonic()
        completion = self.h.turn("still works?")
        self.assertEqual(completion.choices[0].message.content, "mock reply: 1")
        self.assertLess(time.monotonic() - started, 10.0)
        stderr_logs = list((self.h.root / "logs").glob("stderr-*.log"))
        self.assertTrue(stderr_logs)
        self.assertGreater(stderr_logs[0].stat().st_size, 100_000)


class ProcessTreeTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX process-group assertion")
    def test_terminate_tree_reaps_grandchildren(self) -> None:
        script = "sh -c 'sleep 60 & sleep 60'"
        proc = subprocess.Popen(["sh", "-c", script], **proctree.spawn_kwargs())
        time.sleep(0.3)
        pgid = os.getpgid(proc.pid)
        proctree.terminate_tree(proc, timeout=2.0)
        self.assertIsNotNone(proc.poll())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("process group still has members after terminate_tree")

    def test_terminate_tree_is_idempotent(self) -> None:
        proc = subprocess.Popen([sys.executable, "-c", "pass"], **proctree.spawn_kwargs())
        proc.wait(timeout=10)
        proctree.terminate_tree(proc)  # already dead: must not raise
        self.assertFalse(proctree.is_alive(proc))

    def test_closing_a_pooled_session_kills_its_child(self) -> None:
        h = Harness()
        try:
            pooled = h.manager.session_for(h.session, cwd=str(h.work))
            pid = pooled.conn.pid
            self.assertIsNotNone(pid)
            h.manager.close_session(h.session, reason="test")
            time.sleep(0.2)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            self.assertEqual(h.manager.pool_state(), [])
            # the mapping survives, so the next turn can session/load instead of replaying
            self.assertTrue(h.manager.store.get(h.session)["acp_session_id"])
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
