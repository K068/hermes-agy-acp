"""Phase 2: the ``session/request_permission`` state machine (90 s default, deny on timeout)."""

from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path

from agy_acp.permissions import PermissionBroker, bridge_read, answer_from_cli, describe_for_human

from .support import Harness, HarnessTestCase


class BridgePolicyTests(unittest.TestCase):
    """The bridge arrives as ``mcp_hermes-bridge_<tool>`` with kind ``other``: classify by action."""

    def test_read_actions_are_recognised_and_writes_are_not(self) -> None:
        self.assertIs(bridge_read({"name": "mcp_hermes-bridge_hermes_memory",
                                   "rawInput": '{"action": "search", "query": "x"}'}), True)
        self.assertIs(bridge_read({"name": "mcp_hermes-bridge_hermes_skill",
                                   "rawInput": {"action": "list"}}), True)
        self.assertIs(bridge_read({"name": "mcp_hermes-bridge_hermes_memory",
                                   "rawInput": {"action": "forget", "refs": ["e"]}}), False)
        self.assertIs(bridge_read({"name": "mcp_hermes-bridge_hermes_memory",
                                   "rawInput": {"action": "revise", "ref": "e"}}), False)
        self.assertIsNone(bridge_read({"name": "run_command", "kind": "execute"}))


class GateRoutingTests(unittest.TestCase):
    """Asks go to Hermes' own approval gate when one is reachable, with the file inbox as fallback."""

    OPTIONS = [{"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
               {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"}]

    def _broker(self, gate, timeout: float = 0.3, policy: str = "none"):
        import tempfile as _tf
        tmp = _tf.mkdtemp(prefix="perm-gate-")
        events: list[dict] = []
        broker = PermissionBroker(permission_dir=Path(tmp), timeout=timeout, auto_allow_bridge=True,
                                  bridge_policy_name=policy, native_gate=gate, log=events.append)
        return broker, Path(tmp), events

    def _message(self, name: str, raw: dict, kind: str = "other") -> dict:
        return {"jsonrpc": "2.0", "id": 7, "method": "session/request_permission",
                "params": {"toolCall": {"name": name, "kind": kind, "rawInput": json.dumps(raw)},
                           "options": self.OPTIONS}}

    def test_an_approved_gate_lets_the_write_through_without_a_file(self) -> None:
        calls: list[tuple] = []

        def gate(tool: str, reason: str, rule_key: str) -> dict:
            calls.append((tool, reason, rule_key))
            return {"approved": True, "message": None, "outcome": "approved"}

        broker, folder, events = self._broker(gate)
        answer = broker.handle(self._message("mcp_hermes-bridge_hermes_memory",
                                            {"action": "revise", "ref": "e", "value": "v"}), session_id="s")
        self.assertEqual(answer["result"]["outcome"]["optionId"], "allow-once")
        self.assertEqual(calls[0][2], "agy:hermes_memory:revise")
        self.assertIn("Hermes memory", calls[0][1])
        self.assertEqual(list(folder.glob("perm-*.json")), [], "no inbox file when the gate answered")

    def test_a_denied_gate_rejects_the_tool(self) -> None:
        broker, _folder, _events = self._broker(lambda *a: {"approved": False, "outcome": "denied",
                                                            "user_summary": "You denied it"})
        answer = broker.handle(self._message("mcp_hermes-bridge_hermes_memory",
                                            {"action": "forget", "refs": ["e"]}), session_id="s")
        self.assertEqual(answer["result"]["outcome"]["optionId"], "reject-once")

    def test_a_broken_gate_falls_back_to_the_inbox(self) -> None:
        def gate(*_a) -> dict:
            raise RuntimeError("gate exploded")

        broker, folder, events = self._broker(gate, timeout=0.2)
        answer = broker.handle(self._message("mcp_hermes-bridge_hermes_memory",
                                            {"action": "revise", "ref": "e", "value": "v"}), session_id="s")
        # nobody answered the file request within 0.2 s, so it is a timeout deny — but a file existed
        self.assertEqual(answer["result"]["outcome"]["optionId"], "reject-once")
        self.assertTrue([e for e in events if e.get("event") == "permission.gate_failed"])
        self.assertTrue([e for e in events if e.get("event") == "permission.ask"])

    def test_approval_required_is_not_a_denial(self) -> None:
        """The real gate answers ``status: approval_required`` when no surface exists — use the inbox."""
        broker, _folder, events = self._broker(
            lambda *a: {"approved": False, "status": "approval_required",
                        "pattern_key": "plugin_rule:agy:hermes_memory:save"}, timeout=0.2)
        answer = broker.handle(self._message("mcp_hermes-bridge_hermes_memory",
                                             {"action": "revise", "ref": "e", "value": "v"}), session_id="s")
        self.assertEqual(answer["result"]["outcome"]["optionId"], "reject-once")  # inbox timed out
        self.assertTrue([e for e in events if e.get("event") == "permission.ask"],
                        "the file inbox must have been asked, not skipped")

    def test_no_human_surface_falls_back_instead_of_denying(self) -> None:
        broker, _folder, events = self._broker(lambda *a: {"approved": False, "outcome": "blocked"}, timeout=0.2)
        answer = broker.handle(self._message("mcp_hermes-bridge_hermes_memory",
                                            {"action": "revise", "ref": "e", "value": "v"}), session_id="s")
        self.assertEqual(answer["result"]["outcome"]["optionId"], "reject-once")
        self.assertTrue([e for e in events if e.get("event") == "permission.gate_no_surface"])

    def test_reads_never_reach_the_gate(self) -> None:
        calls: list[tuple] = []
        broker, _folder, _events = self._broker(lambda *a: calls.append(a) or {"approved": True},
                                                policy="reads")
        broker.handle(self._message("mcp_hermes-bridge_hermes_memory", {"action": "search", "query": "x"}),
                      session_id="s")
        self.assertEqual(calls, [])

    def test_bridge_writes_do_not_prompt_under_the_default_policy(self) -> None:
        """Hermes' own memory tools are ungated, so the bridge's must be too."""
        calls: list[tuple] = []
        broker, folder, events = self._broker(lambda *a: calls.append(a) or {"approved": False}, policy="all")
        for action in ({"action": "revise", "ref": "e", "value": "v"}, {"action": "forget", "refs": ["e"]}):
            answer = broker.handle(self._message("mcp_hermes-bridge_hermes_memory", action), session_id="s")
            self.assertEqual(answer["result"]["outcome"]["optionId"], "allow-once")
        self.assertEqual(calls, [], "no gate call for bridge writes by default")
        self.assertEqual(list(folder.glob("perm-*.json")), [])
        self.assertTrue([e for e in events if e.get("event") == "permission.auto_allow"])

    def test_reads_policy_still_asks_for_writes(self) -> None:
        calls: list[tuple] = []

        def gate(tool, reason, rule_key):
            calls.append((tool, reason, rule_key))
            return {"approved": True, "outcome": "approved"}

        broker, _folder, _events = self._broker(gate, policy="reads")
        broker.handle(self._message("mcp_hermes-bridge_hermes_memory", {"action": "revise", "ref": "e", "value": "v"}),
                      session_id="s")
        self.assertEqual(calls[0][2], "agy:hermes_memory:revise")

    def test_the_prompt_reads_like_a_sentence(self) -> None:
        reason, key = describe_for_human({"name": "mcp_hermes-bridge_hermes_memory",
                                          "rawInput": {"action": "forget"}})
        self.assertIn("Hermes memory", reason)
        self.assertEqual(key, "agy:hermes_memory:forget")
        reason, key = describe_for_human({"name": "run_command", "kind": "execute",
                                          "rawInput": {"CommandLine": "printf hi > e.txt"}})
        self.assertIn("printf hi", reason)
        self.assertEqual(key, "agy:run_command:execute")


class CapabilityModeTests(unittest.TestCase):
    """T3-style tiers: read-only refuses writes outright, full stops asking, supervised asks."""

    OPTIONS = [{"optionId": "allow-once", "kind": "allow_once", "name": "Allow"},
               {"optionId": "reject-once", "kind": "reject_once", "name": "Reject"}]

    def _broker(self, mode: str, gate=None):
        import tempfile as _tf
        folder = Path(_tf.mkdtemp(prefix="cap-"))
        return PermissionBroker(permission_dir=folder, timeout=0.2, bridge_policy_name="all",
                                capability_mode=mode, native_gate=gate, log=lambda e: None), folder

    def _msg(self, name: str, raw: dict, kind: str = "other") -> dict:
        return {"jsonrpc": "2.0", "id": 1, "method": "session/request_permission",
                "params": {"toolCall": {"name": name, "kind": kind, "rawInput": json.dumps(raw)},
                           "options": self.OPTIONS}}

    def test_readonly_asks_before_a_bridge_write(self) -> None:
        """Read-only means "ask first", not "impossible": the write goes to the gate."""
        called: list[tuple] = []
        broker, _folder = self._broker("readonly", gate=lambda *a: called.append(a) or {"approved": True})
        answer = broker.handle(self._msg("mcp_hermes-bridge_hermes_memory", {"action": "forget", "refs": ["e"]}),
                               session_id="s")
        self.assertEqual(answer["result"]["outcome"]["optionId"], "allow-once")
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][2], "agy:hermes_memory:forget")

    def test_readonly_asks_before_command_execution(self) -> None:
        called: list[tuple] = []
        broker, _folder = self._broker("readonly", gate=lambda *a: called.append(a) or {"approved": False})
        answer = broker.handle(self._msg("run_command", {"CommandLine": "rm -rf /tmp/x"}, kind="execute"),
                               session_id="s")
        self.assertEqual(len(called), 1)
        self.assertEqual(answer["result"]["outcome"]["optionId"], "reject-once")

    def test_readonly_still_allows_bridge_reads(self) -> None:
        broker, _folder = self._broker("readonly")
        answer = broker.handle(self._msg("mcp_hermes-bridge_hermes_memory", {"action": "search", "query": "x"}),
                               session_id="s")
        self.assertEqual(answer["result"]["outcome"]["optionId"], "allow-once")



    def test_full_mode_stops_asking_entirely(self) -> None:
        called: list[tuple] = []
        broker, folder = self._broker("full", gate=lambda *a: called.append(a) or {"approved": False})
        for name, raw, kind in (("run_command", {"CommandLine": "rm -rf build"}, "execute"),
                                ("mcp_hermes-bridge_hermes_memory", {"action": "forget", "refs": ["e"]}, "other")):
            answer = broker.handle(self._msg(name, raw, kind=kind), session_id="s")
            self.assertEqual(answer["result"]["outcome"]["optionId"], "allow-once")
        self.assertEqual(called, [])
        self.assertEqual(list(folder.glob("perm-*.json")), [])

    def test_supervised_still_asks_for_execution(self) -> None:
        called: list[tuple] = []
        broker, _folder = self._broker("supervised", gate=lambda *a: called.append(a) or {"approved": True})
        broker.handle(self._msg("run_command", {"CommandLine": "rm -rf build"}, kind="execute"), session_id="s")
        self.assertEqual(len(called), 1)


class ReadonlyTests(HarnessTestCase):
    scenario = "permission_read"

    def test_read_only_tools_are_auto_allowed_without_asking(self) -> None:
        started = time.monotonic()
        completion = self.h.turn("read something")
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual(completion.choices[0].message.content, "mock reply: 1")
        outcome = [e for e in self.h.records() if e.get("event") == "permission.outcome"][0]["outcome"]
        self.assertEqual(outcome["result"]["outcome"]["optionId"], "allow-once")


class PermissionTimeoutTests(HarnessTestCase):
    scenario = "permission"
    settings = {"permission_timeout": 0.7}

    def test_silence_is_answered_with_deny_and_never_hangs(self) -> None:
        started = time.monotonic()
        completion = self.h.turn("run rm -rf build (needs permission)")
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5.0, "the 0.7 s timeout must end the wait")
        self.assertGreaterEqual(elapsed, 0.6)
        # a denied permission ends the turn without text: agy reports stopReason "refusal"
        self.assertIn("no text", completion.choices[0].message.content)
        self.assertIn("refusal", completion.choices[0].message.content)
        outcomes = [e for e in self.h.records() if e.get("event") == "permission.outcome"]
        self.assertEqual(len(outcomes), 1)
        selected = outcomes[0]["outcome"]["result"]["outcome"]
        self.assertEqual(selected["optionId"], "reject-once")  # explicit deny, not a silent cancel
        decisions = [e for e in self.h.event_dicts() if e.get("event") == "permission.decision"]
        self.assertEqual(decisions[0]["source"], "timeout")
        self.assertEqual(decisions[0]["reason"], "User permission timed out")

    def test_the_account_gate_is_free_again_after_a_timeout(self) -> None:
        self.h.turn("first turn (needs permission)")
        # a second, unrelated chat must be able to run right away
        import os

        os.environ["HERMES_SESSION_ID"] = "other-chat"
        started = time.monotonic()
        completion = self.h.client.chat.completions.create(
            model="", messages=[{"role": "user", "content": "second"}], timeout=20.0)
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertTrue(completion.choices[0].message.content.startswith("mock reply"))


class PermissionAnswerTests(HarnessTestCase):
    scenario = "permission"
    settings = {"permission_timeout": 10.0}

    def test_a_human_answer_through_the_file_channel_is_honoured(self) -> None:
        permission_dir = Path(self.h.settings.permission_dir)
        answers: list[str] = []

        def answer_when_asked() -> None:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                for request_file in permission_dir.glob("perm-*.json"):
                    permission_id = request_file.stem
                    if answer_from_cli(permission_dir, permission_id, True, reason="approved in test"):
                        answers.append(permission_id)
                        return
                time.sleep(0.05)

        thread = threading.Thread(target=answer_when_asked)
        thread.start()
        completion = self.h.turn("do the thing (needs permission)")
        thread.join(5)
        self.assertEqual(len(answers), 1)
        self.assertEqual(completion.choices[0].message.content, "mock reply: 1")
        outcome = [e for e in self.h.records() if e.get("event") == "permission.outcome"][0]["outcome"]
        self.assertEqual(outcome["result"]["outcome"]["optionId"], "allow-once")

    def test_waiting_for_a_permission_releases_the_account_gate(self) -> None:
        os_env = __import__("os").environ
        box: dict[str, object] = {}

        def slow_chat() -> None:
            os_env["HERMES_SESSION_ID"] = "chat-permission"
            box["slow"] = self.h.client.chat.completions.create(
                model="", messages=[{"role": "user", "content": "needs permission"}], timeout=20.0)

        thread = threading.Thread(target=slow_chat)
        thread.start()
        time.sleep(0.6)  # now inside the permission wait, holding no gate
        os_env["HERMES_SESSION_ID"] = "chat-other"
        started = time.monotonic()
        completion = self.h.client.chat.completions.create(
            model="", messages=[{"role": "user", "content": "unrelated"}], timeout=20.0)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 6.0, "a waiting dialog must not starve another chat")
        self.assertTrue(completion.choices[0].message.content.startswith("mock reply"))
        thread.join(20)


if __name__ == "__main__":
    unittest.main()
