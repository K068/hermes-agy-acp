"""Phase 4: the Hermes control plane — bridge mount, turn-1 injection, switches, trace.

Everything here talks to the mock agent (or to ``hermes-bridge`` directly over stdio); no real
``agy`` process and no network call is involved.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from agy_acp.control_plane import build_control_plane, wrap_control_plane
from agy_acp.mcp_mount import BRIDGE_NAME, bridge_env, mount, server_entry
from agy_acp.memory import estimate_tokens
from agy_acp.settings import Settings

from .mcp_client import McpClient
from .support import Harness, HarnessTestCase


def _fake_skill(home: Path, name: str = "calendar-ops", body: str = "") -> None:
    folder = home / "skills" / "productivity" / name
    folder.mkdir(parents=True, exist_ok=True)
    text = textwrap.dedent(f"""\
        ---
        name: {name}
        description: Calendar scheduling rules for the user.
        ---

        # {name}

        Always confirm before writing to the calendar.
        """)
    (folder / "SKILL.md").write_text(text + body, encoding="utf-8")


def _fake_sr_cli(folder: Path) -> Path:
    """A stand-in for the scope-recall CLI: prints the documented JSON contract."""
    path = folder / "fake-scope-recall"
    path.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import json, os, sys
        argv = sys.argv[1:]
        if argv[:2] == ["memory", "search"]:
            q = argv[2] if len(argv) > 2 else ""
            items = [{"ref": "event-a@1", "content": f"fact about {q} (one)", "occurred_at": "2026-11-20T10:00:00Z"},
                     {"ref": "event-b@1", "content": f"fact about {q} (two)", "occurred_at": "2026-11-21T10:00:00Z"}]
            print(json.dumps({"items": items}))
        elif argv[:2] == ["memory", "show"]:
            print(json.dumps({"items": [{"ref": argv[2], "content": "one stored memory"}]}))
        elif argv[:2] == ["memory", "forget"]:
            print(json.dumps({"ok": True, "forgotten": argv[2:-1]}))
        elif argv[:2] == ["memory", "revise"]:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "argv.log"), "a") as fh:
                fh.write(json.dumps(argv) + "\\n")
            if "--retract" in argv:
                print(json.dumps({"action": "revise", "data": {"ref": argv[2], "retracted": True}}))
            else:
                print(json.dumps({"action": "revise", "data": {"ref": argv[2], "revision": 2,
                                                              "value": {"revision": 2}}}))
        """), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class BridgeMountTests(HarnessTestCase):
    def test_session_new_carries_the_bridge(self) -> None:
        self.h.turn("hello")
        new = [e for e in self.h.records() if e.get("event") == "session/new"][0]
        servers = new["params"].get("mcpServers") or []
        self.assertEqual(len(servers), 1)
        entry = servers[0]
        self.assertEqual(entry["name"], BRIDGE_NAME)
        self.assertTrue(entry["command"])
        self.assertTrue(entry["args"][0].endswith("bridge_mcp.py"))
        env = {item["name"]: item["value"] for item in entry["env"]}
        self.assertEqual(env["HERMES_HOME"], str(self.h.root))
        self.assertIn("AGY_ACP_BRIDGE_PAGE_BYTES", env)

    def test_pure_passthrough_mounts_nothing(self) -> None:
        h = Harness(inject_hermes_layer=False)
        try:
            h.turn("hello")
            new = [e for e in h.records() if e.get("event") == "session/new"][0]
            self.assertEqual(new["params"].get("mcpServers"), [])
            prompt = h.prompts()[0]["text"]
            self.assertEqual(prompt, "hello")
            self.assertNotIn("<hermes_control_plane>", prompt)
        finally:
            h.close()

    def test_env_carries_the_store_root_and_cli(self) -> None:
        """agy runs the bridge with HOME=its own account home, so both must be resolved host-side."""
        store = self.h.root / ".scope-recall" / "shared"
        store.mkdir(parents=True, exist_ok=True)
        settings = Settings(home=self.h.root)
        env = bridge_env(settings, session_id="s")
        self.assertEqual(env["AGY_ACP_SR_ROOT"], str(store))

    def test_entry_shape_is_acp_compatible(self) -> None:
        settings = Settings(home=Path("/tmp/x"))
        entry = server_entry(settings, session_id="s")
        self.assertEqual(set(entry), {"name", "command", "args", "env"})
        self.assertTrue(all(set(item) == {"name", "value"} for item in entry["env"]))


class BridgeProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        _fake_skill(self.home)
        self.sr = _fake_sr_cli(self.home)
        self.env = {"HERMES_HOME": str(self.home), "AGY_ACP_SR_CLI": str(self.sr),
                    "AGY_ACP_BRIDGE_PAGE_BYTES": "4096"}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_at_most_five_tools_and_memory_is_polymorphic(self) -> None:
        client = McpClient(env=self.env)
        try:
            info = client.initialize()
            self.assertEqual(info["serverInfo"]["name"], "hermes-bridge")
            tools = client.tools()
            self.assertLessEqual(len(tools), 5)
            names = [t["name"] for t in tools]
            self.assertIn("hermes_memory", names)
            memory = next(t for t in tools if t["name"] == "hermes_memory")
            props = memory["inputSchema"]["properties"]
            self.assertEqual(set(props["action"]["enum"]), {"search", "show", "revise", "forget"})
            self.assertIn("retract", props)
            self.assertIn("value", props)
            self.assertIn("page", props)
            self.assertIn("page_size", props)
            # no file/terminal/search tool may be re-declared here
            forbidden = {"read_file", "write_file", "terminal", "bash", "search_files", "run_command"}
            self.assertFalse(forbidden & set(names))
        finally:
            client.close()

    def test_mcp_handshake_survives_a_strict_client(self) -> None:
        """agy's client probes ``server/discover`` first and drops the server when that errors."""
        client = McpClient(env=self.env, handshake="strict")
        try:
            info = client.initialize()
            self.assertEqual(info["serverInfo"]["name"], "hermes-bridge")
            self.assertEqual(client.last_protocol_version, "2025-11-25")  # echoed, not downgraded
            self.assertEqual([t["name"] for t in client.tools()], ["hermes_memory", "hermes_skill"])
        finally:
            client.close()

    def test_search_reads_scope_recall_and_pages(self) -> None:
        client = McpClient(env=self.env)
        try:
            client.initialize()
            text, is_error = client.call("hermes_memory", {"action": "search", "query": "tailscale",
                                                           "page_size": 1})
            payload = json.loads(text)
            self.assertFalse(is_error)
            self.assertTrue(payload["ok"])
            self.assertEqual(len(payload["items"]), 1)
            self.assertTrue(payload["has_more"])
            self.assertEqual(payload["next_page"], 2)
            self.assertIn("event-a@1", payload["items"][0])
            page2 = json.loads(client.call("hermes_memory", {"action": "search", "query": "tailscale",
                                                             "page_size": 1, "page": 2})[0])
            self.assertIn("event-b@1", page2["items"][0])
            self.assertFalse(page2["has_more"])
        finally:
            client.close()

    def test_revise_drives_the_real_cli_contract(self) -> None:
        client = McpClient(env=self.env)
        try:
            client.initialize()
            text, is_error = client.call("hermes_memory", {"action": "revise", "ref": "event-a",
                                                           "revision": 3, "value": "it is Y, not X"})
            payload = json.loads(text)
            self.assertFalse(is_error)
            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["revision"], 2)
            argv = json.loads((self.home / "argv.log").read_text().strip().splitlines()[-1])
            self.assertEqual(argv[:3], ["memory", "revise", "event-a"])
            self.assertIn("--expected-revision", argv)
            self.assertEqual(argv[argv.index("--expected-revision") + 1], "3")
            self.assertEqual(argv[argv.index("--value") + 1], "it is Y, not X")

            text2, _ = client.call("hermes_memory", {"action": "revise", "ref": "event-a", "retract": True})
            self.assertTrue(json.loads(text2)["ok"])
            argv2 = json.loads((self.home / "argv.log").read_text().strip().splitlines()[-1])
            self.assertIn("--retract", argv2)
            self.assertNotIn("--value", argv2)
        finally:
            client.close()

    def test_revise_refuses_without_a_value_or_retract(self) -> None:
        client = McpClient(env=self.env)
        try:
            client.initialize()
            payload = json.loads(client.call("hermes_memory", {"action": "revise", "ref": "event-a"})[0])
            self.assertFalse(payload["ok"])
            self.assertIn("value", payload["error"])
            self.assertFalse((self.home / "argv.log").exists())  # nothing reached the CLI
        finally:
            client.close()

    def test_there_is_no_create_verb(self) -> None:
        """scope-recall has no create (Hermes captures turns instead), so the bridge must not fake one."""
        client = McpClient(env=self.env)
        try:
            client.initialize()
            memory = next(t for t in client.tools() if t["name"] == "hermes_memory")
            self.assertEqual(set(memory["inputSchema"]["properties"]["action"]["enum"]),
                             {"search", "show", "revise", "forget"})
            self.assertNotIn("text", memory["inputSchema"]["properties"])
            answer, is_error = client.call("hermes_memory", {"action": "save", "text": "x"})
            payload = json.loads(answer)
            self.assertTrue(is_error)
            self.assertFalse(payload["ok"])
            self.assertIn("allowed", payload)
            self.assertFalse((self.home / "agy_acp" / "bridge_notes.jsonl").exists())
        finally:
            client.close()

    def test_legacy_notes_from_the_old_save_action_stay_searchable(self) -> None:
        notes = self.home / "agy_acp" / "bridge_notes.jsonl"
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text(json.dumps({"id": "bn-1", "ts": 1.0, "session": "s",
                                     "text": "user prefers QQ over Telegram"}) + "\n", encoding="utf-8")
        client = McpClient(env={**self.env, "AGY_ACP_LEGACY_NOTES": "1"})
        try:
            client.initialize()
            found = json.loads(client.call("hermes_memory", {"action": "search", "query": "Telegram"})[0])
            self.assertTrue(any("bridge-note" in item for item in found["items"]))
        finally:
            client.close()
        # off by default: the dead-end file is not silently part of search results
        client = McpClient(env={k: v for k, v in self.env.items() if k != "AGY_ACP_LEGACY_NOTES"})
        try:
            client.initialize()
            found = json.loads(client.call("hermes_memory", {"action": "search", "query": "Telegram"})[0])
            self.assertFalse(any("bridge-note" in item for item in found["items"]))
        finally:
            client.close()

    def test_results_are_capped_at_the_page_budget(self) -> None:
        big = "x" * 200_000
        _fake_skill(self.home, name="huge", body="\n" + big)
        client = McpClient(env={"HERMES_HOME": str(self.home), "AGY_ACP_BRIDGE_PAGE_BYTES": "4096"})
        try:
            client.initialize()
            text, _ = client.call("hermes_skill", {"action": "list", "page_size": 50})
            self.assertLessEqual(len(text.encode("utf-8")), 4096)
            text, _ = client.call("hermes_skill", {"action": "view", "name": "huge"})
            self.assertLessEqual(len(text.encode("utf-8")), 4096)
            payload = json.loads(text)
            self.assertTrue(payload["has_more"])
        finally:
            client.close()

    def test_unknown_tool_and_method_are_refused_cleanly(self) -> None:
        client = McpClient(env=self.env)
        try:
            client.initialize()
            _, is_error = client.call("hermes_read_file", {"path": "/etc/passwd"})
            self.assertTrue(is_error)
            message = client.request("tools/nope")
            self.assertEqual(message["error"]["code"], -32601)
        finally:
            client.close()


class ControlPlaneInjectionTests(HarnessTestCase):
    def test_turn_one_carries_the_block_and_turn_two_does_not(self) -> None:
        _fake_skill(self.h.root)
        first = self.h.turn("turn one please")
        self.assertIn("<hermes_control_plane>", self.h.prompts()[0]["text"])
        self.assertIn("hermes_memory", self.h.prompts()[0]["text"])
        self.h.turn_messages([
            {"role": "user", "content": "turn one please"},
            {"role": "assistant", "content": first.choices[0].message.content},
            {"role": "user", "content": "turn two please"},
        ])
        second = self.h.prompts()[1]["text"]
        self.assertNotIn("<hermes_control_plane>", second)
        self.assertEqual(second, "turn two please")

    def test_the_block_stays_within_the_token_budget(self) -> None:
        settings = Settings(home=self.h.root, control_plane_max_tokens=500)
        (self.h.root / "SOUL.md").write_text("# Persona\n\n" + ("Be terse and concrete. " * 400), encoding="utf-8")
        block = build_control_plane(settings)
        self.assertGreater(len(block), 0)
        self.assertLessEqual(estimate_tokens(block), 500)
        wrapped = wrap_control_plane(block)
        self.assertTrue(wrapped.startswith("<hermes_control_plane>"))
        self.assertTrue(wrapped.endswith("</hermes_control_plane>"))

    def test_tool_instructions_are_stripped_from_the_persona_excerpt(self) -> None:
        settings = Settings(home=self.h.root)
        block = build_control_plane(
            settings,
            system_prompt="# Persona\nBe direct.\n\n## Tools\nCall read_file(path).\nUse terminal(cmd).\n",
            persona="")
        self.assertIn("Be direct", block)
        self.assertNotIn("read_file", block)
        self.assertNotIn("terminal(", block)

    def test_memory_tail_and_control_plane_coexist(self) -> None:
        block = ("<memory-context>\n[System note: recalled]\n\nMemory evidence follows.\n"
                 '{"items":[{"id":"mem-alpha","text":"user is in Wuhan"}]}\n</memory-context>')
        self.h.turn(f"what do you know?\n\n{block}")
        text = self.h.prompts()[0]["text"]
        self.assertIn("<hermes_control_plane>", text)
        self.assertIn("<recalled_memory", text)
        self.assertIn("mem-alpha", text)
        self.assertLess(text.index("<hermes_control_plane>"), text.index("<recalled_memory"))


class TraceTests(HarnessTestCase):
    settings = {"enable_ndjson_trace": True}

    def test_frames_are_traced_and_long_ones_truncated(self) -> None:
        self.h.turn("trace me")
        traces = list((self.h.root / ".hermes_traces").glob("*.ndjson"))
        self.assertEqual(len(traces), 1)
        lines = [json.loads(line) for line in traces[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(lines)
        directions = {line["dir"] for line in lines}
        self.assertEqual(directions, {"in", "out"})
        out_methods = {line["method"] for line in lines if line["dir"] == "out"}
        self.assertIn("session/new", out_methods)
        self.assertIn("session/prompt", out_methods)
        for line in lines:
            self.assertIn("ts", line)
            self.assertLessEqual(len(json.dumps(line, ensure_ascii=False)), 8000)

    def test_trace_off_by_default(self) -> None:
        h = Harness()
        try:
            h.turn("no trace")
            self.assertFalse(list((h.root / ".hermes_traces").glob("*.ndjson")))
        finally:
            h.close()

    def test_truncation_marker_for_an_oversized_frame(self) -> None:
        from agy_acp.trace import TraceWriter

        path = self.h.root / "t.ndjson"
        writer = TraceWriter(path, max_queue=8)
        writer.frame("in", {"method": "session/update", "params": {"text": "y" * 100_000}})
        writer.close()
        entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(entry["truncated"])
        self.assertLessEqual(len(json.dumps(entry, ensure_ascii=False)), 8000)


if __name__ == "__main__":
    unittest.main()
