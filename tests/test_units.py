"""Unit tests for settings, the SQLite map and turn slicing (no subprocess)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agy_acp.prompt import build_turn, last_user_message, message_text
from agy_acp.settings import Settings
from agy_acp.store import SessionStore


class SettingsTests(unittest.TestCase):
    def test_env_overrides(self) -> None:
        env = {
            "AGY_ACP_BIN": "/opt/acp-ant/bin/agy",
            "AGY_ACP_ARGS": "acp --disable-telemetry --sandbox",
            "AGY_ACP_MAX_SESSIONS": "2",
            "AGY_ACP_IDLE_SECONDS": "60",
            "AGY_ACP_PERMISSION_TIMEOUT": "5",
            "AGY_ACP_MAX_PROMPT_CHARS": "1000",
            "AGY_ACP_ALLOW_BACKGROUND": "1",
            "AGY_ACP_ON_NONLINEAR": "resync",
            "AGY_ACP_HOME": "/tmp/agy-home",
            "AGY_ACP_PROXY": "http://127.0.0.1:7890",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = Settings.from_env(home=Path("/tmp/hermes-test-home"))
        self.assertEqual(settings.command, "/opt/acp-ant/bin/agy")
        self.assertEqual(settings.args, ("acp", "--disable-telemetry", "--sandbox"))
        self.assertEqual(settings.max_sessions, 2)
        self.assertEqual(settings.idle_seconds, 60.0)
        self.assertEqual(settings.permission_timeout, 5.0)
        self.assertEqual(settings.max_prompt_chars, 1000)
        self.assertTrue(settings.allow_background)
        self.assertEqual(settings.nonlinear_action, "resync")
        self.assertEqual(settings.child_home, "/tmp/agy-home")
        self.assertEqual(settings.proxy, "http://127.0.0.1:7890")
        self.assertEqual(settings.state_db, Path("/tmp/hermes-test-home/agy_acp/sessions.db"))

    def test_bad_numbers_fall_back_to_defaults(self) -> None:
        with mock.patch.dict(os.environ, {"AGY_ACP_MAX_SESSIONS": "many"}, clear=False):
            settings = Settings.from_env(home=Path("/tmp/hermes-test-home"))
        self.assertEqual(settings.max_sessions, 4)

    def test_derived_paths(self) -> None:
        home = Path("/tmp/hermes-test-home")
        settings = Settings(home=home)
        self.assertEqual(settings.work_dir, home / "agy_acp" / "tmp")
        self.assertEqual(settings.cancel_dir, home / "agy_acp" / "cancel")
        self.assertEqual(settings.permission_dir, home / "agy_acp" / "permissions")


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.tmp.name) / "s.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_upsert_and_read(self) -> None:
        self.store.upsert("chat", acp_session_id="acp-1", cwd="/w", model="gemini-4-neon")
        row = self.store.get("chat")
        self.assertEqual(row["acp_session_id"], "acp-1")
        self.assertEqual(self.store.injected_ids("chat"), set())

    def test_injected_ids_are_merged_not_replaced(self) -> None:
        self.store.upsert("chat", acp_session_id="acp-1", cwd="/w", model="")
        self.store.add_injected_ids("chat", ["a", "b"])
        merged = self.store.add_injected_ids("chat", ["b", "c"])
        self.assertEqual(merged, {"a", "b", "c"})
        self.assertEqual(self.store.injected_ids("chat"), {"a", "b", "c"})

    def test_record_turn_tracks_the_previous_hash(self) -> None:
        self.store.upsert("chat", acp_session_id="acp-1", cwd="/w", model="")
        first = self.store.record_turn("chat", "h1")
        self.assertEqual(first["turn_count"], 0)
        second = self.store.record_turn("chat", "h2")
        self.assertEqual(second["last_user_hash"], "h1")
        self.assertEqual(second["turn_count"], 1)
        row = self.store.get("chat")
        self.assertEqual((row["last_user_hash"], row["prev_user_hash"], row["turn_count"]), ("h2", "h1", 2))

    def test_rebind_clears_per_session_memory(self) -> None:
        self.store.upsert("chat", acp_session_id="acp-1", cwd="/w", model="")
        self.store.add_injected_ids("chat", ["a"])
        self.store.record_turn("chat", "h1")
        self.store.rebind("chat", "acp-2")
        row = self.store.get("chat")
        self.assertEqual(row["acp_session_id"], "acp-2")
        self.assertEqual(row["injected_memory_ids"], "[]")
        self.assertEqual(row["turn_count"], 0)

    def test_stale_sessions(self) -> None:
        self.store.upsert("old", acp_session_id="a", cwd="/w", model="")
        with mock.patch("agy_acp.store.time.time", return_value=10**10):
            self.assertEqual(self.store.stale_sessions(60), ["old"])


class PromptTests(unittest.TestCase):
    def test_last_user_message_walks_backwards(self) -> None:
        messages = [{"role": "user", "content": "one"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "two"}]
        self.assertEqual(message_text(last_user_message(messages)), "two")

    def test_list_content_is_flattened_and_images_become_acp_blocks(self) -> None:
        content = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
        ]
        turn = build_turn([{"role": "user", "content": content}], already_injected=set(), cwd="/tmp",
                          session_id="s", turn_number=1, max_chars=10_000, work_dir=Path("/tmp"))
        self.assertEqual(turn.text, "look at this")
        kinds = [block["type"] for block in turn.blocks]
        self.assertEqual(kinds, ["text", "image"])
        self.assertEqual(turn.blocks[1]["mimeType"], "image/png")
        self.assertEqual(turn.images_dropped, 1)
        self.assertTrue(any("remote" in note or "skipped" in note for note in turn.notes))

    def test_empty_turn_produces_no_blocks(self) -> None:
        turn = build_turn([{"role": "assistant", "content": "hi"}], already_injected=set(), cwd="/tmp",
                          session_id="s", turn_number=1, max_chars=10, work_dir=Path("/tmp"))
        self.assertEqual(turn.blocks, [])

    def test_user_hash_is_stable_across_whitespace_wrapping(self) -> None:
        a = build_turn([{"role": "user", "content": "same"}], already_injected=set(), cwd="/tmp",
                       session_id="s", turn_number=1, max_chars=100, work_dir=Path("/tmp"))
        b = build_turn([{"role": "user", "content": "same"}], already_injected=set(), cwd="/tmp",
                       session_id="s", turn_number=1, max_chars=100, work_dir=Path("/tmp"))
        self.assertEqual(a.user_hash, b.user_hash)


if __name__ == "__main__":
    unittest.main()


def _yaml_available() -> bool:
    try:
        import yaml  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_yaml_available(), "PyYAML is not importable in this interpreter (-S runs)")
class SettingsSourceTests(unittest.TestCase):
    """``agy_acp:`` in config.yaml is read too, and env still wins."""

    def test_config_yaml_is_honoured_without_env(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.yaml").write_text(
                "agy_acp:\n"
                "  capability_mode: readonly\n"
                "  memory_tokens: 900\n"
                "  max_concurrent: 3\n"
                "  bridge_name: hermes-mem\n", encoding="utf-8")
            old = os.environ.pop("HERMES_HOME", None)
            os.environ["HERMES_HOME"] = str(home)
            try:
                from agy_acp.settings import Settings

                settings = Settings.from_env()
                self.assertEqual(settings.capability_mode, "readonly")
                self.assertEqual(settings.memory_tokens, 900)
                self.assertEqual(settings.max_concurrent_turns, 3)
                self.assertEqual(settings.bridge_prefix, "mcp_hermes-mem_")
            finally:
                if old is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = old

    def test_env_overrides_the_file(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.yaml").write_text("agy_acp:\n  capability_mode: readonly\n", encoding="utf-8")
            old_home = os.environ.get("HERMES_HOME")
            old_mode = os.environ.get("AGY_ACP_CAPABILITY_MODE")
            os.environ["HERMES_HOME"] = str(home)
            os.environ["AGY_ACP_CAPABILITY_MODE"] = "full"
            try:
                from agy_acp.settings import Settings

                self.assertEqual(Settings.from_env().capability_mode, "full")
            finally:
                for key, value in (("HERMES_HOME", old_home), ("AGY_ACP_CAPABILITY_MODE", old_mode)):
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class UsageParsingTests(unittest.TestCase):
    """agy 1.5.15 spells the cache fields its own way; missing a key reads as "no cache". """

    def test_the_real_frame_shape_is_understood(self) -> None:
        from agy_acp.client import _usage

        usage = _usage({"inputTokens": 2528, "outputTokens": 46, "cachedReadTokens": 8458,
                        "cachedWriteTokens": 0, "thoughtTokens": 37, "totalTokens": 11032})
        self.assertEqual(usage.prompt_tokens, 2528)
        self.assertEqual(usage.completion_tokens, 46)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 8458)
        self.assertEqual(usage.thought_tokens, 37)
        self.assertEqual(usage.total_tokens, 11032)

    def test_older_spellings_still_parse(self) -> None:
        from agy_acp.client import _usage

        self.assertEqual(_usage({"input_tokens": 10, "cached_tokens": 3}).prompt_tokens_details.cached_tokens, 3)
        self.assertEqual(_usage({"prompt_tokens": 10, "cacheReadTokens": 4}).prompt_tokens_details.cached_tokens, 4)


class BackgroundOriginTests(unittest.TestCase):
    """Delegated children are still automation: only an explicit allowlist lets them use the account."""

    def test_origins_parse_from_env_and_yaml(self) -> None:
        from agy_acp.settings import _as_list

        self.assertEqual(_as_list("subagent, cron"), ("subagent", "cron"))
        self.assertEqual(_as_list(["subagent"]), ("subagent",))
        self.assertEqual(_as_list(""), ())
        self.assertEqual(_as_list(None), ())

    def test_a_trusted_origin_passes_the_guard(self) -> None:
        import os
        import tempfile

        from agy_acp.settings import Settings

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.yaml").write_text(
                "agy_acp:\n  allow_background_origins: [subagent]\n", encoding="utf-8")
            saved = {k: os.environ.get(k) for k in ("HERMES_HOME", "HERMES_SESSION_SOURCE", "HERMES_CRON_SESSION")}
            os.environ["HERMES_HOME"] = str(home)
            os.environ.pop("HERMES_SESSION_SOURCE", None)
            os.environ.pop("HERMES_CRON_SESSION", None)
            try:
                settings = Settings.from_env()
                self.assertEqual(settings.allow_background_origins, ("subagent",))
                self.assertFalse(settings.allow_background)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class ManagerRebuildTests(unittest.TestCase):
    """Changing how the child launches must rebuild the manager, not be ignored until a restart."""

    def test_a_launch_change_replaces_the_cached_manager(self) -> None:
        import tempfile
        from pathlib import Path as _P

        from agy_acp import client as client_mod
        from agy_acp.settings import Settings

        tmp = tempfile.mkdtemp(prefix="rebuild-")
        base = dict(home=_P(tmp), state_db=_P(tmp) / "s.db", command="true", args=(), max_sessions=1)
        first = client_mod.get_manager(Settings(**base, child_home="/tmp/a"))
        same = client_mod.get_manager(Settings(**base, child_home="/tmp/a"))
        self.assertIs(first, same, "identical launch settings must reuse the manager")

        changed = client_mod.get_manager(Settings(**base, child_home="/tmp/b"))
        self.assertIsNot(first, changed, "a new account home must not be served by the old manager")
        self.assertNotIn(id(first), [id(m) for m in client_mod._MANAGERS.values()])
        changed.close_all()


class RawConfigReaderTests(unittest.TestCase):
    """The gateway's interpreter has no PyYAML; Hermes' loader drops unknown sections. Read it ourselves."""

    TEXT = """\
model:
  provider: agy-acp
agy_acp:
  child_home: /opt/acp-ant/accounts/g1
  proxy: "http://127.0.0.1:7890"
  max_concurrent: 4
  capability_mode: supervised
  allow_background_origins:
    - subagent
    - cron
  bridge_tools: [calendar-ops, agy-acp-provider]
delegation:
  provider: whatever
"""

    def test_the_block_is_read_without_yaml(self) -> None:
        from agy_acp.settings import _raw_section

        section = _raw_section(self.TEXT)
        self.assertEqual(section["child_home"], "/opt/acp-ant/accounts/g1")
        self.assertEqual(section["proxy"], "http://127.0.0.1:7890")
        self.assertEqual(section["max_concurrent"], "4")
        self.assertEqual(section["allow_background_origins"], ["subagent", "cron"])
        self.assertEqual(section["bridge_tools"], ["calendar-ops", "agy-acp-provider"])

    def test_other_sections_are_not_mixed_in(self) -> None:
        from agy_acp.settings import _raw_section

        section = _raw_section(self.TEXT)
        self.assertNotIn("provider", section)
        self.assertNotIn("model", section)

    def test_an_absent_block_is_empty(self) -> None:
        from agy_acp.settings import _raw_section

        self.assertEqual(_raw_section("model:\n  provider: x\n"), {})
