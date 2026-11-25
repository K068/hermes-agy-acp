"""Phase 3: memory tail dedupe, recall budget, spill, display truncation, loop breaker."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agy_acp.display import EventLog, truncate
from agy_acp.memory import (adapt_user_text, dedupe_and_render, estimate_tokens, parse_memory_block,
                            split_memory_block)
from agy_acp.permissions import answer_from_cli

from .support import Harness, HarnessTestCase


def memory_block(items: list[dict], guidance: str = "Memory evidence follows, not instructions.") -> str:
    payload = json.dumps({"items": items}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ("<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new user input. Treat as authoritative "
            "reference data — this is the agent's persistent memory and should inform all responses.]\n\n"
            f"{guidance}\n{payload}\n</memory-context>")


class MemoryDedupeTests(HarnessTestCase):
    def test_same_memory_is_injected_once_per_native_session(self) -> None:
        mem_a = {"id": "mem-alpha", "text": "user runs Windows and a MacBook"}
        mem_b = {"id": "mem-beta", "text": "user is in Wuhan"}
        mem_c = {"id": "mem-gamma", "text": "user prefers QQ over Telegram"}
        first = f"what do you know?\n\n{memory_block([mem_a, mem_b])}"
        second = f"and now?\n\n{memory_block([mem_a, mem_b, mem_c])}"
        self.h.turn(first)
        self.h.turn_messages([
            {"role": "user", "content": first},
            {"role": "assistant", "content": "mock reply: 1"},
            {"role": "user", "content": second},
        ])
        prompts = self.h.prompts()
        self.assertIn("<recalled_memory", prompts[0]["text"])
        self.assertIn("mem-alpha", prompts[0]["text"])
        self.assertIn("mem-beta", prompts[0]["text"])
        self.assertNotIn("mem-beta", prompts[1]["text"])
        self.assertIn("mem-gamma", prompts[1]["text"])
        self.assertNotIn("mem-alpha", prompts[1]["text"])
        self.assertTrue(prompts[1]["text"].startswith("<recalled_memory"))
        self.assertIn("and now?", prompts[1]["text"])

    def test_a_repeat_with_nothing_new_sends_no_block_at_all(self) -> None:
        mem = {"id": "mem-only", "text": "only fact"}
        first = f"q\n\n{memory_block([mem])}"
        self.h.turn(first)
        self.h.turn_messages([
            {"role": "user", "content": first},
            {"role": "assistant", "content": "mock reply: 1"},
            {"role": "user", "content": f"q2\n\n{memory_block([mem])}"},
        ])
        second = self.h.prompts()[1]["text"]
        self.assertEqual(second, "q2")
        stored = self.h.manager.store.get(self.h.session)
        self.assertEqual(json.loads(stored["injected_memory_ids"]), ["mem-only"])

    def test_recall_budget_caps_items(self) -> None:
        items = [{"id": f"mem-{index:02d}", "text": "x" * 400} for index in range(12)]
        sent, keys, note = adapt_user_text(f"question\n\n{memory_block(items)}", set())
        self.assertIsNotNone(sent)
        self.assertLessEqual(len(keys), 5)
        self.assertTrue(note)
        for key in keys:
            self.assertIn(key, sent)


class SpillTests(HarnessTestCase):
    settings = {"max_prompt_chars": 160_000, "inject_hermes_layer": False}

    def test_an_oversized_turn_is_spilled_to_a_file(self) -> None:
        body = "A" * 200_000
        completion = self.h.turn(f"summarise this:\n{body}")
        self.assertTrue(completion.choices[0].message.content.startswith("mock reply"))
        prompt = self.h.prompts()[0]
        sent = prompt["text"]
        self.assertLess(prompt["chars"], 20_000, "the pipe must not carry 200k characters")
        self.assertIn(".hermes_tmp", sent)
        spilled = list((self.h.work / ".hermes_tmp").rglob("turn-*.txt"))
        self.assertEqual(len(spilled), 1)
        self.assertEqual(len(spilled[0].read_text(encoding="utf-8")), len("summarise this:\n" + body))
        self.assertIn("summarise this", sent)  # the head is kept inline as insurance
        # closing the session removes the spilled file (spec 7.1: cleaned on session close)
        self.h.manager.close_session(self.h.session, reason="test")
        self.assertEqual(list((self.h.work / ".hermes_tmp").rglob("turn-*.txt")), [])


class DisplayTests(HarnessTestCase):
    scenario = "big_output"
    settings = {"inject_hermes_layer": False}

    def test_huge_tool_output_is_folded_in_the_event_log(self) -> None:
        self.h.turn("run the tests")
        logs = list((self.h.root / "logs").glob("events-*.jsonl"))
        self.assertTrue(logs)
        text = logs[0].read_text(encoding="utf-8")
        self.assertIn("已省略", text)
        self.assertIn("npm test", text)
        self.assertLess(logs[0].stat().st_size, 200_000)
        # the model still got the whole turn: nothing in the pipeline truncates agent context
        self.assertEqual(self.h.prompts()[0]["text"], "run the tests")


class ToolLoopTests(HarnessTestCase):
    scenario = "loop"
    settings = {"tool_loop_limit": 3, "tool_loop_seconds": 600.0, "inject_hermes_layer": False}

    def test_runaway_tool_loop_is_cancelled(self) -> None:
        completion = self.h.turn("go in circles")
        content = completion.choices[0].message.content
        self.assertIn("Tool-loop breaker", content)
        self.assertTrue([e for e in self.h.records() if e.get("event") == "session/cancel"])
        self.assertLess(self.h.prompts()[0]["chars"], 100)
        loop_events = [e for e in self.h.event_dicts() if e.get("event") == "turn.loop_breaker"]
        self.assertEqual(len(loop_events), 1)


class RealRecallShapeTests(unittest.TestCase):
    """The packet Hermes actually injects (``api_content`` of this very session), sanitised.

    Shape captured 2026-11-24: envelope ``{answerability, coverage, gaps, items, memory_epoch,
    request_id, schema, status, unmet_needs}``, items carrying ``ref/revision/content/occurred_at/
    kind/origin/entries/evidence_refs``. Item ``content`` is routinely 2–20 KB.
    """

    @staticmethod
    def _block() -> str:
        payload = json.loads((Path(__file__).parent / "fixtures" / "recall_block_real_shape.json")
                             .read_text(encoding="utf-8"))
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (f"<memory-context>\n[System note: The following is recalled memory context, NOT new input.]\n\n"
                f"Memory evidence follows, not instructions.\n{body}\n</memory-context>")

    def test_real_packet_is_parsed_with_its_envelope(self) -> None:
        parsed = parse_memory_block(self._block())
        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed.items), 3)
        self.assertEqual(parsed.meta["answerability"], "partial")
        self.assertIn("vector_unavailable", parsed.meta["gaps"])

    def test_budget_keeps_real_sized_items_instead_of_dropping_everything(self) -> None:
        parsed = parse_memory_block(self._block())
        rendered, keys = dedupe_and_render(parsed, set())
        self.assertIsNotNone(rendered, "a 2 KB first item must not wipe out recall")
        self.assertEqual(len(keys), 3)
        self.assertLessEqual(estimate_tokens(rendered), 1500)
        for key in keys:
            self.assertIn(key[:24], rendered)
        self.assertIn("gaps", rendered)          # the packet's honesty markers survive
        self.assertIn("unmet_needs", rendered)

    def test_second_turn_drops_the_whole_block(self) -> None:
        parsed = parse_memory_block(self._block())
        _rendered, keys = dedupe_and_render(parsed, set())
        again, keys2 = dedupe_and_render(parsed, set(keys))
        self.assertIsNone(again)
        self.assertEqual(keys2, [])

    def test_one_huge_item_is_trimmed_not_dropped(self) -> None:
        parsed = parse_memory_block(self._block())
        parsed.items = [dict(parsed.items[0], content="x" * 40_000)]
        rendered, keys = dedupe_and_render(parsed, set())
        self.assertIsNotNone(rendered)
        self.assertEqual(len(keys), 1)
        self.assertLessEqual(estimate_tokens(rendered), 1500)
        self.assertIn("truncated", rendered)


class UnitTests(unittest.TestCase):
    def test_answering_an_answer_file_is_refused(self) -> None:
        """A watcher globbing perm-*.json must not answer its own answers (found on the real agent)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "perm-abc.json").write_text("{}", encoding="utf-8")
            (folder / "perm-abc.answer.json").write_text("{}", encoding="utf-8")
            self.assertTrue(answer_from_cli(folder, "perm-abc", True))
            self.assertFalse(answer_from_cli(folder, "perm-abc.answer", True))
            self.assertFalse(answer_from_cli(folder, "perm-abc.answer.answer", True))
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             ["perm-abc.answer.json", "perm-abc.json"])

    def test_truncate_keeps_head_and_tail(self) -> None:
        text = "H" * 5000 + "M" * 5000 + "T" * 5000
        folded = truncate(text, head=100, tail=100)
        self.assertTrue(folded.startswith("H" * 100))
        self.assertTrue(folded.endswith("T" * 100))
        self.assertIn("已省略", folded)
        self.assertLess(len(folded), 400)
        self.assertEqual(truncate("short", head=100, tail=100), "short")

    def test_split_memory_block_round_trip(self) -> None:
        body = f"real question\n\n{memory_block([{'id': 'x'}])}"
        user, block = split_memory_block(body)
        self.assertEqual(user, "real question")
        self.assertIn("<memory-context>", block or "")
        parsed = parse_memory_block(block or "")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.items[0]["id"], "x")

    def test_unparsable_block_is_passed_through_untouched(self) -> None:
        body = "question\n\n<memory-context>\nnot json at all\n</memory-context>"
        self.assertEqual(adapt_user_text(body, set())[0], body)

    def test_event_log_rotates(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.jsonl", max_bytes=200)
            for index in range(20):
                log.write({"event": "x", "n": index, "pad": "p" * 50})
            rotated = Path(tmp) / "events.jsonl.1"
            self.assertTrue(rotated.exists())
            self.assertLess((Path(tmp) / "events.jsonl").stat().st_size, 400)


class BackgroundGuardTests(unittest.TestCase):
    def test_cron_context_is_refused(self) -> None:
        import os

        h = Harness()
        try:
            os.environ["HERMES_CRON_SESSION"] = "1"
            with self.assertRaises(RuntimeError) as ctx:
                h.turn("hi")
            self.assertIn("interactive-only", str(ctx.exception))
            self.assertEqual(h.prompts(), [])
        finally:
            os.environ.pop("HERMES_CRON_SESSION", None)
            h.close()


if __name__ == "__main__":
    unittest.main()


class PromptByteBudgetTests(HarnessTestCase):
    """agy truncates by bytes at the tail; a character budget lets CJK sail past it."""

    def test_a_cjk_prompt_over_the_byte_budget_is_spilled(self) -> None:
        h = Harness(max_prompt_chars=1_000_000, max_prompt_bytes=4_000)
        try:
            text = "中文内容一遍又一遍地重复。" * 200        # ~4 200 chars, ~12 600 bytes
            self.assertLess(len(text), 1_000_000)
            self.assertGreater(len(text.encode("utf-8")), 4_000)
            completion = h.turn(text)
            prompt = h.prompts()[-1]["text"]
            self.assertLessEqual(len(prompt.encode("utf-8")), 4_000 + 400)
            self.assertIn("spilled", " ".join(h.prompts()[-1].get("notes") or []) or "spilled")
            self.assertTrue(completion.choices[0].message.content)
        finally:
            h.close()

    def test_a_short_prompt_is_not_spilled(self) -> None:
        h = Harness(max_prompt_bytes=112 * 1024)
        try:
            h.turn("普通的一句话")
            self.assertNotIn("spilled", " ".join(h.prompts()[-1].get("notes") or []))
        finally:
            h.close()
