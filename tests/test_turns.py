"""Phase 2: one live turn per account, cancel, 429 backoff, crash recovery, linearity."""

from __future__ import annotations

import os
import threading
import time
import unittest

from agy_acp.manager import NonlinearTurnError
from agy_acp.protocol import AcpError

from .support import Harness, HarnessTestCase


def _layer_free(text: str) -> str:
    """Drop a leading ``<hermes_control_plane>…</hermes_control_plane>`` block (control-plane tests aside)."""
    start = text.find("<hermes_control_plane>")
    end = text.find("</hermes_control_plane>")
    if start == 0 and end != -1:
        return text[end + len("</hermes_control_plane>"):].lstrip("\n")
    return text


class DeltaTests(HarnessTestCase):
    settings = {"system_prompt": "You are the frozen agy_acp preamble."}

    def test_only_the_newest_user_turn_crosses_the_pipe(self) -> None:
        history = [{"role": "system", "content": "HERMES SYSTEM PROMPT " * 200},
                   {"role": "user", "content": "turn one question"}]
        self.h.turn_messages(history)
        history += [{"role": "assistant", "content": "mock reply: 1"},
                    {"role": "user", "content": "turn two question"}]
        self.h.turn_messages(history)
        history += [{"role": "assistant", "content": "mock reply: 2"},
                    {"role": "user", "content": "turn three question"}]
        self.h.turn_messages(history)
        prompts = self.h.prompts()
        texts = [_layer_free(p["text"]) for p in prompts]
        self.assertEqual(texts, ["turn one question", "turn two question", "turn three question"])
        self.assertLess(max(len(t) for t in texts), 40)
        # the control plane rides only on turn 1, so the delta itself stays tiny
        self.assertTrue(prompts[0]["text"].startswith("<hermes_control_plane>"))
        for prompt in prompts[1:]:
            self.assertNotIn("<hermes_control_plane>", prompt["text"])

    def test_system_prompt_is_sent_once_at_session_new(self) -> None:
        self.h.turn("hello")
        self.h.turn_messages([{"role": "user", "content": "hello"},
                              {"role": "assistant", "content": "mock reply: 1"},
                              {"role": "user", "content": "again"}])
        new = [e for e in self.h.records() if e.get("event") == "session/new"]
        self.assertEqual(new[0]["system_prompt"], self.h.settings.system_prompt)
        for prompt in self.h.prompts():
            self.assertNotIn(self.h.settings.system_prompt, prompt["text"])


class ParallelTurnsTests(HarnessTestCase):
    """Concurrency is a setting, not a religion: two slots really do run at once."""

    scenario = "default"
    settings = {"turn_timeout": 30.0, "max_concurrent_turns": 2}

    def setUp(self) -> None:
        os.environ["MOCK_AGY_TURN_SECONDS"] = "0.7"
        super().setUp()

    def tearDown(self) -> None:
        os.environ.pop("MOCK_AGY_TURN_SECONDS", None)
        super().tearDown()

    def test_two_slots_run_in_parallel(self) -> None:
        """Two *sessions* (not two calls on one client, which is serialized by that client's own lock)."""
        from agy_acp.prompt import build_turn

        starts: dict[str, float] = {}
        ends: dict[str, float] = {}
        errors: list[str] = []

        def run(session: str) -> None:
            try:
                pooled = self.h.manager.session_for(session, cwd=str(self.h.work), model="")
                prompt = build_turn([{"role": "user", "content": f"parallel {session}"}],
                                    already_injected=set(), cwd=str(self.h.work), session_id=session,
                                    turn_number=1, max_chars=160_000, work_dir=self.h.work,
                                    settings=self.h.settings)
                starts[session] = time.monotonic()
                self.h.manager.run_turn(pooled, prompt, timeout=30.0)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{session}: {type(exc).__name__}: {exc}")
            finally:
                ends[session] = time.monotonic()

        threads = [threading.Thread(target=run, args=(name,)) for name in ("par-a", "par-b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertEqual(errors, [])
        window = max(ends.values()) - min(starts.values())
        self.assertLess(window, 1.25, f"two 0.7 s turns spanned {window:.2f}s: they were serialized")
        # they really overlapped
        self.assertGreater(min(ends.values()), max(starts.values()))


class InterruptedTurnTests(HarnessTestCase):
    """A turn cut off by a provider restart is marked, reported once, then forgotten."""

    def test_the_marker_is_taken_once_and_told_to_the_model(self) -> None:
        self.h.manager.store.begin_inflight("test-session", user_hash="h")
        self.assertTrue(self.h.manager.inflight(), "the marker must be recorded before the turn runs")

        self.h.turn("what were we doing?")
        sent = self.h.prompts()[-1]["text"]
        self.assertIn("interrupted", sent.lower())
        self.assertIn("not saved", sent)
        self.assertEqual(self.h.manager.inflight(), [])

        # the next turn must be clean: the note is not sticky
        self.h.turn("carry on")
        self.assertNotIn("interrupted", self.h.prompts()[-1]["text"].lower())

    def test_the_note_can_be_switched_off(self) -> None:
        h = Harness(on_interrupted="ignore")
        try:
            h.manager.store.begin_inflight("test-session")
            h.turn("hello")
            self.assertNotIn("interrupted", h.prompts()[-1]["text"].lower())
            self.assertEqual(h.manager.inflight(), [])
        finally:
            h.close()


class ModelSelectionTests(HarnessTestCase):
    def test_requested_thought_level_reaches_the_agent(self) -> None:
        """Regression: ``"thought" in (category, id)`` is tuple membership, so the level was never sent."""
        self.h.turn("pick a model", model="gemini-4-neon-low")
        sets = [e for e in self.h.records() if e.get("event") == "session/set_config_option"]
        by_id = {e["configId"]: e["value"] for e in sets}
        self.assertEqual(by_id.get("thought_level"), "low")
        self.assertIn(by_id.get("model", "gemini-4-neon"), ("gemini-4-neon",))

    def test_switching_model_mid_session_reaches_the_agent(self) -> None:
        """Regression: the model was only applied at session creation, so a switch was silently ignored."""
        self.h.turn("first", model="gemini-4-neon-low")
        before = len([e for e in self.h.records() if e.get("event") == "session/set_config_option"])
        self.h.turn("second", model="gemini-4-neon-high")
        sets = [e for e in self.h.records() if e.get("event") == "session/set_config_option"][before:]
        by_id = {e["configId"]: e["value"] for e in sets}
        self.assertTrue(sets, "a mid-session switch must reach the agent, not be silently dropped")
        # the base model id is already selected, so the level is what changes
        self.assertEqual(by_id.get("thought_level"), "high")
        self.assertIn(by_id.get("model", "gemini-4-neon"), ("gemini-4-neon",))

    def test_an_unknown_level_keeps_the_agent_default(self) -> None:
        self.h.turn("pick a model", model="gemini-4-neon-turbo")
        sets = [e for e in self.h.records() if e.get("event") == "session/set_config_option"]
        self.assertFalse([e for e in sets if e["configId"] == "thought_level"])


class SerializationTests(HarnessTestCase):
    scenario = "default"
    settings = {"turn_timeout": 30.0, "max_concurrent_turns": 1}

    def test_a_single_slot_still_serializes(self) -> None:
        started = time.monotonic()
        threads = [threading.Thread(target=self.h.turn, args=(f"turn {i}",)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        self.assertGreater(time.monotonic() - started, 1.2)

    def setUp(self) -> None:
        os.environ["MOCK_AGY_TURN_SECONDS"] = "0.7"
        super().setUp()

    def tearDown(self) -> None:
        os.environ.pop("MOCK_AGY_TURN_SECONDS", None)
        super().tearDown()

    def test_turns_never_overlap_account_wide(self) -> None:
        results: list[float] = []
        errors: list[BaseException] = []

        def run(name: str) -> None:
            try:
                started = time.monotonic()
                os.environ["HERMES_SESSION_ID"] = name
                self.h.client.chat.completions.create(
                    model="", messages=[{"role": "user", "content": name}], timeout=30.0)
                results.append(time.monotonic() - started)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(f"chat-{i}",)) for i in range(2)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        total = time.monotonic() - started
        self.assertFalse(errors, errors)
        self.assertEqual(len(results), 2)
        self.assertGreater(total, 1.3)  # serialized: 2 × 0.7 s
        spans = [(e["started"], e["ended"]) for e in self.h.records() if e.get("event") == "session/prompt.end"]
        # the mock appends "prompt.end" after answering, so the file can lag the reply by a few ms
        deadline = time.monotonic() + 3.0
        while len(spans) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
            spans = [(e["started"], e["ended"]) for e in self.h.records() if e.get("event") == "session/prompt.end"]
        self.assertEqual(len(spans), 2)
        spans.sort()
        self.assertLessEqual(spans[0][1], spans[1][0] + 0.05)


class CancelTests(HarnessTestCase):
    settings = {"turn_timeout": 30.0}

    def _wait_active_turn(self, timeout: float = 15.0) -> None:
        """Wait until the prompt has actually been sent (session building is not a live turn)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pooled = self.h.manager._pool.get(self.h.session)
            if pooled is not None and pooled.active_turn is not None:
                return
            time.sleep(0.02)
        self.fail("the turn never became active")

    def test_cancel_reaches_the_agent_in_milliseconds(self) -> None:
        os.environ["MOCK_AGY_TURN_SECONDS"] = "20"
        os.environ["MOCK_AGY_CHUNK_DELAY"] = "0.05"
        try:
            box: dict[str, object] = {}

            def run() -> None:
                box["completion"] = self.h.client.chat.completions.create(
                    model="", messages=[{"role": "user", "content": "long job"}], timeout=30.0)

            thread = threading.Thread(target=run)
            thread.start()
            self._wait_active_turn()
            requested = time.monotonic()
            self.assertTrue(self.h.manager.cancel(self.h.session))
            thread.join(10)
            elapsed = time.monotonic() - requested
            self.assertLess(elapsed, 2.0, "cancel must not wait for the turn timeout")
            cancels = [e for e in self.h.records() if e.get("event") == "session/cancel"]
            self.assertEqual(len(cancels), 1)
            completion = box["completion"]
            self.assertEqual(completion.choices[0].finish_reason, "stop")
        finally:
            os.environ.pop("MOCK_AGY_TURN_SECONDS", None)
            os.environ.pop("MOCK_AGY_CHUNK_DELAY", None)

    def test_client_cancel_from_a_stranger_thread_targets_the_last_turn(self) -> None:
        """Hermes' interrupt loop calls ``client.cancel()`` from a thread with no session context."""
        os.environ["MOCK_AGY_TURN_SECONDS"] = "20"
        try:
            box: dict[str, object] = {}

            def run() -> None:
                box["completion"] = self.h.client.chat.completions.create(
                    model="", messages=[{"role": "user", "content": "stop me"}], timeout=30.0)

            thread = threading.Thread(target=run)
            thread.start()
            self._wait_active_turn()
            os.environ["HERMES_SESSION_ID"] = "some-other-context"
            self.assertTrue(self.h.client.cancel())
            thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertTrue([e for e in self.h.records() if e.get("event") == "session/cancel"])
        finally:
            os.environ.pop("MOCK_AGY_TURN_SECONDS", None)

    def test_cancel_while_the_session_is_still_being_built_sends_nothing(self) -> None:
        """Stop pressed during spin-up: the prompt must never reach the agent (found on the real CLI)."""
        original = self.h.manager.session_for

        def slow_session_for(*args, **kwargs):
            time.sleep(0.6)
            return original(*args, **kwargs)

        self.h.manager.session_for = slow_session_for  # type: ignore[method-assign]
        box: dict[str, object] = {}

        def run() -> None:
            box["completion"] = self.h.turn("long job")

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.2)
        self.assertTrue(self.h.client.cancel())
        thread.join(15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.h.prompts(), [])  # zero prompts crossed the pipe
        content = box["completion"].choices[0].message.content or ""
        self.assertIn("cancelled", content.lower())
        cancelled = [e for e in self.h.event_dicts() if e.get("event") == "turn.cancelled_before_send"]
        self.assertEqual(len(cancelled), 1)

    def test_cancel_with_no_active_turn_is_a_no_op(self) -> None:
        self.assertFalse(self.h.manager.cancel("never-started"))
        self.assertFalse(self.h.client.cancel())


class RateLimitTests(HarnessTestCase):
    scenario = "ratelimit"

    def test_one_429_triggers_exactly_one_retry(self) -> None:
        completion = self.h.turn("try me")
        self.assertEqual(completion.choices[0].message.content, "mock reply: 1")
        self.assertEqual(len(self.h.prompts()), 2)  # the failed attempt + the retry, and no more

    def test_retry_can_be_disabled(self) -> None:
        h = Harness(scenario="ratelimit", retry_429=False)
        try:
            with self.assertRaises(AcpError) as ctx:
                h.turn("try me")
            self.assertTrue(ctx.exception.rate_limited)
            self.assertEqual(len(h.prompts()), 1)
        finally:
            h.close()


class CrashTests(HarnessTestCase):
    scenario = "crash"

    def test_child_death_drops_the_session_and_the_next_turn_reloads(self) -> None:
        with self.assertRaises(AcpError) as ctx:
            self.h.turn("boom")
        self.assertIn(ctx.exception.kind, ("transport", "server"))
        self.assertEqual(self.h.manager.pool_state(), [])
        os.environ["MOCK_AGY_SCENARIO"] = "default"
        completion = self.h.turn("after the crash")
        self.assertEqual(completion.choices[0].message.content, "mock reply: 1")


class LinearityTests(HarnessTestCase):
    def test_regenerating_a_turn_is_refused(self) -> None:
        self.h.turn("same question")
        with self.assertRaises(NonlinearTurnError) as ctx:
            self.h.turn("same question")
        self.assertIn("cannot rewind", str(ctx.exception))
        self.assertEqual(len(self.h.prompts()), 1)  # nothing was sent the second time

    def test_resync_opens_a_fresh_native_session(self) -> None:
        h = Harness(nonlinear_action="resync")
        try:
            h.turn("same question")
            first = h.manager.pool_state()[0]["acp_session"]
            h.turn("same question")
            second = h.manager.pool_state()[0]["acp_session"]
            self.assertNotEqual(first, second)
            self.assertEqual(len([e for e in h.records() if e.get("event") == "session/new"]), 2)
        finally:
            h.close()

    def test_linearity_check_is_off_when_disabled(self) -> None:
        h = Harness(enforce_linear=False)
        try:
            h.turn("same question")
            h.turn("same question")
            self.assertEqual(len(h.prompts()), 2)
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
