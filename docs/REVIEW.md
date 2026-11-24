# Review guide — `agy_acp` 0.1.0

Everything here is reviewable **without** the subscription, the network, or Hermes:
the whole test suite drives a mock stdio agent (`mock_agy_acp.py`), and the only real-CLI
artefact is `evidence/probe-real.log` (handshake + `session/new` + `session/load` + `session/close`;
**no prompt was ever sent**, so no quota was spent).

```
CODE-MAP.md          file → responsibility → spec section
evidence/
  test-run.log       57 tests, OK (the run that ships with this package)
  probe-real.log     real agy 1.5.15 answers, no prompt sent
  real-turn.md       the ONE real session/prompt spent (2026-11-24): content, usage, timing, and the
                     model-selection bug it exposed
  plugins-validate.txt  `hermes plugins validate` output (2 known advisories)
docs/
  SPEC.md            the v3.0 Route A spec, transcribed (reference for every claim below)
  REVIEW.md          this file
```

## 1. What to check first (highest-risk claims)

| # | Claim | Where to look | How to falsify |
|---|---|---|---|
| 1 | Only the newest user turn crosses the pipe — no history replay | `prompt.build_turn`, `client._create_chat_completion` | `test_turns.DeltaTests` asserts the three prompt `text` values are exactly `turn one/two/three question` |
| 2 | The frozen system prompt is sent once, in `session/new` | `manager._create` (`_meta.antigravity.systemPrompt.append`) | `DeltaTests::test_system_prompt_is_sent_once_at_session_new` |
| 3 | `session/load` restores after a kill, and failure **raises** instead of replaying | `manager._create` | `test_pool.PoolTests::test_restart_loads_*`, `::test_load_failure_raises_instead_of_replaying` |
| 4 | Turns never overlap account-wide; a permission wait releases the gate | `manager.TurnGate`, `permissions.PermissionBroker` | `SerializationTests` (mock records start/end per prompt, asserted non-overlapping), `test_waiting_for_a_permission_releases_the_account_gate` |
| 5 | Permission silence → explicit `reject` after the deadline, turn ends cleanly | `permissions.handle`/`_await`/`_respond` | `PermissionTimeoutTests` (deadline set to 0.7 s; asserts `optionId == "reject-once"` and `source == "timeout"`) |
| 6 | Stop is millisecond-level and survives the "stranger thread" caller | `client.cancel`, `manager.cancel`, `manager._make_tick` | `CancelTests::test_client_cancel_from_a_stranger_thread_targets_the_last_turn` |
| 7 | Memory items already in the native context are not re-sent | `memory.adapt_user_text`, `store.add_injected_ids` | `MemoryDedupeTests` (second prompt must not contain `mem-alpha`/`mem-beta`, must contain `mem-gamma`) |
| 8 | 160 K+ turns spill to a file, cleaned on session close | `prompt.spill_text`, `manager._cleanup_spill` | `SpillTests` (200 000 chars) |
| 9 | 1 MB tool output cannot reach the chat DB or a UI unfolded | `display.truncate`, `display.summarize_update`, `manager._route_update` | `DisplayTests` (event log < 200 KB, contains `已省略`) |
| 10 | A runaway tool loop is cancelled, not charged | `display.ToolCounter`, `manager._make_tick` | `ToolLoopTests` (limit 3) |
| 11 | 429 → exactly one retry | `client._run_with_retry` | `RateLimitTests` (asserts 2 prompts, and 1 when disabled) |
| 12 | Cron / subagents / auxiliary traffic are refused | `client.background_origin`, `client._auxiliary_call_frame` | `BackgroundGuardTests::test_cron_context_is_refused`; aux guard is code-read only (see caveat 5) |
| 13 | The real Hermes seam resolves the profile and builds this client | `__init__.AgyAcpProfile` | `test_hermes_seam` (runs in the real Hermes interpreter, throwaway `HERMES_HOME`) |
| 14 | The bridge exposes only Hermes-only capability (memory, skills), never file/shell/search, and caps every result at 4 KB | `bridge_mcp.tool_schemas`, `page_items`, `render` | `test_bridge.BridgeProtocolTests` including the 200 KB skill body case |
| 15 | Real scope-recall packets (2–20 KB per item) are budgeted by truncating excerpts, never by dropping recall wholesale | `memory.dedupe_and_render`, `tests/fixtures/recall_block_real_shape.json` | `test_memory_display.RealRecallShapeTests` (0.2.1: the old code returned `None` for every real packet) |
| 16 | Active writes use scope-recall's real contract (`revise` with `--expected-revision`/`--value`/`--retract`), never an invented create | `bridge_mcp.memory_revise` | `test_bridge.test_revise_drives_the_real_cli_contract`; argv shape also accepted by the real CLI (answers `SOURCE_MISSING`, not a usage error) |
| 17 | The bridge survives a strict MCP handshake (`server/discover` first, newest protocol) | `bridge_mcp.py` `SUPPORTED_VERSIONS`, `server/discover` branch | `test_bridge.test_mcp_handshake_survives_a_strict_client`; real agy log went from `invalid request` to no error, and a real turn called `mcp_hermes-bridge_hermes_memory` |
| 18 | Bridge reads do not stall on a permission nobody answers; writes still ask | `permissions.bridge_read`, `AGY_ACP_AUTO_ALLOW_BRIDGE` | `test_permissions.BridgePolicyTests`; real turn F went from 101 s + denial to 20.8 s with a real answer |
| 19 | A mid-session model switch reaches the agent | `manager.session_for` + config-cache refresh | `test_turns.test_switching_model_mid_session_reaches_the_agent`; real run reported `gemini-4-neon-high` |
| 19a | A restored session keeps the control plane (`mcpServers` on `session/load`, not `[]`) | `manager._create` load branch | `test_pool.test_restart_loads_the_native_session_instead_of_replaying`; real agy: before the fix the load process logged `Loading/Resuming` with no `Preloading`, after it logs `Loading/Resuming` + `Preloading 1 MCP servers` |
| 19b | Bridge memory actions behave like Hermes' own memory tools (no prompt) unless asked otherwise | `permissions.bridge_policy`, `AGY_ACP_BRIDGE_APPROVAL` | `test_permissions.GateRoutingTests.test_bridge_writes_do_not_prompt_under_the_default_policy`, and `…test_reads_policy_still_asks_for_writes` for the opt-in |
| 20 | Approval prompts go through Hermes' own gate, so they appear in the UI the user already uses, and `always` is remembered by Hermes rather than by us | `permissions.PermissionBroker.handle` → `manager._native_gate`, `request_tool_approval` | `test_permissions.GateRoutingTests` (approve, deny, raise→inbox, `approval_required`→inbox, reads never gate); live call returned `status: approval_required` in a headless process, which is why that case falls back |
| 21 | Two turns really run at once when the slot count is 2, and one slot still serializes | `manager.TurnGate(capacity)`, `AGY_ACP_MAX_CONCURRENT` | `test_turns.ParallelTurnsTests` (two sessions overlap) + `SerializationTests` (one slot does not) |
| 22 | `readonly` refuses rather than prompts, `full` stops asking, `supervised` keeps asking | `permissions.PermissionBroker.handle` | `test_permissions.CapabilityModeTests` (4 cases, no gate call, no inbox file) |
| 23 | A turn killed by a restart is announced once, not repeated | `store.begin_inflight/end_inflight/take_interrupted`, `client` note | `test_turns.InterruptedTurnTests` (note present once, gone next turn; `ignore` honoured) |
| 24 | Windows process-tree kill works on a real Windows box | `proctree.terminate_tree` (`taskkill /F /T` + Job Object) | PC run: child+grandchild heartbeat froze after the kill, 3.19 s, `is_alive=false` |
| 25 | Settings come from `config.yaml` too, env winning | `settings.config_section` + `_pick` | `test_units.SettingsSourceTests` |
| 26 | `readonly` asks instead of refusing, for both the bridge and agy's own writes | `permissions.PermissionBroker.handle` | `test_permissions.CapabilityModeTests.test_readonly_asks_before_a_bridge_write` / `…command_execution` |
| 27 | Cache usage is parsed from agy's real field names (`cachedReadTokens`) | `client._usage` | `test_units.UsageParsingTests`; real frames: 8 458 cached / 2 528 input on a warm turn |
| 28 | A real `agy_acp` turn is written into scope-recall | Hermes' capture hooks (unchanged) | real run: 4 `source_events`, searchable by token, tagged with the entry |
| 29 | A delegated child reaches the account only through the origin allowlist, and the profile names an absolute `agy` | `client._create_chat_completion` guard, `settings.allow_background_origins`, `__init__._resolve_binary` | `test_hermes_seam.BackgroundGuardTests` (refused by default, allowed with the origin); `resolve_runtime_provider('agy-acp')` in a fresh process returns `command=/opt/acp-ant/bin/agy` |
| 30 | The control plane is turn-1 only and stays ≤500 tokens | `control_plane.py`, `prompt.build_turn` | `ControlPlaneInjectionTests` (turn 2 must be the bare delta), `test_the_block_stays_within_the_token_budget` |
| 16 | Zero core edits | — | `git -C /root/.hermes/hermes-agent status --porcelain` is empty in the workspace this was built in; the package touches nothing outside its own directory |

## 2. Run it

```bash
tar xzf agy_acp-0.1.1.tar.gz
cd agy_acp
python3 -m unittest discover -s tests -t .        # stdlib only; no pytest, no network
# 57 tests, ~25 s
```

Requirements: Python ≥ 3.10 (developed on 3.14), POSIX for the process-group assertions. The
Windows-specific paths (`taskkill /F /T`, Job Object) are guarded and skip there by design —
see caveat 7.

## 3. Where the spec and the code deliberately differ

Read these before filing a "spec violation".

1. **`asyncio` (§2.4, §3, §6.1) → threads.** Hermes' provider seam is a *synchronous*
   `client.chat.completions.create()` called from the agent loop's worker thread; the in-tree
   `agent/acp_client.py` is synchronous for the same reason. The transport keeps the properties the
   spec asks for (event-driven reads, zero idle polling, millisecond cancel) with one reader thread
   and a request table. `asyncio` inside a background loop would add a hop and no property.
2. **§7.2 "streaming event mapping" is not implemented as SSE-like streaming.** `supports_streaming`
   is `False` (as every in-tree ACP provider); the answer arrives when the turn ends. `stream=True`
   callers get the two-chunk shape Hermes expects. Reason: an incremental response object for a
   stateful client is an untested path in core, and shipping it untested is worse than not shipping it.
3. **§4.1 "intercept scope-recall's system-prompt hook" was unnecessary.** In *this* Hermes
   (v0.23.x) the memory provider's `prefetch()` text is appended to the **current user message**
   (`agent/turn_context.compose_user_api_content` → `build_memory_context_block`), not to the system
   prompt. So the tail-injection is already the framework's behaviour; what the provider adds is
   parsing/dedupe/budget (`memory.py`). No hook is disabled, so nothing else changes behaviour.
4. **§5.3 / §6.2 "disable the edit/regenerate buttons" is not possible from a plugin.** Enforced one
   layer lower: a re-sent turn is refused *before* anything is sent (`manager.check_linearity` →
   `raise_nonlinear`), with `AGY_ACP_ON_NONLINEAR=resync` as the explicit escape hatch. Reviewers
   should treat "the UI buttons still exist" as a known gap, not a bug.
5. **`interactive_only` is a policy guard.** Cron (`HERMES_CRON_SESSION`), gateway background
   sources and delegated subagents are refused via contextvars. Auxiliary traffic (compression,
   titling, vision) is detected by walking stack frames for `auxiliary_client.py`, because core
   passes providers no `aux_task` flag — this is the weakest check in the package and is documented
   as such (`client._auxiliary_call_frame`).
6. **§4.3 recall knobs (`top_k`, `similarity_threshold`) are not configurable from here.**
   `top_k`/token budget are enforced in `memory.dedupe_and_render`; the *retrieval* threshold lives
   in scope-recall's own config and is not re-implemented.
7. **Windows is written but not executed here.** `proctree.py` implements `taskkill /F /T` plus an
   optional Job Object; only the POSIX process-group path has a test on this host. Treat the Windows
   path as "reviewed, unrun".
8. **No MCP passthrough (§5.1).** `session/new` sends `mcpServers: []`. Deliberate: fewer tools,
   smaller fixed cost. A whitelist would be an additive change.

## 4. Things a reviewer should *not* assume

- `evidence/probe-real.log` proves the protocol shape and that this account is signed in.
  `evidence/real-turn.md` records **five** real prompts over three rounds (turn 1 `OK`; a resumed
  turn 2 `OK2`; two turns in one live process; one tool-calling turn).
- **Prefix cache: works in-process, not across a resume.** Back-to-back turns on one child read 6657
  of 8822 context tokens from cache (≈75.5%, turn 2 was 2.92 s vs 17.06 s); the same session resumed
  27 minutes later reported `cache_read_tokens: 0`. Budget ~8.7–8.9k new input on the first turn and
  after every restart, ~2.2k on later turns. These are *this account's* numbers, not a promise.
- **Native context retention is proven**: turn 2 answered a question about turn 1 with no history
  sent by Hermes.
- **The permission path is proven against the CLI**: an `execute` tool asked first, the file-channel
  answer was accepted, the command ran and wrote the file.
- **Cancel works in both windows** (live turn: 0.20 s to return, `stopReason: cancelled`; still
  spinning up: no prompt sent, zero tokens) and **the spill path runs on the real agent** (path
  reference in the stored prompt, file cleaned on close).
- **The real agent loop drives this client**: `AIAgent.run_conversation` → one API call, `ECHO`,
  `provider=agy-acp`, `input_tokens 8699` taken from the provider, child closed by `agent.close()`.
  With the plugin absent from `HERMES_HOME`, the same agent degrades to an HTTP client on
  `acp://agy` and retries 5 cycles — that is the mis-install signature.
- **Reap + restore**: idle reaping kills the child, keeps the mapping, and a fresh manager restores
  the same native session in 1.75 s with zero prompts.
- The 8714 input tokens on a 22-character message are the agent's own system prompt + tool
  declarations, not provider overhead — the number to watch if "fixed cost" ever becomes a concern.
- The 23 model ids in `__init__.FALLBACK_MODELS` are from the probe (2026-11-23). The live account
  advertised **27** entries a day later (it grew `gemini-4-carbon`, `gemini-pro-orchestrator-argon`,
  …), so treat the list as a convenience fallback, not a catalog.
- Test timings are deliberately loose (e.g. cancel < 2 s for a 20 s turn) so a slow CI box does not
  flake; they are not performance assertions.
- Model selection was found broken by the real turn and fixed in 0.1.1 (`"thought" in (category, id)`
  is tuple membership, not substring). The regression test only exists from 0.1.1 on; anything earlier
  ran at the account's default thought level.

## 5. Known edges worth a second pair of eyes

- **Locking was fixed in 0.1.2 and deserves a hard look**: `session_for` now uses a separate
  `_create_lock` for spawning, takes the pool lock only to read/write the dict, and tears down
  outside it. The previous shape deadlocked for 120 s against the real CLI on every `session/load`
  (agy broadcasts `session_info_update` before answering). Check `_ensure_capacity_locked`,
  `_take`/`_teardown`, and `close_all` for a path that reintroduces "spawn while holding the pool lock".

- `manager.TurnGate.pause/resume` are thread-agnostic by design (the permission request is answered
  on the reader thread while the turn thread holds the gate). Check the pause/resume/release
  interleaving for a double-release or a starved resume.
- `protocol.AgyConnection.request(..., tick=...)` loops in `tick_interval` slices; check the timeout
  arithmetic when `tick` is None vs. present, and that a pending entry is always popped.
- `memory.parse_memory_block` scans for the first position whose remainder parses as JSON. Adversarial
  user text containing a `{` is fine (it fails to parse and the scan continues), but a *valid* JSON
  prefix would be treated as the memory payload — see `test_unparsable_block_is_passed_through_untouched`
  for the intended fallback.
- `client._last_session` is the only cross-thread state shared between the turn and
  `cancel()`; it is written from the turn thread and read from a stranger thread without a lock
  (a string assignment — atomic under CPython, worth a look on other implementations).
