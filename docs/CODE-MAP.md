# Code map

| File | Lines | Responsibility | Spec |
|---|---|---|---|
| `__init__.py` | 93 | `ProviderProfile` registration (`agy-acp`, `external_process`, `acp://agy`), static model list | 1.1 |
| `settings.py` | 174 | Every `AGY_ACP_*` knob → one dataclass; derived paths (state db, logs, spill, cancel, permission dirs) | 8 |
| `protocol.py` | 379 | Spawn, NDJSON duplex, request table by id, server→client dispatch, stderr pump, timeouts, error classification (`rate_limit`/`auth`/`transport`), `session/cancel` | 3.1, 3.2, 6.2 |
| `proctree.py` | 160 | POSIX process-group TERM→KILL→verify-empty; Windows `taskkill /F /T` + optional Job Object; console hiding | 3.1 |
| `store.py` | 175 | SQLite map `hermes_session_id → acp_session_id`, `injected_memory_ids`, turn hashes, idle queries | 3.2, 4.2 |
| `manager.py` | 612 | Pool (idle 1800 s, LRU N), `TurnGate` (Semaphore(1) with pause/resume), `session_for`, `run_turn`, `session/load` restore, resync, linearity, spill cleanup | 3, 6.1, 6.2 |
| `client.py` | 320 | OpenAI-shaped facade; turn slicing; interactive-only / aux guards; one 429 retry; `cancel()` / `close()` | 1.2, 6.2, 7.1 |
| `prompt.py` | 154 | Newest-user-turn extraction, `data:` images → ACP blocks, 160 K spill with path reference + inline head | 7.1 |
| `memory.py` | 182 | `<memory-context>` split/parse, per-item identity (id or content hash), dedupe against the store, budget (≤5 items / ≤1500 tokens), `<recalled_memory>` re-render | 4.1–4.3 |
| `permissions.py` | 275 | Read-only auto-allow; request file + answer file; 90 s deadline → explicit reject; gate release during the wait; `answer_from_cli` | 5.3 |
| `display.py` | 126 | head+tail fold with the spec's omission marker; NDJSON event log with rotation; tool-call counter/breaker | 5.2, 5.4 |
| `fs.py` | 46 | `fs/read_text_file`, `fs/write_text_file` confined to the session cwd (unused by the real agent; kept for compatibility) | 3.2 |
| `cli.py` | 113 | `status`, `sessions`, `permissions`, `answer`, `cancel` — operate on the same files the running gateway uses | — |
| `bridge_mcp.py` | ~430 | `hermes-bridge` MCP stdio server: `hermes_memory` (search/show/revise/forget) + `hermes_skill` (list/view), 4 KB paging, no local tools | 1, 5.1 |
| `mcp_mount.py` | 55 | ACP `mcpServers` entry (absolute interpreter + script, env for store/scope-recall) | 1 |
| `control_plane.py` | 105 | ≤500-token turn-1 block: persona excerpt with tool lines stripped + bridge usage rules | 2 |
| `trace.py` | 110 | queue-backed NDJSON frame trace at the stdio boundary, truncated, never blocking | 3 |
| `tests/test_bridge.py` | 300 | mount, tool cap, paging, turn-1-only injection, passthrough switch, trace | 4 |
| `tests/mcp_client.py` | 100 | MCP stdio client used to drive the bridge like an agent would | — |
| `probe.py` | 73 | Free probe: handshake + `session/new` + bogus `session/load` + `session/close`. **No prompt** | Phase-0 |
| `mock_agy_acp.py` | 388 | The mock agent: 10 scenarios, records every prompt and permission outcome to JSONL | — |
| `tests/support.py` | 117 | Isolated harness (temp home, mock, client, record readers) | — |
| `tests/test_protocol.py` | 105 | Handshake, unknown method, request timeout, stderr flood, tree kill, session close | 3.1–3.2 |
| `tests/test_pool.py` | 114 | Reuse, restart→load, load failure (raise vs reset), LRU, idle reap, busy protection | 3.1–3.2 |
| `tests/test_turns.py` | 240 | Delta-only, frozen system prompt, serialization, cancel (2 paths), 429, crash recovery, linearity | 2.1, 2.3, 6 |
| `tests/test_permissions.py` | 111 | Read-only auto-allow, timeout→deny, human answer, gate released while waiting | 5.3, 6.1 |
| `tests/test_memory_display.py` | 171 | Dedupe, budget, spill+cleanup, folding, loop breaker, cron refusal, unit cases | 4, 5.2, 5.4, 7.1 |
| `tests/test_units.py` | 141 | Settings, store, turn slicing, image parts | 8 |
| `tests/test_hermes_seam.py` | 120 | Real Hermes interpreter: profile, credential resolution, client creation, one mock turn | 1.1 |
