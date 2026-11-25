# agy_acp — native stateful Antigravity provider for Hermes

Route A: **one live `agy acp` child per chat**, driven directly over stdin/stdout JSON-RPC. Hermes
sends the newest user turn only; history, tool state and the prefix cache stay inside the agent.
The ACP-Ant HTTP gateway (`/opt/acp-ant/acp_ant.py`:8095 + `/opt/acp-ant-shim`:8098) is not used by
this provider at all.

```
Hermes agent loop
   └── client.chat.completions.create(model, messages=[…])        # agent/acp_client.py contract
        └── AgyAcpClient            client.py     turn slicing, guards, OpenAI facade
             ├── AgyAcpSessionManager  manager.py  pool (idle+LRU), Semaphore(1) gate, turn execution
             │    ├── AgyConnection    protocol.py NDJSON duplex, request table, stderr pump, frame trace
             │    │     └── `agy acp`  (one child per chat, own process group / Job Object)
             │    ├── PermissionBroker permissions.py  90 s ask→deny state machine
             │    ├── SessionStore      store.py      hermes_session_id → acp_session_id (SQLite)
             │    └── EventLog          display.py    folded tool activity, loop breaker
             ├── memory/prompt          memory.py, prompt.py   scope-recall tail + dedupe, 160 K spill
             └── control plane          control_plane.py + bridge_mcp.py + mcp_mount.py
```

## Settings: env or config.yaml

Every knob has an `AGY_ACP_*` environment variable and an optional `agy_acp:` section in the
profile's `config.yaml`; **the environment wins** when both are set:

```bash
hermes config set agy_acp.max_concurrent 3
hermes config set agy_acp.capability_mode readonly
```

## Concurrency

`AGY_ACP_MAX_CONCURRENT` (default 4) is how many turns may run at once against the single account.
It is a real limit, not a queue: when every slot is busy the request waits up to its own timeout and
then fails with a message naming the setting, so nothing is silently dropped. Raise it with care —
parallel prompts are the fastest route to a 429 on one consumer subscription. With one slot the old
behaviour returns: a permission dialog temporarily hands its slot back so it cannot starve another
chat.

### Delegated children

`AGY_ACP_ALLOW_BACKGROUND_ORIGINS` names the automations allowed to drive the account even though
`allow_background` is off: `subagent`, `cron`, `kanban`, `webhook`, `background`, `heartbeat`
(comma list in the environment, a YAML list in `config.yaml`). Empty by default — spec §1.2 makes
this provider human-only, and opening it to `delegate_task` is a deliberate, visible choice:

```bash
hermes config set agy_acp.allow_background_origins subagent
```

The child is launched with `HOME` = `agy_acp.child_home` and the proxy from `agy_acp.proxy`; both are
read from `config.yaml` on every call, and a change to a launch setting (binary, args, account home,
proxy, bridge name, concurrency, capability mode, origins) rebuilds the session manager instead of
waiting for a restart. Delegation still resolves the binary through Hermes' external-process ladder,
which reads `process_command` from the provider profile **at gateway start**. The profile resolves it to an
absolute path (env `AGY_ACP_BIN`/`AGY_BIN` → `PATH` → `/opt/acp-ant/bin/agy`), so after changing where
`agy` lives — or after this plugin's first install — restart the gateway once.

## Capability modes (T3 Code's idea, scaled down)

| mode | agy's shell/file writes | bridge memory ops | use |
|---|---|---|---|
| `readonly` | allowed to read; **a write asks** (gate/inbox) | reads allowed; a write asks | "nothing changes unless I say so" |
| `supervised` *(default)* | shell uses the Hermes terminal guard, file writes ask | allowed (as in Hermes) | normal use |
| `full` | allowed | allowed | you trust the session completely |

The mode is read once when the session's broker is built; nothing inside a turn can escalate it.
`readonly` never refuses outright — it routes the write to the same human gate as everything else,
so a refusal is a decision somebody made, not a wall.

Usage numbers: agy reports `inputTokens` / `outputTokens` / `cachedReadTokens` / `thoughtTokens`;
the provider maps them onto the OpenAI shape (`prompt_tokens_details.cached_tokens`), so cache hits
are visible — measured 77 % on a warm second turn.

## A turn that died with the provider

Before every `session/prompt` the session is marked in `agy_acp_inflight`; the mark is cleared in the
same `finally` that frees the turn slot. If the mark is still there next time (gateway restarted
mid-turn), the model gets one line — *"your previous turn was interrupted … it was not saved"* —
appended to that turn's delta, so it does not pretend the work happened. `AGY_ACP_ON_INTERRUPTED=ignore`
turns the note off; `cli status` lists any live marks.

## The control plane (Hermes overlay, opt-out with `AGY_ACP_INJECT_HERMES_LAYER=0`)

agy keeps every local capability: file edits, code search, shell. Hermes therefore adds **nothing**
in that space — it adds the two things agy cannot see:

| Piece | What it is | Cost per turn |
|---|---|---|
| `hermes-bridge` MCP (stdio, mounted in `session/new`) | 2 polymorphic tools: `hermes_memory` (`search`/`show`/`revise`/`forget`, paged — no create verb) and `hermes_skill` (`list`/`view`, paged). Hard cap 4 KB per result with `has_more`/`next_page` | 0 tokens until called; ~2 tool schemas |
| Turn-1 `<hermes_control_plane>` block | ≤500-token excerpt of the persona (tool-instruction lines stripped) + "when to use the bridge" | once, on the first turn only |

Rules the implementation enforces:

- **Turn 1 only.** The block rides in the first `session/prompt` user text, after the memory tail and
  after the spill, so it becomes part of the cached prefix. Turn 2+ sends the delta and nothing else
  (`test_bridge.ControlPlaneInjectionTests`).
- **No duplicate tools.** The bridge never exposes read/write/shell/search; `hermes_memory` and
  `hermes_skill` are the whole surface, and the tool count is capped at 5 (`MAX_TOOLS`).
- **No create verb, by design.** Neither scope-recall nor Hermes offers "add one memory": new
  memories come from turn capture + consolidation. An earlier version exposed `action=save` writing
  `bridge_notes.jsonl`; it was removed because that file is not a claim (no recall ranking, no
  consolidation, no dedupe, invisible to other entries) and because the tool answer implied a write
  that had not happened. The model is told to state the fact in its answer instead — the capture
  pipeline records it. Legacy `bridge_notes.jsonl` files are still merged into `search`, never written.
- **Shared stores work.** The bridge asks the store by `--root` first (`AGY_ACP_SR_ROOT`, a
  `scope-recall/shared-root` pointer, or `~/.scope-recall/shared`), and falls back to
  `--host hermes --instance-root $HERMES_HOME`, switching locators once if the store answers
  `shared_store_entry`.
- **Pure passthrough** (`AGY_ACP_INJECT_HERMES_LAYER=0`): `mcpServers: []`, no block, byte-for-byte
  the same request as before the overlay existed.

## What was verified against the real CLI (2026-11-23, agy 1.5.15, Google AI Pro account)

A probe that sends **no** prompt (so it costs no quota) confirmed:

| Fact | Evidence |
|---|---|
| `initialize` → `protocolVersion: 1`, `agentCapabilities.loadSession: true`, `sessionCapabilities: {load, close, resume, list}`, `_meta.antigravity.systemPrompt.append: true` | probe output, recorded in this repo's history |
| `session/new {cwd, mcpServers}` → `{sessionId, models.availableModels, modes, configOptions[model, thought_level, mode], _meta.antigravity.tier}` | 23 model ids, e.g. `gemini-4-neon-high`, `claude-sonnet-5-5-high`, `gpt-oss-120b-medium` |
| `session/load {sessionId, cwd, mcpServers}` restores a session **after the process was killed** | created a session in process A, `kill -9`, loaded it in process B → same `sessionId`, same configOptions |
| unknown id → `-32002 "Session not found: …"` (i.e. the method exists) | probe |
| Model selection is `session/set_config_option {configId: "model"|"thought_level", value}` | probe + `/opt/acp-ant/acp_ant.py` |
| The child needs egress: `session/new` hangs forever on this host without `HTTPS_PROXY=http://127.0.0.1:7890` | measured: 120 s timeout direct, ~2 s through the proxy |

`session/prompt` and `session/request_permission` were exercised against the bundled mock only —
see "Testing" — because a real prompt spends subscription quota.

## Configuration

Everything is an environment variable (no core config keys). Set them where the gateway runs
(`~/.hermes/.env` for this hub).

| Variable | Default | Meaning |
|---|---|---|
| `AGY_ACP_BIN` | `agy` (else `AGY_BIN`) | binary; use the real executable, not a `.cmd` shim |
| `AGY_ACP_ARGS` | `acp --disable-telemetry` | argv tail (shlex) |
| `AGY_ACP_HOME` | *(inherited)* | `HOME` for the child — i.e. which `agy` login it uses |
| `AGY_ACP_PROXY` | *(none)* | exported as `HTTPS_PROXY`/`HTTP_PROXY` (+ lowercase) for the child |
| `AGY_ACP_MAX_SESSIONS` | `4` | LRU cap on live children |
| `AGY_ACP_IDLE_SECONDS` | `14400` | idle timeout before a child is closed (mapping kept) |
| `AGY_ACP_TURN_TIMEOUT` | `900` | per-turn wall clock |
| `AGY_ACP_PERMISSION_TIMEOUT` | `90` | ask→deny deadline |
| `AGY_ACP_AUTO_ALLOW_READONLY` | `1` | read/grep/ls/glob/search auto-allow |
| `AGY_ACP_MAX_PROMPT_BYTES` | `114688` (112 KiB) | spill threshold in **bytes** — `agy` truncates the prompt by bytes at the tail, silently |
| `AGY_ACP_MAX_PROMPT_CHARS` | `160000` | the older character budget; either limit spills the turn | above this the turn is spilled to a file |
| `AGY_ACP_TOOL_LOOP_LIMIT` / `_SECONDS` / `_ACTION` | `15` / `180` / `cancel` | tool-start count and elapsed time outside ACP permission requests. The full prompt timeout still includes approval waits |
| `AGY_ACP_ENFORCE_LINEAR` / `AGY_ACP_ON_NONLINEAR` | `1` / `error` | regenerate / history-edit policy (`error` or `resync`) |
| `AGY_ACP_ON_LOAD_FAILURE` | `error` | `error` (refuse) or `reset` (fresh session) |
| `AGY_ACP_RETRY_429` / `_DELAY` | `1` / `5` | exactly one retry for a rate-limited turn |
| `AGY_ACP_ALLOW_BACKGROUND` | `0` | override the interactive-only guard |
| `AGY_ACP_ALLOW_AUX` | `0` | allow compression/titling/vision traffic (discouraged) |
| `AGY_ACP_MODEL` | *(session default)* | default model when Hermes passes none |
| `AGY_ACP_SYSTEM_PROMPT` | *(none)* | frozen prefix, sent once in `session/new` |
| `AGY_ACP_STATE_DB` / `AGY_ACP_LOG_DIR` | `$HERMES_HOME/agy_acp/sessions.db` / `$HERMES_HOME/logs/agy_acp` | state and logs |
| `AGY_ACP_MAX_CONCURRENT` | `4` | turn slots on the one account (1 = strict serialization) |
| `AGY_ACP_CAPABILITY_MODE` | `supervised` | `readonly` (ask before execution or writes) / `supervised` (native shell guard, file writes ask) / `full` (nothing asks) |
| `AGY_ACP_APPROVAL_WAIT` | *(= permission_timeout)* | extra patience for a *write* prompt on the file channel, capped by `turn_timeout` |
| `AGY_ACP_MEMORY_TOKENS` / `_ITEM_CHARS` | `2000` / `1200` | recalled-memory budget per turn, and the excerpt length per item |
| `AGY_ACP_ON_INTERRUPTED` | `note` | tell the model its previous turn died with the provider (`ignore` to stay silent) |
| `AGY_ACP_BRIDGE_NAME` | `hermes-bridge` | MCP server name; tool names become `mcp_<name>_*` |
| `AGY_ACP_LEGACY_NOTES` | `0` | merge `bridge_notes.jsonl` (only from versions that had `action=save`) |
| `AGY_ACP_DISPLAY_HEAD` / `_TAIL` | `2048` / `2048` | display-layer fold size |
| `AGY_ACP_INJECT_HERMES_LAYER` | `1` | mount `hermes-bridge` + inject the turn-1 block; `0` = pure agy |
| `AGY_ACP_BRIDGE_SKILLS` | *(none)* | comma-separated skill dirs to surface as extra tools (cap 5 total) |
| `AGY_ACP_BRIDGE_PAGE_BYTES` | `4096` | hard cap per bridge result |
| `AGY_ACP_CONTROL_PLANE_TOKENS` | `500` | budget for the turn-1 block |
| `AGY_ACP_BRIDGE_CMD` / `_ARGS` | this python + `bridge_mcp.py` | bridge launch override |
| `AGY_ACP_SR_CLI` / `AGY_ACP_SR_ROOT` | auto | scope-recall CLI path / shared-store root (resolved host-side and handed to the bridge) |
| `AGY_ACP_BRIDGE_APPROVAL` | `all` | `all` = the bridge never prompts (matches Hermes, whose `revise`/`forget` tools are ungated); `reads` = memory writes ask; `none` = every bridge action asks. `AGY_ACP_AUTO_ALLOW_BRIDGE=0` still means `none` |
| `AGY_ACP_APPROVAL_BACKEND` | `auto` | `auto` = Hermes' own approval gate when importable, else the file inbox; `hermes` = gate only; `file` = inbox only |
| `AGY_ACP_BRIDGE_TRACE` | *(none)* | path of a raw MCP stdio trace (frames in and out) for handshake debugging |
| `AGY_ACP_NDJSON_TRACE` | `0` | write per-frame traces (`in`/`out`, truncated) |
| `AGY_ACP_TRACE_DIR` | `$HERMES_HOME/.hermes_traces` | where those traces land |

Profile: `agy-acp` (aliases `agy`, `antigravity`, `antigravity-acp`), `auth_type=external_process`,
`base_url=acp://agy`, `default_aux_model=""`, `supports_streaming=False`.

## Acceptance criteria (spec §9) → tests

| Spec item | Test |
|---|---|
| 前缀缓存与增量：2–5 轮只发增量，记忆在用户消息里 | `test_turns.DeltaTests::test_only_the_newest_user_turn_crosses_the_pipe`, `::test_system_prompt_is_sent_once_at_session_new` |
| 记忆去重：同一记忆只注入一次 | `test_memory_display.MemoryDedupeTests::test_same_memory_is_injected_once_per_native_session`, `::test_a_repeat_with_nothing_new_sends_no_block_at_all` |
| Windows 进程树清理 | `test_protocol.ProcessTreeTests` (POSIX group assertion here; `taskkill /F /T` + Job Object path on Windows) |
| 权限超时防死锁：90 s → deny → 释放锁 | `test_permissions.PermissionTimeoutTests::*` (0.7 s timeout in the test), `PermissionAnswerTests::test_waiting_for_a_permission_releases_the_account_gate` |
| 即时中断：`session/cancel` 500 ms 内 | `test_turns.CancelTests::test_cancel_reaches_the_agent_in_milliseconds` |
| 大文本与工具输出防护 | `test_memory_display.SpillTests`, `DisplayTests`, `ToolLoopTests` |
| 并发串行化 (Semaphore(1)) | `test_turns.SerializationTests::test_turns_never_overlap_account_wide` |
| 429 退避（≤1 次重试） | `test_turns.RateLimitTests` |
| 进程崩溃降级 + `session/load` | `test_turns.CrashTests`, `test_pool.PoolTests::test_restart_loads_the_native_session_instead_of_replaying` |
| Hermes 接线（profile/凭据/客户端） | `test_hermes_seam.HermesSeamTests` (runs in the real Hermes interpreter) |

## Testing

```bash
cd ~/.hermes/plugins && python3 -m unittest discover -s agy_acp/tests -t .
```

All of it runs against `mock_agy_acp.py` — a stdio server speaking the same dialect — so the loop
is free and cannot 429. 57 tests, ~25 s. Scenarios: `default`, `permission`, `permission_read`,
`ratelimit`, `crash`, `loop`, `big_output`, `silent_new`, `stderr_flood`, `load_fail`.

Two real-CLI facts this mock cannot prove — that the handshake succeeds on the real binary and that
the account's model list matches `FALLBACK_MODELS` — are covered by `python3 -m agy_acp.probe`
(step 1 above). One real `session/prompt` **has** been spent (2026-11-24, 22-character input,
`end_turn`, 6.56 s, `inputTokens 8714 / outputTokens 21`) — see `evidence/real-turn.md`; a second
turn, which is the only way to observe prefix-cache reuse, has not.

## Switching to the real `agy acp` (3 steps)

```bash
cd ~/.hermes/plugins

# 1) FREE probe — handshake, capabilities, model list, session/load support. Sends no prompt.
AGY_ACP_BIN=/opt/acp-ant/bin/agy \
AGY_ACP_HOME=/opt/acp-ant/accounts/g1 \
AGY_ACP_PROXY=http://127.0.0.1:7890 \
python3 -m agy_acp.probe        # expect "ok": true, 23 model ids, loadSession true, tier "Google AI Pro"

# 2) env + route the chat at the stateful provider, then restart the gateway (it reads env at start)
cat >> ~/.hermes/.env <<'EOF'
AGY_ACP_BIN=/opt/acp-ant/bin/agy
AGY_ACP_HOME=/opt/acp-ant/accounts/g1
AGY_ACP_PROXY=http://127.0.0.1:7890
AGY_ACP_MODEL=gemini-4-neon-high
EOF
hermes gateway restart
hermes config set model.provider agy-acp
hermes config set model.default gemini-4-neon-high

# 3) ONE interactive turn in a chat, then verify it stayed stateful
python3 -m agy_acp.cli status    # one row: turns 1, injected_memory 0; no second child after turn 2
pgrep -af 'agy acp'              # exactly one process, and its cwd is the chat's workspace
```

Without the proxy this host hangs in `session/new` (measured: 120 s, then timeout) because the
Antigravity endpoint is unreachable directly — that is the single most likely first failure.

Rollback: `hermes config set model.provider opencode-go` (restore `model.default`), then
`python3 -c "from agy_acp.client import get_manager; get_manager().close_all()"` to reap children.

Operator commands (`~/.hermes/plugins`, any of them work while the gateway runs):
`status`, `sessions`, `permissions`, `answer <id> allow|deny`, `cancel <session>`.

## Quick check that `hermes-bridge` really mounted (real `agy`, no prompt, no quota)

```bash
cd ~/.hermes/plugins
# 1) the mount is in the request (free: session/new only) and agy spawns the bridge itself
AGY_ACP_BIN=/opt/acp-ant/bin/agy AGY_ACP_HOME=/opt/acp-ant/accounts/g1 \
AGY_ACP_PROXY=http://127.0.0.1:7890 HERMES_HOME=/tmp/agy-mount-check \
python3 - <<'PY'
import os, sys, subprocess, time
sys.path.insert(0, os.getcwd())
from agy_acp.manager import AgyAcpSessionManager
from agy_acp.settings import Settings
s = Settings.from_env(); s.ensure_dirs()
m = AgyAcpSessionManager(s)
p = m.session_for("mount-check", cwd="/tmp", model="gemini-4-neon-low")
time.sleep(2)
print(subprocess.run(["pgrep", "-af", "bridge_mcp.py"], capture_output=True, text=True).stdout or "NOT RUNNING")
m.close_all()
PY
# expect: .../python3 .../agy_acp/bridge_mcp.py   ← agy started it
grep -h "MCP servers of the session" /opt/acp-ant/accounts/g1/.gemini/antigravity-cli/log/acp-server-*.log | tail -1
# expect: Preloading 1 MCP servers of the session

# 2) the bridge answers on its own (no agy involved)
HERMES_HOME=$HOME/.hermes python3 - <<'PY'
import sys; sys.path.insert(0, "/root/.hermes/plugins")
from agy_acp.tests.mcp_client import McpClient
c = McpClient(env={"HERMES_HOME": "/root/.hermes"}); c.initialize()
print(len(c.tools()), "tools:", [t["name"] for t in c.tools()])
print(c.call("hermes_memory", {"action": "search", "query": "Tailscale", "page_size": 2})[0][:200])
c.close()
PY
```

## Deviations from the spec — and why

1. **Synchronous transport, not `asyncio`.** Hermes' provider seam is `client.chat.completions.create(...)`
   called from the agent loop's worker thread; the in-tree `agent/acp_client.py` is synchronous for
   exactly that reason. The stateful transport therefore uses one reader thread + a request table,
   with the same properties the spec asks of asyncio: event-driven reads, zero polling while idle,
   millisecond cancel. A stray `asyncio` loop in a background thread would only add a hop.
   The stop button is wired end to end: Hermes' interrupt loop calls `client.cancel()` on any
   non-HTTP client (`agent/client_lifecycle.py::_abort_request_slot_client`), which for this provider
   means "write `session/cancel` for the last turn" — no process teardown, no waiting.
2. **No token streaming.** `supports_streaming=False`, like every ACP provider in-tree; the turn is
   one blocking `session/prompt` and the answer arrives when the agent finishes its tool loop. `stream=True`
   callers still get the two-chunk shape Hermes expects. Real streaming needs the response object to be
   consumed incrementally by the core — an untested path for a stateful client, so it is not shipped.
3. **Permission UI.** Hermes core has no plugin-facing permission modal, so the broker writes
   `<session>.json` into `$HERMES_HOME/agy_acp/permissions/` (answerable by the CLI, a UI plugin, or
   `PermissionBroker(handler=…)`) and answers **deny** at the deadline. Read-only tools auto-allow.
4. **"Disable edit/regenerate" cannot remove UI buttons from a plugin.** It is enforced where it
   matters: a re-sent turn (regenerate, or an edit that rewinds) is refused before anything is sent,
   with a message that says what to do instead — or `AGY_ACP_ON_NONLINEAR=resync` forks a fresh native
   session. Nothing is ever replayed silently.
5. **`interactive_only` is a policy guard, not a capability boundary.** Cron (`HERMES_CRON_SESSION`),
   delegating subagents and gateway background sources are refused; auxiliary traffic (compression,
   titling, vision) is refused by inspecting the calling module, because the core does not pass an
   `aux_task` flag to providers. Both have explicit escape hatches.
6. **Tool activity in the chat.** The agent owns its tools, so Hermes never sees tool call/result
   objects — it sees text. Activity goes to `logs/agy_acp/events-<session>.jsonl` (folded to
   head+tail with `...[已省略 N 字节，完整输出保留在 agy 上下文中]...`) so nothing thousands of
   tokens long reaches the chat DB or a UI. A future UI plugin can tail that file.
7. **Windows tree cleanup is best-effort by design.** `taskkill /F /T` is the primary path; a Job
   Object with `KILL_ON_JOB_CLOSE` is attached when `ctypes` allows it. Both are bounded, and the
   POSIX path (process group SIGTERM → SIGKILL) is the one covered by tests here.

## Known costs / limits

- **How memory actually gets written.** Hermes' scope-recall plugin registers global turn hooks
  (`turn_start` / `pre_llm_call` / `post_llm_call` / post-tool) and records each turn through
  `sync_turn_source_events`; that path is independent of the model provider, so `agy_acp` neither
  blocks nor replaces it. What the model can do *on its own* is `revise` (new value or `retract`) and
  `forget` — scope-recall has **no create verb anywhere**, including its own tool surface
  (`recall/inspect/profile/entity/revise/forget/status`): new memories come from the capture +
  consolidation pipeline, not from a tool call — so the bridge exposes no create verb either:
  `search`, `show`, `revise`, `forget`. Legacy `bridge_notes.jsonl` files are read, not written.
- **Traces live under `$HERMES_HOME/.hermes_traces/`**, not inside the session workspace: a debugging
  artefact should never appear in a user's repository.
- **The bridge's own actions ride Hermes' rules: no prompt.** scope-recall's `revise`/`forget` tools are
  not gated by Hermes (they call the store core directly), so gating the bridge's memory actions would
  be stricter than the host. Ask for prompts only with `AGY_ACP_BRIDGE_APPROVAL=reads|none`.
- In supervised mode, ACP `run_command` requests containing `CommandLine` use
  `tools.approval.check_all_command_guards()` with the active Hermes turn context. Ordinary
  commands proceed without a card. Flagged commands follow Hermes' approval mode and remembered
  grants, and the native guard's denial ends that request. `agy` still runs the command itself.
- Other escalations, including file writes, use `tools.approval.request_tool_approval()`. Its
  remembered answers use `plugin_rule:agy:…` keys. A command sent through this fallback uses a
  key specific to its arguments. When the tool gate is unavailable or returns `approval_required`,
  the file inbox can be answered with `python3 -m agy_acp.cli answer <id> allow|deny`.
- **agy's MCP client probes before it connects.** It sends `server/discover`
  (protocol `2026-07-28` in `_meta`) *before* `initialize`, and an error there drops the whole server
  with `MCP server connection closed unexpectedly … invalid request`. `Preloading 1 MCP servers of the
  session` in agy's log therefore does **not** prove the bridge mounted: check for the absence of
  `invalid request` and, for real proof, watch a turn call `mcp_hermes-bridge_*`. `bridge_mcp.py`
  answers the probe, the optional `resources`/`prompts`/`logging` surfaces, and never replies to
  notifications.
- **Byte budgets, not character budgets.** `agy` truncates an oversized `session/prompt` at ~128 KiB
  of UTF-8, silently and at the tail — which is where the question is. One CJK character is three
  bytes, so the spill threshold is measured in bytes (112 KiB default) and the inline head that
  survives a spill is byte-capped too.
- **Bridge tools opt out of lazy loading.** agy moves larger tool sets to on-demand unless a tool
  declares `_meta: {"antigravity/alwaysLoad": true}`; a lazily loaded tool never appears in the
  model's function list.
- **Tool output never reaches the provider.** agy's `tool_call` updates carry `title/status/input` and
  nothing else, so there is no 10–20 KB tool payload to fold on our side; the fold in `display.py`
  exists for the fields we do render (and for a future agy that sends content).
- **No third-party MCP passthrough.** `session/new` carries exactly one server (`hermes-bridge`) or
  none; a whitelist for other servers is still an additive change.

- One turn at a time **per process**. A second chat's turn waits for the gate (and is released
  during a permission wait); that is deliberate — parallel prompts are the fastest route to a 429.
- A turn that `agy` answers with no text (denied permission, refusal) produces an explicit
  `(agy returned no text; stop_reason=…)` line instead of an empty assistant message.
- The model list is static (23 ids from the probe). Enumerating live would spawn an `agy` process on
  every model-picker build; use `session/new` configOptions if you need the truth right now.
- `plugins validate` reports two advisories: four guarded private imports (each with a fallback) and
  "in-process only" isolation — the same trade-off every `create_client` provider makes.
