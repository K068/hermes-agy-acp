<div align="center">

# hermes-agy-acp

**A native, stateful Antigravity provider for Hermes — one ACP session per chat, deltas only, no history replay.**

[![tests](https://img.shields.io/badge/tests-115%20passing-brightgreen)](#testing)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![license](https://img.shields.io/badge/license-MIT-black)](LICENSE)
[![agy](https://img.shields.io/badge/agy-1.5.15-purple)](#compatibility)

[English](#english) · [中文](#中文)

</div>

---

## English

### What this is

A Hermes Agent **provider plugin**. It drives the official Antigravity CLI (`agy acp`) as a
long-lived, bidirectional JSON-RPC child process, so the model keeps its own native conversation
state. Hermes sends **only the newest user turn**; history, tool state and the prompt cache live
inside `agy`.

It replaces the older HTTP-gateway approach (an OpenAI-compatible shim in front of a per-request
`agy` process that replayed the whole conversation on every call).

### Why it exists

| | HTTP gateway (before) | this provider |
|---|---|---|
| Conversation state | none — full replay per request | native, inside `agy` |
| Prompt prefix reuse | none | **77 % cache hit** measured on a warm turn |
| Tokens per follow-up | whole history | the delta (~2.5 k vs ~10.5 k measured) |
| Tools | re-declared every call; schema bloat | `agy`'s own tools, unchanged |
| Latency over a long chat | grows with history | flat |
| Hermes-side history edits | silently forked the model's state | refused before sending (`NonlinearTurnError`) |

### Architecture

```
Hermes agent loop
  └── client.chat.completions.create(model, messages)        OpenAI-compatible seam
       └── AgyAcpClient                 client.py   delta slicing, guards, usage mapping
            ├── AgyAcpSessionManager    manager.py  child pool (idle 4 h / LRU 4), N turn slots
            │    ├── AgyConnection      protocol.py NDJSON duplex, request table, stderr pump
            │    │     └── agy acp                  one child per chat, own process group
            │    ├── PermissionBroker   permissions.py  capability modes, Hermes approval gate
            │    ├── SessionStore       store.py       hermes_session ⇄ acp_session (SQLite)
            │    └── EventLog           display.py     folded activity, loop breaker
            ├── memory / prompt         memory.py, prompt.py   recall tail, 160 KB spill
            └── control plane           control_plane.py + bridge_mcp.py + mcp_mount.py
```

### Features

**Session model**

- **Delta-only turns.** `session/prompt` carries the newest user message; nothing else is re-sent.
- **Crash-safe resume.** A restarted Hermes re-attaches with `session/load` (verified across `kill -9`).
- **In-flight marker.** A turn cut off by a restart is reported once to the model, then cleared.
- **Non-linear edits refused.** Re-send / regenerate / history edit raises before a single token is
  spent; `AGY_ACP_ON_NONLINEAR=resync` opens a fresh native session instead.
- **Pooling + teardown.** Idle/LRU eviction, POSIX process groups, Windows `taskkill /F /T` + Job Object.
  The Windows path is tested on a real machine (child **and** grandchild die).

**Safety**

- **Capability modes** — `readonly` / `supervised` / `full`, decided once per session.
- **Hermes' own approval gate.** ACP permission requests are routed through
  `tools.approval.request_tool_approval`, so the prompt is the one you already answer, `always`
  answers persist, and a headless context falls back to a file inbox instead of guessing.
- **Human-in-the-loop by default.** Cron / subagents / aux calls are refused unless you name the
  origins you trust (`AGY_ACP_ALLOW_BACKGROUND_ORIGINS`).
- **Bounded everything.** Turn timeout, permission timeout, tool-loop breaker, `session/cancel`, and a
  Stop button that works even while the session is still spinning up.

**Control plane (Hermes overlay, opt-out)**

`agy` keeps its own file, search and terminal tools. Hermes adds only what `agy` cannot see, through
one small stdio MCP server plus a single Turn-1 block:

| | |
|---|---|
| `hermes_memory` | `search` · `show` · `revise` · `forget` — paged, 4 KB per page |
| `hermes_skill` | `list` · `view`, plus one tool per configured skill (≤ 5 tools total) |
| `<hermes_control_plane>` | ≤ 500-token persona excerpt, injected on turn 1 only, so the cache stays intact |

There is deliberately **no create verb**: new memories come from Hermes' turn-capture pipeline, so the
model is told to state the fact rather than pretend to write it.

### Requirements

- Hermes Agent with the plugin provider contract (`requires_hermes >= 0.22.5`)
- The official `agy` CLI, signed in (`agy` in `PATH`, or `AGY_ACP_BIN=/path/to/agy`)
- Python 3.10+ for the plugin itself (stdlib only — no pip dependencies)

### Install

```bash
git clone https://github.com/K068/hermes-agy-acp.git ~/hermes-agy-acp
ln -s ~/hermes-agy-acp/agy_acp ~/.hermes/plugins/agy_acp     # Hermes loads the package directory
hermes plugins validate ~/.hermes/plugins/agy_acp
hermes model        # pick "Antigravity (agy acp)"
```

Optional but recommended in `~/.hermes/config.yaml` (environment variables win when both are set):

```yaml
agy_acp:
  command: /opt/acp-ant/bin/agy          # absolute path keeps delegation's resolver happy
  child_home: /opt/acp-ant/accounts/g1   # the HOME that holds the agy sign-in
  proxy: http://127.0.0.1:7890           # the agent needs its own egress if Google is blocked
  max_concurrent: 4
  capability_mode: supervised
```

### Configuration

Every key has an `AGY_ACP_*` environment variable and an `agy_acp:` entry in `config.yaml`
(environment wins). The ones that matter day to day:

| Key | Default | What it does |
|---|---|---|
| `command` / `AGY_ACP_BIN` | `agy` | the binary to run |
| `child_home` / `AGY_ACP_HOME` | *(empty)* | `HOME` for the child — this is where the sign-in lives |
| `proxy` / `AGY_ACP_PROXY` | *(empty)* | `HTTPS_PROXY` for the child |
| `model` | *(account default)* | e.g. `gemini-4-neon-xhigh`, `gemini-4-hydrogen-high` |
| `max_concurrent` | `4` | turn slots on the one account; `1` restores strict serialization |
| `capability_mode` | `supervised` | `readonly` / `supervised` / `full` |
| `allow_background_origins` | *(empty)* | which automations may drive the account (`subagent`, `cron`, …) |
| `inject_hermes_layer` | `true` | `false` = pure `agy`, no MCP, no Turn-1 block |
| `max_prompt_chars` | `160000` | above this the turn is spilled to a file and referenced by path |
| `memory_tokens` | `2000` | recalled-memory budget per turn |
| `bridge_approval` | `all` | `all` / `reads` / `none` — when the bridge may prompt |
| `enable_ndjson_trace` | `false` | per-frame stdio trace under `.hermes_traces/` |

### Usage

Once selected as the provider it behaves like any other model. Useful extras:

```bash
python -m agy_acp.cli status          # sessions, slots, capability mode, in-flight turns
python -m agy_acp.cli cancel <chat>   # stop a running turn
python -m agy_acp.cli permissions     # pending approval requests (file channel)
python -m agy_acp.cli answer <id> allow
```

### Verified on a real machine

Measured against `agy 1.5.15` with a Google AI Pro account, not simulated:

| What | Number |
|---|---|
| Prefix cache on a warm second turn | **8 458 cached / 2 528 new** input tokens → 77 % |
| Same follow-up without a cache (cold first turn) | 10 515 input tokens |
| Native context survived the delta | the model answered a fact from turn 1 with no history sent |
| Delegated child (`delegate_task` on this provider) | `HARDOK` in 22.4 s, `api_calls=1` |
| Windows process-tree teardown | child + grandchild stopped, 3.19 s, `is_alive() == false` |
| Crash recovery | `kill -9` on the child, a new process resumed the same `session/load` |
| A real turn captured into scope-recall | 4 `source_events`, searchable, tagged with the entry |

### Compatibility

Measured against `agy 1.5.15` (ACP over stdio):

- `initialize` advertises `loadSession`, `sessionCapabilities{load,close,resume,list}` and `configOptions{model,thought_level,mode}`.
- `session/new` / `session/load` accept `mcpServers`, and the agent really spawns them on both paths.
- The agent's MCP client probes `server/discover` **before** `initialize` and drops the whole server
  if that errors — the bridge answers the probe for exactly this reason.
- `usage` uses camelCase, including `cachedReadTokens` (the field a naive `cacheReadTokens` lookup misses).
- Tool updates carry `title/status/input` only; tool **output** never crosses ACP.

### Testing

```bash
python -m unittest discover -s tests -t .      # 115 tests, ~30 s, no network, no quota
```

`mock_agy_acp.py` speaks the same JSON-RPC as the real CLI and covers ten scenarios, including
`permission`, `permission_read`, `ratelimit`, `crash`, `loop`, `big_output`, `silent_new`,
`stderr_flood` and `load_fail`. Nothing in the suite touches the real account.

### Design notes

- **No core edits.** The plugin lives entirely under `plugins/agy_acp/`.
- **Reasons over retries.** When the agent's native session cannot be restored, the provider raises a
  message naming the two ways forward rather than silently replaying history.
- **Fail loudly at the boundary.** Protocol errors are classified (`rate_limit`, `auth`, `transport`),
  so Hermes' fallback chain can act on them.
- **Known limits.** One consumer account, so concurrency is a slot count, not a queue; the agent's own
  prompt is ~8.7 k tokens before your text; a model switch inside a live session resets the cache.

---

## 中文

### 这是什么

一个 Hermes Agent 的**模型 Provider 插件**。它把官方 Antigravity CLI（`agy acp`）作为长期存活的双向
JSON-RPC 子进程来驱动，对话状态由模型自己持有。Hermes **只发最新一轮用户输入**，历史、工具状态和
前缀缓存全部留在 `agy` 内部。

它替代的是旧的 HTTP 网关方案（OpenAI 兼容 shim + 每次请求新开 `agy` 进程、把整段对话重放一遍）。

### 为什么要做

| | 旧 HTTP 网关 | 本 Provider |
|---|---|---|
| 对话状态 | 没有，每次全量重放 | 原生保存在 `agy` 里 |
| 前缀缓存复用 | 无 | 实测温轮**命中 77%** |
| 后续轮次 token | 整段历史 | 仅增量（实测 ~2.5k vs ~10.5k） |
| 工具 | 每次重复声明，schema 膨胀 | 完全用 `agy` 自带的 |
| 长对话延迟 | 随历史增长 | 基本恒定 |
| Hermes 侧改历史 | 静默让模型状态分叉 | 发送前直接拒绝（`NonlinearTurnError`） |

### 架构

```
Hermes 主循环
  └── client.chat.completions.create(model, messages)        与内置 provider 同一接缝
       └── AgyAcpClient                 client.py   增量切片、守卫、用量映射
            ├── AgyAcpSessionManager    manager.py  子进程池（空闲 4h / LRU 4）、N 个轮次槽位
            │    ├── AgyConnection      protocol.py NDJSON 双向、请求表、stderr 泵
            │    │     └── agy acp                  一 chat 一进程，独立进程组
            │    ├── PermissionBroker   permissions.py  能力档、Hermes 原生审批闸门
            │    ├── SessionStore       store.py      hermes 会话 ⇄ acp 会话（SQLite）
            │    └── EventLog           display.py    折叠活动日志、工具循环熔断
            ├── memory / prompt         memory.py, prompt.py   记忆尾注、160KB 落盘
            └── 控制面                   control_plane.py + bridge_mcp.py + mcp_mount.py
```

### 功能

**会话模型**

- **只有增量**：`session/prompt` 只带最新用户消息，其余一概不重发。
- **崩溃可恢复**：Hermes 重启后用 `session/load` 接回原生上下文（经 `kill -9` 实测）。
- **在途标记**：被重启打断的那一轮，会向模型说明一次"上一轮被中断且未保存"，随后自动清除。
- **拒绝非线性编辑**：重发 / 回退 / 改历史在发出前就报错，零请求；`AGY_ACP_ON_NONLINEAR=resync` 可改开新会话。
- **池化与收尸**：空闲/LRU 淘汰；POSIX 用进程组，Windows 用 `taskkill /F /T` + Job Object，
  **Windows 分支在真机上验证过**（子进程与孙进程都被清掉）。

**安全**

- **能力档**：`readonly` / `supervised` / `full`，每会话读一次，轮内不可提权。
- **走 Hermes 自己的审批**：ACP 的权限请求转发给 `tools.approval.request_tool_approval`，
  也就是你平时点的那张卡；选"总是允许"会被记住；无人在场时降级成文件收件箱，而不是瞎猜。
- **默认只给人类用**：cron / 子 agent / 辅助调用一律拒绝，除非你显式点名信任的来源
  （`AGY_ACP_ALLOW_BACKGROUND_ORIGINS`）。
- **处处有界**：轮次超时、权限超时、工具循环熔断、`session/cancel`，以及"会话还在握手时按停止也有效"。

**控制面（可一键关掉的 Hermes 叠加层）**

`agy` 原本的文件、搜索、终端能力一律不动，Hermes 只补它看不见的东西：一个极小的 stdio MCP
服务 + Turn-1 一次性注入。

| | |
|---|---|
| `hermes_memory` | `search` · `show` · `revise` · `forget`，分页，单页 4KB 上限 |
| `hermes_skill` | `list` · `view`，外加每个配置技能一个工具（总数 ≤ 5） |
| `<hermes_control_plane>` | ≤ 500 token 的人设摘要，仅第 1 轮注入，保住前缀缓存 |

**故意没有"新建记忆"这个动词**：新记忆由 Hermes 的回合捕获管道产生，所以工具会直接告诉模型
"把事实说出来就行"，而不是假装写入了什么。

### 环境要求

- Hermes Agent（`requires_hermes >= 0.22.5`，需要插件 provider 契约）
- 已登录的官方 `agy` CLI（在 `PATH` 里，或用 `AGY_ACP_BIN=/path/to/agy` 指定）
- Python 3.10+（插件只用标准库，无 pip 依赖）

### 安装

```bash
git clone https://github.com/K068/hermes-agy-acp.git ~/hermes-agy-acp
ln -s ~/hermes-agy-acp/agy_acp ~/.hermes/plugins/agy_acp     # Hermes 加载的是这个包目录
hermes plugins validate ~/.hermes/plugins/agy_acp
hermes model        # 选 "Antigravity (agy acp)"
```

建议在 `~/.hermes/config.yaml` 里补上（环境变量优先级更高）：

```yaml
agy_acp:
  command: /opt/acp-ant/bin/agy          # 绝对路径，delegation 的解析器才找得到
  child_home: /opt/acp-ant/accounts/g1   # 存放 agy 登录态的 HOME
  proxy: http://127.0.0.1:7890           # Google 被墙时，给子进程单独出口
  max_concurrent: 4
  capability_mode: supervised
```

### 配置

每个键都有对应的 `AGY_ACP_*` 环境变量和 `agy_acp:` 配置项（同时出现时环境变量优先）。
日常最常用的几个：

| 键 | 默认 | 作用 |
|---|---|---|
| `command` / `AGY_ACP_BIN` | `agy` | 要启动的二进制 |
| `child_home` / `AGY_ACP_HOME` | 空 | 子进程的 `HOME`，登录态就在这 |
| `proxy` / `AGY_ACP_PROXY` | 空 | 子进程的 `HTTPS_PROXY` |
| `model` | 账号默认 | 如 `gemini-4-neon-xhigh`、`gemini-4-hydrogen-high` |
| `max_concurrent` | `4` | 单账号的并行轮次槽位；设 `1` 回到严格串行 |
| `capability_mode` | `supervised` | `readonly` / `supervised` / `full` |
| `allow_background_origins` | 空 | 允许哪些自动化驱动账号（`subagent`、`cron`…） |
| `inject_hermes_layer` | `true` | 设 `false` = 纯净 agy，无 MCP、无 Turn-1 注入 |
| `max_prompt_chars` | `160000` | 超过则落盘并以路径引用 |
| `memory_tokens` | `2000` | 每轮记忆注入预算 |
| `bridge_approval` | `all` | `all` / `reads` / `none`，桥什么时候要人批 |
| `enable_ndjson_trace` | `false` | 帧级 stdio 追踪，写到 `.hermes_traces/` |

### 使用

选成 provider 之后与普通模型无异。常用辅助命令：

```bash
python -m agy_acp.cli status          # 会话、槽位、能力档、在途轮次
python -m agy_acp.cli cancel <chat>   # 停掉正在跑的那一轮
python -m agy_acp.cli permissions     # 文件通道里待批的请求
python -m agy_acp.cli answer <id> allow
```

### 真机实测

以下数字来自 `agy 1.5.15` + Google AI Pro 账号，不是模拟：

| 项目 | 数字 |
|---|---|
| 温轮前缀缓存 | **命中 8 458 / 新增 2 528** 输入 token → 77% |
| 无缓存的冷首轮 | 10 515 输入 token |
| 原生上下文确实保留 | 没发历史，模型答出了第 1 轮里的事实 |
| 子 agent（`delegate_task` 走本 provider） | 22.4 秒回 `HARDOK`，`api_calls=1` |
| Windows 进程树清理 | 子进程 + 孙进程都被停掉，3.19 秒，`is_alive() == false` |
| 崩溃恢复 | 子进程 `kill -9` 后，新进程用 `session/load` 接回同一会话 |
| 真实一轮被写进 scope-recall | 4 条 `source_events`，可检索，带 entry 归属 |

### 兼容性

针对 `agy 1.5.15`（stdio 上的 ACP）实测：

- `initialize` 声明 `loadSession`、`sessionCapabilities{load,close,resume,list}`、`configOptions{model,thought_level,mode}`；
- `session/new` 与 `session/load` 都接受 `mcpServers`，且两条路径上 agy 都真的会拉起 MCP 子进程；
- agy 的 MCP 客户端会在 `initialize` **之前**先探 `server/discover`，回错就丢弃整个 server——
  桥之所以要实现这个探针，就是因为踩过；
- `usage` 用 camelCase，缓存字段是 `cachedReadTokens`（按 `cacheReadTokens` 找会永远读到 0）；
- 工具更新只带 `title/status/input`，工具**输出**从不跨界。

### 测试

```bash
python -m unittest discover -s tests -t .      # 115 个用例，约 30 秒，不联网、不耗配额
```

`mock_agy_acp.py` 用与真 CLI 相同的 JSON-RPC 说话，覆盖十个场景：`permission`、`permission_read`、
`ratelimit`、`crash`、`loop`、`big_output`、`silent_new`、`stderr_flood`、`load_fail` 等。
整个测试套件不碰真实账号。

### 设计取舍

- **零核心改动**：全部收在 `plugins/agy_acp/` 内。
- **宁可报错也不重放**：原生会话恢复不了时，抛出说明两条出路的错误，而不是偷偷重发历史。
- **边界处大声失败**：协议错误分类（`rate_limit` / `auth` / `transport`），好让 Hermes 的 fallback 链接手。
- **已知限制**：单消费账号，所以并发是"槽位"而不是排队；模型的系统提示本身约 8.7k token；
  会话中途换模型会让前缀缓存失效。

---

<div align="center">

MIT © K068 · built for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) · not affiliated with Google

</div>
