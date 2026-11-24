# Specification: Hermes Native Stateful `agy_acp` Provider (v3.0 – Route A)

> Transcribed from the shared Gemini document `https://share.gemini.google/7qKukKPnuCV6`
> («Hermes Native Agy Provider Evaluation», 2026-11-23) for offline review. Wording is the
> document's; only the numbering/lists were normalised. §8 was elided in the shared copy itself
> ("YAML …") and is reproduced here as-is.

## 1. 目标、架构约束与合规边界

### 1.1 核心架构目标
- **彻底弃用** `ACP-Ant` HTTP 网关与无状态全量历史重放模式。
- 在 Hermes 中实现原生有状态 Provider/Plugin，直接通过双向 `stdin`/`stdout` 管道管理官方 `agy acp` 子进程。
- **增量托管原则**：Hermes 侧不再向 `agy` 发送全量历史 `messages`，每轮对话仅发送**最新一轮的增量输入**，历史上下文、工具执行状态与 Prompt Cache 完全托管给 `agy` 进程本地维护。

### 1.2 合规与使用边界（防封号红线）
- **严格单账号模式**：严禁多账号池与自动化账号轮询（Account Rotation）。仅使用本机官方 `agy` 登录的单一消费级账号，严禁提取、转发或代理 OAuth Token。
- **仅限人工交互驱动（Human-in-the-Loop）**：该 Provider 在 Hermes 中必须硬编码标记为 `interactive_only: true`。**严禁**被 Hermes 的后台定时任务（Cron）、自主循环 Agent 或其他子 Agent 自动编排调用，确保所有 `session/prompt` 均由真实人类用户的直接输入触发。

## 2. 防算力浪费架构设计（借鉴 T3 Code & Pi）

1. **前缀缓存绝对不可变（T3 Code 模式）**：严禁在每轮对话中动态修改 System Prompt（例如注入实时时间戳、动态变化的记忆列表或实时系统状态）。任何位于历史前缀的 1 字节变动都会导致整个 200K+ 的 Prefix Cache 失效，引发算力暴增与 `429` 风控。
2. **极简工具声明与按需加载（Pi 模式）**：`session/new` 初始化时**不注入**冗余的 Hermes 内置工具与无关 MCP Server。仅保留 `agy` 原生工具与极少数白名单工具，将每轮固定开销控制在最低水位。
3. **即时中断传播（`session/cancel`）**：当用户在 Hermes 前端点击"停止生成"或关闭会话窗口时，Provider 必须**毫秒级**向 `agy` 的 `stdin` 写入标准 `session/cancel` JSON-RPC 通知，强制终止底层推理与工具循环，防止后台空转烧毁配额。
4. **事件驱动零轮询**：严禁使用定时轮询检查子进程状态，完全基于 `asyncio` 异步流读取 `stdout` NDJSON，空闲期间 CPU 占用趋近于 0。

## 3. 进程与会话生命周期管理 (`AgyAcpSessionManager`)

### 3.1 双维度进程保活池
- 维护字典 `sessions: dict[hermes_session_id, AgyProcessHandle]`，采用**时间 + 容量**双重淘汰策略：
  - **空闲超时**：单个会话空闲超过 `1800s`（30 分钟）后优雅退出释放内存。
  - **LRU 上限**：同时保活的子进程上限设为 `3`~`5` 个（可配置）；超出上限时按 LRU 淘汰最久未活跃的进程（淘汰前确保 `acp_session_id` 已落盘）。
- **跨平台进程卫生（重点适配 Windows）**：
  - 拉起子进程时优先解析 `agy` 真实可执行文件路径（绕过 `.cmd` 包装层），强制 UTF-8 编码（`encoding="utf-8", bufsize=1`）。
  - 必须为每个子进程的 `stderr` 启动独立的异步读取协程并重定向至日志文件，防止管道缓冲区满导致子进程静默死锁。
  - 进程销毁时必须清理整个进程树（Windows 使用 Job Object 或 `taskkill /F /T`，Unix 使用进程组 `SIGTERM`/`SIGKILL`），杜绝僵尸进程。

### 3.2 JSON-RPC 握手、持久化与降级策略
- **能力协商**：首次拉起子进程后发送 `initialize` 请求，校验服务端返回的协议版本与 `capabilities`（确认支持 `session/load`）。
- **会话创建与映射**：发送 `session/new`（传入 `cwd` 与目标模型），将返回的原生 `acp_session_id` 持久化至本地 SQLite 映射表：

```sql
CREATE TABLE IF NOT EXISTS agy_acp_sessions (
    hermes_session_id TEXT PRIMARY KEY,
    acp_session_id TEXT NOT NULL,
    cwd TEXT NOT NULL,
    model TEXT NOT NULL,
    injected_memory_ids TEXT DEFAULT '[]', -- JSON 数组，供 scope-recall 去重
    last_active_at INTEGER NOT NULL
);
```

- **半开检测与显式降级**：
  - 复用保活进程前先检查 `process.returncode is None` 及 `stdin` 可写状态。
  - 若进程已退出且本地存在 `acp_session_id`，重启后优先调用 `session/load` 恢复原生状态。
  - **严禁静默全量重放**：若 `session/load` 失败（如 `agy` 升级导致缓存不兼容），必须向前端抛出明确错误，由用户手动确认是否"压缩历史摘要并开启全新会话"。

## 4. `scope-recall` 记忆插件完整适配方案

在有状态 `agy_acp` 架构下，传统的"每轮改写 System Prompt 注入记忆"会彻底摧毁 Prefix Cache。必须对 `scope-recall` 进行以下 4 项深度适配：

### 4.1 禁用 System Prompt 注入，改用增量尾注（Turn-Level Suffix）
- **拦截默认 Hook**：当当前会话路由至 `agy_acp` 时，禁用 `scope-recall` 修改 System Prompt 的前置钩子。
- **增量消息挂载**：仅在构造当轮 `session/prompt` 的用户增量文本时，将召回的记忆以轻量 XML 标签附加在用户输入之前（作为当轮新增 Token，**零破坏历史前缀缓存**）：

```xml
<recalled_memory scope="project:hermes">
  - [mem_8a2f] 用户本地为 Windows + MacBook 双端环境，通过 Tailscale 同步。
</recalled_memory>
{用户本轮实际输入的文本}
```

### 4.2 会话级记忆去重（防重复注入浪费 Token）
- 在 `AgyProcessHandle` 与 SQLite 映射表中维护集合 `injected_memory_ids: set[str]`。
- 每轮 `scope-recall` 检索出候选记忆后，**自动过滤掉已存在于 `injected_memory_ids` 中的记忆 ID**（因为它们已经在 `agy` 的原生上下文中了，无需二次发送）。
- 只有真正新增的记忆才会被注入本轮增量，并立刻写入 `injected_memory_ids` 持久化。

### 4.3 召回预算硬上限
- 强制配置 `scope-recall` 针对 `agy_acp` 的检索参数：`top_k <= 5`、`similarity_threshold >= 0.72`、`max_injected_tokens <= 1500`（超出则按相关性截断）。

### 4.4 记忆写回（Write-Back）清洗
- `agy acp` 内部会执行大量工具调用，产生海量中间输出。
- `scope-recall` 的异步写回钩子**严禁**监听 `session/update` 中的原始工具输出流，仅在整轮 `session/prompt` 正常结束（收到最终 Result）后，提取干净的 `(user_delta, final_assistant_text)` 对送入记忆提取管道。

## 5. 工具调用深度规范（防 Bug 与防 Token 暴增）

### 5.1 工具所有权隔离（杜绝双重工具 Schema 膨胀）
- **原生工具完全下放**：文件读写、终端执行、代码搜索等本地能力**完全交给 `agy` 内置工具执行**。Hermes 侧**严禁**向 `agy` 透传 Hermes 自带的同类本地工具（如 `read_file`、`bash`）。
- **MCP 白名单透传**：仅在 `session/new` 时透传经过严格筛选的、与 `agy` 能力互补的 MCP Server，且单个会话透传的外部 MCP 工具总数建议不超过 `5` 个。

### 5.2 工具输出"展示层截断 vs 模型层保真"分离
- **模型层（`agy` 内部）**：禁用 Hermes 侧对 `agy` 上下文的工具输出强制截断插件，保护 `agy` 原生上下文完整与缓存命中。
- **展示与存储层（Hermes 侧）**：对写入 Hermes 前端 UI 和本地聊天记录 DB 的内容执行强制折叠截断（例如仅保留首 2KB + 尾 2KB，中间替换为 `...[已省略 N 字节，完整输出保留在 agy 上下文中]...`）。

### 5.3 权限确认（`session/request_permission`）防死锁状态机
- 严格校验 JSON-RPC 请求的 `id`，确保回复的 `result` 与 `id` 精确匹配。
- **强制超时兜底**：收到 `session/request_permission` 后启动 `90s` 倒计时。若用户未在 90s 内操作，Provider 自动向 `agy` 回复 `deny`（附带原因 `"User permission timed out"`），并立即释放全局锁。
- **细粒度自动放行（可选）**：仅允许对明确只读类工具（如 `read_file`、`grep`、`ls`）配置自动放行；所有写文件、删文件、执行 Shell 命令必须转发前端人工确认。

### 5.4 工具死循环熔断（防 Token 暴增）
- 为每轮 `session/prompt` 维护计数器 `tool_call_count`；单轮连续工具调用超过阈值（默认 `15` 次）或单轮持续超过 `180s` 时，前端弹出"检测到高频工具循环，点击立即中断"提示条。

## 6. 并发控制与防 429 风控机制

### 6.1 账号级全局并发锁
- 引入账号级全局锁 `asyncio.Semaphore(1)`，确保同一时刻全系统仅有一个会话处于向 `agy` 发送 `session/prompt` 并等待模型响应的活跃状态。
- **锁释放优化**：当活跃会话进入 `session/request_permission` 等待人工确认阶段时，临时释放推理锁（或由超时机制快速回收），避免单个挂起的确认弹窗饿死其他会话。

### 6.2 429 优雅退避与非线性操作拦截
- **禁止高频重试**：捕获到 `429` 或速率限制错误时，严禁立即重试；采用指数退避（`5s -> 15s`，最多自动重试 1 次）或直接在 UI 展示冷却倒计时。
- **拦截非线性操作**：针对该 Provider **禁用**"编辑历史消息"和"重新生成（Regenerate）"按钮；若需要分叉对话，必须显式点击"新建分支会话（触发 `session/new`）"。

## 7. 增量消息切片与流式事件映射

### 7.1 增量输入提取与大文本落盘保护
- 仅提取当前轮次新增的用户文本（附加 4.1 节的 `scope-recall` 增量记忆）及附带的文件路径引用。
- **160K 大文本保护**：单次增量消息超过 `160,000` 字符时，自动写入会话专属临时文件（如 `.hermes_tmp/{session_id}/{turn_id}.txt`）并替换为路径引用提示，规避管道缓冲区截断；临时文件在会话关闭或进程退出时统一清理。

### 7.2 `stdout` NDJSON 流式映射

| `agy acp` `session/update` 事件类型 | Hermes 内部映射目标 | 特殊处理要求 |
|---|---|---|
| 文本增量块 (`text_delta`) | `StreamChunk(type="text")` | 实时推流，无缓冲延迟 |
| 思考过程块 (`reasoning_delta`) | `StreamChunk(type="reasoning")` | 前端默认折叠展示 |
| 工具调用开始 (`tool_call_start`) | `ToolCard(status="running")` | 递增 `tool_call_count` 计数 |
| 工具执行结果 (`tool_call_result`) | `ToolCard(status="done")` | 应用 5.2 节展示层 4KB 截断 |
| 权限请求 (`session/request_permission`) | `PermissionModal` | 启动 90s 超时防死锁计时器 |

## 8. 配套 Hermes 参数调优与配置模板

*（Elided in the shared document: "YAML …" — no template was published. The provider's own
environment variable table in `README.md` covers the same ground.)*

## 9. 验收测试清单 (Acceptance Criteria)

- [ ] **前缀缓存与增量验证**：抓包或开启 `agy` 调试日志，确认第 2~5 轮对话仅发送当轮增量文本，且 `scope-recall` 注入的记忆位于用户消息内、未触发 System Prompt 变更。
- [ ] **记忆去重验证**：连续两轮提及同一项目关键词，确认 `scope-recall` 仅在第 1 轮注入对应 `memory_id`，第 2 轮成功跳过。
- [ ] **Windows 进程树清理**：在 Windows 下触发空闲超时或 LRU 淘汰，确认任务管理器中无残留 `agy` 或 `cmd.exe` 僵尸进程。
- [ ] **权限超时防死锁**：触发需要权限确认的 Shell 命令且不点击任何按钮，确认 90s 后自动回复 `deny`、释放全局锁，且其他会话可正常发起新请求。
- [ ] **即时中断验证**：在模型长输出或工具循环期间点击前端"停止"，确认 `session/cancel` 在 500ms 内发出且底层进程停止吐出 Token。
- [ ] **大文本与工具输出防护**：粘贴 200,000 字符文本，确认自动落盘并转为路径引用；执行输出 1MB 日志的命令，确认 Hermes 前端流畅不卡顿且仅展示截断摘要。
