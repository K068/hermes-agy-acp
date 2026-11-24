"""Configuration for the native stateful ``agy acp`` provider.

Every knob has an ``AGY_ACP_*`` environment variable (env wins) and an optional ``agy_acp:``
section in ``$HERMES_HOME/config.yaml``, so ``hermes config set agy_acp.<key> …`` works too. Values
are read once per client/session-manager construction (cheap) — never on the streaming hot path.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path


def hermes_home() -> Path:
    """The active Hermes home, resolved without importing Hermes (tests run standalone)."""
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env)
    try:  # inside Hermes this is the authoritative answer (profile-aware)
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(str(get_hermes_home()))
    except Exception:
        return Path.home() / ".hermes"


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _scalar(text: str) -> object:
    value = text.strip().strip('"').strip("'")
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    return value


def _parse_value(text: str) -> object:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(part) for part in inner.split(",")] if inner else []
    if text.startswith("{") and text.endswith("}"):
        return {}  # nested mappings are not used by this plugin; ignored rather than mis-read
    return _scalar(text)


def _raw_section(text: str) -> dict[str, object]:
    """Read the ``agy_acp:`` block with no YAML library at all.

    The gateway's interpreter ships without PyYAML, and Hermes' own loader drops sections it does not
    know — which silently emptied every setting in the process that actually runs the model. This
    reader understands exactly what the plugin's own config needs: ``key: value``, ``key: [a, b]`` and
    ``key:`` followed by ``- item`` lines.
    """
    out: dict[str, object] = {}
    inside = False
    list_key = ""
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw[:1].isspace():
            key = raw.split(":", 1)[0].strip() if ":" in raw else ""
            inside = key == "agy_acp"
            list_key = ""
            continue
        if not inside:
            continue
        line = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if line.startswith("- ") and list_key:
            bucket = out.get(list_key)
            if isinstance(bucket, list):
                bucket.append(_scalar(line[2:]))
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value:
            out[key] = _parse_value(value)
            list_key = ""
        elif indent <= 2:
            out[key] = []
            list_key = key
        else:
            list_key = ""
    return out


def config_section() -> dict[str, object]:
    """The ``agy_acp:`` block of the profile's config.yaml (empty when absent or unreadable)."""
    path = hermes_home() / "config.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        section = data.get("agy_acp") if isinstance(data, dict) else None
        if isinstance(section, dict):
            return section
    except Exception:
        pass
    section = _raw_section(text)   # no PyYAML, or YAML the loader refused: read it ourselves
    if section:
        return section
    try:  # last resort inside Hermes: its migrated view (may omit unknown sections)
        from hermes_cli.config import load_config_readonly  # type: ignore

        data = load_config_readonly()
        section = data.get("agy_acp") if isinstance(data, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _pick(name: str, key: str, default: object, *, section: dict[str, object] | None = None) -> object:
    """env var, then the config.yaml key, then the default."""
    raw = os.environ.get(name)
    if raw is not None and raw.strip() != "":
        return raw
    if section is None:
        section = config_section()
    return section.get(key, default)


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text not in ("0", "false", "no", "off")


def _as_int(value: object, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_list(value: object) -> tuple[str, ...]:
    """A list from YAML, or a "a,b" string from the environment."""
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip().lower() for v in value if str(v).strip())
    return tuple(part.strip().lower() for part in str(value or "").split(",") if part.strip())


def _as_str(value: object, default: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


#: The ACP agent's own answer when the user hits stop. Sent as a JSON-RPC notification.
CANCEL_METHOD = "session/cancel"


@dataclass
class Settings:
    """Resolved provider settings (see README for the table)."""

    # ── process launch ────────────────────────────────────────────────
    command: str = ""
    args: tuple[str, ...] = ("acp", "--disable-telemetry")
    child_home: str = ""
    proxy: str = ""
    # ── storage / logs ────────────────────────────────────────────────
    home: Path = field(default_factory=hermes_home)
    state_db: Path | None = None
    log_dir: Path | None = None
    # ── pool ──────────────────────────────────────────────────────────
    max_sessions: int = 4
    #: How many turns may run at once against the one account. 1 restores strict serialization.
    max_concurrent_turns: int = 4
    idle_seconds: float = 14400.0
    connect_timeout: float = 120.0
    turn_timeout: float = 900.0
    # ── turn behaviour ────────────────────────────────────────────────
    model: str = ""
    system_prompt: str = ""
    max_prompt_chars: int = 160_000
    permission_timeout: float = 90.0
    #: Extra patience for a *write* prompt on the file channel (default: same as permission_timeout).
    #: Always bounded by ``turn_timeout``: a turn cannot outlive its own budget.
    approval_wait: float | None = None
    #: readonly = agy may look but not change (its writes are refused outright);
    #: supervised = its shell/file writes ask (default); full = nothing asks.
    #: The bridge itself never asks unless ``bridge_approval`` says so.
    capability_mode: str = "supervised"
    auto_allow_readonly: bool = True
    auto_allow_bridge: bool = True
    #: all = the bridge never prompts (matches Hermes, whose memory tools are ungated);
    #: reads = only memory writes ask; none = every bridge action asks.
    bridge_approval: str = "all"
    #: auto = Hermes' own approval gate when reachable, else this plugin's file inbox;
    #: hermes = only the gate; file = only the inbox (no core import at all).
    approval_backend: str = "auto"
    tool_loop_limit: int = 15
    tool_loop_seconds: float = 180.0
    tool_loop_action: str = "cancel"  # cancel | warn
    interactive_only: bool = True
    allow_background: bool = False
    #: Automation origins that may drive the account even when ``allow_background`` is off:
    #: any of cron / kanban / webhook / background / heartbeat / subagent. Empty = none.
    allow_background_origins: tuple[str, ...] = ()
    allow_auxiliary: bool = False
    enforce_linear: bool = True
    nonlinear_action: str = "error"  # error | resync
    on_load_failure: str = "error"  # error | reset
    retry_429: bool = True
    retry_429_delay: float = 5.0
    streaming: bool = False
    display_head: int = 2048
    display_tail: int = 2048
    event_log_max_bytes: int = 5 * 1024 * 1024
    # ── control plane (Hermes overlay) ────────────────────────────────
    inject_hermes_layer: bool = True
    bridge_name: str = "hermes-bridge"
    #: How much recalled memory one turn may carry, and the longest single excerpt.
    memory_tokens: int = 2000
    memory_item_chars: int = 1200
    #: note = tell the model its previous turn was cut off by a restart; ignore = stay silent.
    on_interrupted: str = "note"
    #: Merge ``bridge_notes.jsonl`` written by versions that had ``action=save``.
    legacy_notes: bool = False
    bridge_tools: tuple[str, ...] = ()          # skill names the bridge may expose (0..2)
    bridge_page_bytes: int = 4096               # hard cap per MCP tool result
    control_plane_max_tokens: int = 500
    bridge_command: str = ""                    # default: this interpreter + bridge_mcp.py
    bridge_args: tuple[str, ...] = ()
    bridge_timeout: float = 20.0
    # ── tracing ───────────────────────────────────────────────────────
    enable_ndjson_trace: bool = False
    trace_dir: Path | None = None

    def __post_init__(self) -> None:
        home = Path(self.home)
        self.home = home
        if self.state_db is None:
            self.state_db = home / "agy_acp" / "sessions.db"
        if self.log_dir is None:
            self.log_dir = home / "logs" / "agy_acp"
        if self.trace_dir is None:
            self.trace_dir = home / ".hermes_traces"
            # The spec's path is relative to the session's workspace; ``.hermes_traces`` under the
            # Hermes home keeps traces out of a user's repo while staying one predictable place.
        self.args = tuple(self.args)
        self.bridge_tools = tuple(str(name) for name in self.bridge_tools if str(name).strip())
        self.bridge_args = tuple(self.bridge_args)

    # ── derived paths ─────────────────────────────────────────────────
    @property
    def bridge_prefix(self) -> str:
        """How agy names the bridge's tools on the wire (``mcp_<server>_<tool>``)."""
        return f"mcp_{self.bridge_name}_"

    @property
    def work_dir(self) -> Path:
        """Where per-turn spill files and mirrors live (never the system temp dir)."""
        return self.home / "agy_acp" / "tmp"

    @property
    def cancel_dir(self) -> Path:
        return self.home / "agy_acp" / "cancel"

    @property
    def permission_dir(self) -> Path:
        return self.home / "agy_acp" / "permissions"

    def ensure_dirs(self) -> None:
        for path in (self.state_db.parent if self.state_db else None, self.log_dir, self.work_dir,
                     self.cancel_dir, self.permission_dir, self.trace_dir):
            if path is not None:
                Path(path).mkdir(parents=True, exist_ok=True)

    # ── construction ──────────────────────────────────────────────────
    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        sec = config_section()

        def pick(env: str, key: str, default: object) -> object:
            return _pick(env, key, default, section=sec)

        args_raw = _as_str(pick("AGY_ACP_ARGS", "args", ""))
        command = _env_str("AGY_ACP_BIN") or _as_str(pick("AGY_ACP_BIN", "command", "")) \
            or _env_str("AGY_BIN") or "agy"
        bridge_tools_raw = pick("AGY_ACP_BRIDGE_SKILLS", "bridge_tools", ())
        if isinstance(bridge_tools_raw, str):
            bridge_tools = tuple(t.strip() for t in bridge_tools_raw.split(",") if t.strip())
        else:
            bridge_tools = tuple(str(t).strip() for t in (bridge_tools_raw or ()) if str(t).strip())
        bridge_args_raw = pick("AGY_ACP_BRIDGE_ARGS", "bridge_args", "")
        if isinstance(bridge_args_raw, (list, tuple)):
            bridge_args = tuple(str(a) for a in bridge_args_raw)
        else:
            bridge_args = tuple(shlex.split(str(bridge_args_raw or "")))
        values: dict[str, object] = dict(
            command=command,
            args=tuple(shlex.split(args_raw)) if args_raw else ("acp", "--disable-telemetry"),
            child_home=_as_str(pick("AGY_ACP_HOME", "child_home", "")),
            proxy=_as_str(pick("AGY_ACP_PROXY", "proxy", "")),
            max_sessions=max(1, _as_int(pick("AGY_ACP_MAX_SESSIONS", "max_sessions", 4), 4)),
            max_concurrent_turns=max(1, _as_int(pick("AGY_ACP_MAX_CONCURRENT", "max_concurrent", 4), 2)),
            idle_seconds=_as_float(pick("AGY_ACP_IDLE_SECONDS", "idle_seconds", 14400.0), 14400.0),
            connect_timeout=_as_float(pick("AGY_ACP_CONNECT_TIMEOUT", "connect_timeout", 120.0), 120.0),
            turn_timeout=_as_float(pick("AGY_ACP_TURN_TIMEOUT", "turn_timeout", 900.0), 900.0),
            model=_as_str(pick("AGY_ACP_MODEL", "model", "")),
            system_prompt=_as_str(pick("AGY_ACP_SYSTEM_PROMPT", "system_prompt", "")),
            max_prompt_chars=_as_int(pick("AGY_ACP_MAX_PROMPT_CHARS", "max_prompt_chars", 160_000), 160_000),
            permission_timeout=_as_float(pick("AGY_ACP_PERMISSION_TIMEOUT", "permission_timeout", 90.0), 90.0),
            approval_wait=(_as_float(raw_wait, 0.0) or None) if (raw_wait := _as_str(
                pick("AGY_ACP_APPROVAL_WAIT", "approval_wait", ""))) else None,
            capability_mode=_as_str(pick("AGY_ACP_CAPABILITY_MODE", "capability_mode", "supervised"),
                                    "supervised").lower(),
            auto_allow_readonly=_as_bool(pick("AGY_ACP_AUTO_ALLOW_READONLY", "auto_allow_readonly", True), True),
            auto_allow_bridge=_as_bool(pick("AGY_ACP_AUTO_ALLOW_BRIDGE", "auto_allow_bridge", True), True),
            bridge_approval=_as_str(pick("AGY_ACP_BRIDGE_APPROVAL", "bridge_approval", "all"), "all").lower(),
            approval_backend=_as_str(pick("AGY_ACP_APPROVAL_BACKEND", "approval_backend", "auto"), "auto").lower(),
            tool_loop_limit=_as_int(pick("AGY_ACP_TOOL_LOOP_LIMIT", "tool_loop_limit", 15), 15),
            tool_loop_seconds=_as_float(pick("AGY_ACP_TOOL_LOOP_SECONDS", "tool_loop_seconds", 180.0), 180.0),
            tool_loop_action=_as_str(pick("AGY_ACP_TOOL_LOOP_ACTION", "tool_loop_action", "cancel"), "cancel"),
            allow_background=_as_bool(pick("AGY_ACP_ALLOW_BACKGROUND", "allow_background", False), False),
            allow_background_origins=_as_list(pick("AGY_ACP_ALLOW_BACKGROUND_ORIGINS",
                                                   "allow_background_origins", ())),
            allow_auxiliary=_as_bool(pick("AGY_ACP_ALLOW_AUX", "allow_auxiliary", False), False),
            enforce_linear=_as_bool(pick("AGY_ACP_ENFORCE_LINEAR", "enforce_linear", True), True),
            nonlinear_action=_as_str(pick("AGY_ACP_ON_NONLINEAR", "nonlinear_action", "error"), "error"),
            on_load_failure=_as_str(pick("AGY_ACP_ON_LOAD_FAILURE", "on_load_failure", "error"), "error"),
            on_interrupted=_as_str(pick("AGY_ACP_ON_INTERRUPTED", "on_interrupted", "note"), "note"),
            retry_429=_as_bool(pick("AGY_ACP_RETRY_429", "retry_429", True), True),
            retry_429_delay=_as_float(pick("AGY_ACP_RETRY_429_DELAY", "retry_429_delay", 5.0), 5.0),
            streaming=_as_bool(pick("AGY_ACP_STREAMING", "streaming", False), False),
            display_head=_as_int(pick("AGY_ACP_DISPLAY_HEAD", "display_head", 2048), 2048),
            display_tail=_as_int(pick("AGY_ACP_DISPLAY_TAIL", "display_tail", 2048), 2048),
            inject_hermes_layer=_as_bool(pick("AGY_ACP_INJECT_HERMES_LAYER", "inject_hermes_layer", True), True),
            enable_ndjson_trace=_as_bool(pick("AGY_ACP_NDJSON_TRACE", "enable_ndjson_trace", False), False),
            bridge_name=_as_str(pick("AGY_ACP_BRIDGE_NAME", "bridge_name", "hermes-bridge"), "hermes-bridge"),
            bridge_tools=bridge_tools,
            bridge_page_bytes=_as_int(pick("AGY_ACP_BRIDGE_PAGE_BYTES", "bridge_page_bytes", 4096), 4096),
            control_plane_max_tokens=_as_int(pick("AGY_ACP_CONTROL_PLANE_TOKENS", "control_plane_max_tokens", 500), 500),
            memory_tokens=_as_int(pick("AGY_ACP_MEMORY_TOKENS", "memory_tokens", 2000), 2000),
            memory_item_chars=_as_int(pick("AGY_ACP_MEMORY_ITEM_CHARS", "memory_item_chars", 1200), 1200),
            legacy_notes=_as_bool(pick("AGY_ACP_LEGACY_NOTES", "legacy_notes", False), False),
            bridge_command=_as_str(pick("AGY_ACP_BRIDGE_CMD", "bridge_command", "")),
            bridge_args=bridge_args,
            bridge_timeout=_as_float(pick("AGY_ACP_BRIDGE_TIMEOUT", "bridge_timeout", 20.0), 20.0),
        )
        state_db = _as_str(pick("AGY_ACP_STATE_DB", "state_db", ""))
        log_dir = _as_str(pick("AGY_ACP_LOG_DIR", "log_dir", ""))
        if state_db:
            values["state_db"] = Path(state_db)
        if log_dir:
            values["log_dir"] = Path(log_dir)
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]
