"""Hermes provider plugin: native stateful ``agy acp`` (Antigravity CLI), Route A.

Import-time contract (``providers/__init__.py`` discovery): this module must call
``register_provider(...)``. It also exposes ``register(ctx)`` so the directory can be loaded by the
general plugin manager without a second copy of the implementation.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from typing import Any

try:  # inside Hermes
    from providers import register_provider
    from providers.base import ProviderProfile

    _HERMES_RUNTIME = True
except ModuleNotFoundError:  # standalone (the plugin's own tests, a reviewer's checkout)
    register_provider = None  # type: ignore[assignment]
    _HERMES_RUNTIME = False

    class ProviderProfile:  # type: ignore[no-redef]
        """Minimal stand-in so the package imports without a Hermes installation."""

        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

        def listed(self) -> bool:
            return True

        def discover_models(self, **kwargs: Any) -> list[dict[str, Any]] | None:
            return None

        def create_client(self, **client_kwargs: Any) -> Any | None:
            return None

from .client import AgyAcpClient, get_manager

__all__ = ["AgyAcpProfile", "AgyAcpClient", "get_manager", "register"]

#: What a real ``session/new`` advertised on 2026-11-23 (agy 1.5.15, Google AI Pro account).
#: A static list keeps the model picker free: enumerating live would spawn an ``agy`` process.
FALLBACK_MODELS = (
    "gemini-4-neon-high",
    "gemini-4-neon-xhigh",
    "gemini-4-neon-medium",
    "gemini-4-neon-low",
    "gemini-4-hydrogen-high",
    "gemini-4-hydrogen-xhigh",
    "gemini-4-hydrogen-medium",
    "gemini-4-hydrogen-low",
    "gemini-4-argon-high",
    "gemini-4-argon-medium",
    "gemini-4-argon-low",
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "claude-sonnet-5-5-high",
    "claude-sonnet-5-5-medium",
    "claude-sonnet-5-5-low",
    "claude-opus-5-5-high",
    "claude-opus-5-5-medium",
    "claude-opus-5-5-low",
    "gpt-oss-120b-medium",
    "gemini-pro-orchestrator-neon",
    "gemini-pro-orchestrator-g1-pro",
    # base ids (no thought-level suffix) that a live session advertised on 2026-11-24;
    # picking one applies the model and leaves the agent's own level default alone.
    "gemini-4-carbon",
    "gemini-pro-orchestrator",
    "gemini-pro-orchestrator-argon",
)

MODEL_ALIASES = {
    "neon": "gemini-4-neon-high",
    "hydrogen": "gemini-4-hydrogen-high",
    "argon": "gemini-4-argon-high",
    "flash": "gemini-3.8-flash-high",
    "sonnet": "claude-sonnet-5-5-high",
    "opus": "claude-opus-5-5-high",
}


class AgyAcpProfile(ProviderProfile):
    """``auth_type="external_process"`` profile whose client is the stateful ACP session manager."""

    def create_client(self, **client_kwargs: Any) -> Any:
        allowed = {"api_key", "base_url", "default_headers", "command", "args", "timeout", "acp_cwd"}
        return AgyAcpClient(**{k: v for k, v in client_kwargs.items() if k in allowed})

    def discover_models(self, **kwargs: Any) -> list[dict[str, Any]] | None:
        """Deliberately static: a live catalog means spawning ``agy`` on every picker build."""
        return None


def _resolve_binary() -> str:
    """The absolute `agy` this install actually uses.

    Hermes' external-process resolver reads ``process_command`` before any of our settings exist and
    looks it up on the *gateway's* PATH, which is sanitised — a bare "agy" then reads as "not
    installed". Resolving here (env → PATH → known locations) keeps delegation and the provider in
    agreement about which binary is the agent.
    """
    for env_name in ("AGY_ACP_BIN", "AGY_BIN"):
        value = os.environ.get(env_name, "").strip()
        if value and Path(value).exists():
            return value
    found = shutil.which("agy")
    if found:
        return found
    for candidate in ("/opt/acp-ant/bin/agy", str(Path.home() / ".local/bin/agy")):
        if Path(candidate).exists():
            return candidate
    return "agy"


_PROFILE = AgyAcpProfile(
    name="agy-acp",
    aliases=("agy", "antigravity", "antigravity-acp"),
    display_name="Antigravity (agy acp)",
    description="Antigravity CLI over ACP — native stateful session, agent-owned tools, no history replay",
    auth_type="external_process",
    base_url="acp://agy",
    env_vars=("AGY_ACP_BIN", "AGY_ACP_ARGS", "AGY_ACP_HOME", "AGY_ACP_PROXY"),
    process_command=_resolve_binary(),
    process_args=("acp", "--disable-telemetry"),
    process_command_env_vars=("AGY_ACP_BIN", "AGY_BIN"),
    process_args_env_var="AGY_ACP_ARGS",
    supports_streaming=False,  # the ACP turn is one blocking request; chunks arrive in one shot
    supports_health_check=False,
    supports_model_listing=False,
    supports_vision=True,
    fallback_models=FALLBACK_MODELS,
    model_aliases=MODEL_ALIASES,
    default_aux_model="",  # auxiliary work must never ride a consumer subscription
)

if register_provider is not None:  # inside Hermes: the import-time registration contract
    register_provider(_PROFILE)


def register(ctx: Any) -> None:  # pragma: no cover - only when loaded as a general plugin
    """No hooks: the provider owns its transport, its memory tail and its display handling."""
    return None
