"""Phase 4: the real Hermes seams — profile registration, credential resolution, client creation.

These run in the Hermes interpreter (``hermes_cli`` importable), in a subprocess with a throwaway
``HERMES_HOME``, so the repository's own config is never touched.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERMES_ROOT = Path("/root/.hermes/hermes-agent")
PLUGINS_ROOT = Path("/root/.hermes/plugins")

CHILD = textwrap.dedent(
    """
    import json, os, sys
    out = {}
    from providers import get_provider_profile, list_providers
    profile = get_provider_profile("agy-acp")
    out["name"] = profile.name
    out["auth_type"] = profile.auth_type
    out["base_url"] = profile.base_url
    out["aliases"] = list(profile.aliases)
    out["models"] = list(profile.fallback_models)[:3]
    out["env_vars"] = list(profile.env_vars)
    out["supports_streaming"] = profile.supports_streaming
    out["default_aux_model"] = profile.default_aux_model
    out["listed"] = profile.listed()
    out["discover_models"] = profile.discover_models()

    from hermes_cli.auth import resolve_external_process_provider_credentials
    creds = resolve_external_process_provider_credentials("agy-acp")
    out["creds_command"] = creds["command"]
    out["creds_args"] = creds["args"]
    out["creds_base_url"] = creds["base_url"]

    client = profile.create_client(**{k: v for k, v in creds.items() if k in
                                      ("api_key", "base_url", "command", "args")})
    out["client_class"] = type(client).__name__
    completion = client.chat.completions.create(
        model="gemini-4-neon", messages=[{"role": "user", "content": "seam check"}])
    out["content"] = completion.choices[0].message.content
    out["finish"] = completion.choices[0].finish_reason
    client.close()
    print(json.dumps(out))
    """
)


def _hermes_python() -> str | None:
    override = os.environ.get("HERMES_PYTHON", "").strip()
    if override and os.path.exists(override):
        return override
    for candidate in sorted(glob.glob("/root/.hermes/installs/*/environments/*/venv/bin/python")):
        probe = subprocess.run([candidate, "-c", "import hermes_cli"], capture_output=True, text=True,
                               timeout=120, cwd="/tmp")
        if probe.returncode == 0:
            return candidate
    return None


class HermesSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.python = _hermes_python()
        if not cls.python:
            raise unittest.SkipTest("no Hermes interpreter found (set HERMES_PYTHON)")

    def test_profile_registers_and_client_speaks_to_the_agent(self) -> None:
        import shutil

        source = PLUGINS_ROOT / "agy_acp"
        with tempfile.TemporaryDirectory(prefix="agy-seam-") as tmp:
            installed = Path(tmp) / "plugins" / "agy_acp"
            shutil.copytree(source, installed, ignore=shutil.ignore_patterns("__pycache__", "tests"))
            mock = installed / "mock_agy_acp.py"
            env = dict(os.environ)
            env.update({
                "HERMES_HOME": tmp,
                "PYTHONPATH": f"{HERMES_ROOT}:{tmp}/plugins",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HERMES_SESSION_ID": "seam-session",
                "AGY_ACP_BIN": sys.executable,
                "AGY_ACP_ARGS": str(mock),
                "AGY_ACP_HOME": "",
                "AGY_ACP_PROXY": "",
                "MOCK_AGY_RECORD": os.path.join(tmp, "records.jsonl"),
                "MOCK_AGY_SESSIONS": os.path.join(tmp, "sessions.json"),
                "MOCK_AGY_SCENARIO": "default",
            })
            proc = subprocess.run([self.python, "-c", CHILD], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                self.fail(f"child failed:\nstdout={proc.stdout}\nstderr={proc.stderr[-4000:]}")
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(result["name"], "agy-acp")
        self.assertEqual(result["auth_type"], "external_process")
        self.assertEqual(result["base_url"], "acp://agy")
        self.assertIn("agy", result["aliases"])
        self.assertEqual(result["supports_streaming"], False)
        self.assertEqual(result["default_aux_model"], "")
        self.assertTrue(result["listed"])
        self.assertIsNone(result["discover_models"])
        self.assertEqual(result["creds_args"], [str(mock)])
        self.assertEqual(result["creds_base_url"], "acp://agy")
        self.assertEqual(result["client_class"], "AgyAcpClient")
        self.assertEqual(result["content"], "mock reply: 1")
        self.assertEqual(result["finish"], "stop")


if __name__ == "__main__":
    unittest.main()


class BackgroundGuardTests(unittest.TestCase):
    """A delegated child is automation; the account opens to it only through the allowlist."""

    def _client(self, origins):
        import tempfile
        from pathlib import Path as _P

        from agy_acp.client import AgyAcpClient
        from agy_acp.settings import Settings

        tmp = tempfile.mkdtemp(prefix="guard-")
        settings = Settings(home=_P(tmp), allow_background_origins=tuple(origins),
                            command="true", args=())
        return AgyAcpClient(settings=settings, acp_cwd=tmp)

    def test_a_subagent_is_refused_by_default(self) -> None:
        import agy_acp.client as client_mod

        original = client_mod.background_origin
        client_mod.background_origin = lambda: "subagent"
        client = self._client(())
        try:
            with self.assertRaises(RuntimeError) as caught:
                client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
            self.assertIn("interactive-only", str(caught.exception))
        finally:
            client_mod.background_origin = original

    def test_the_allowlist_lets_a_subagent_through_the_guard(self) -> None:
        import agy_acp.client as client_mod

        original = client_mod.background_origin
        client_mod.background_origin = lambda: "subagent"
        client = self._client(("subagent",))
        try:
            # the guard passes; the turn then fails on the (fake) binary, which is the proof we got past it
            with self.assertRaises(Exception) as caught:
                client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"},
                                                                    ], timeout=3)
            self.assertNotIn("interactive-only", str(caught.exception))
        finally:
            client_mod.background_origin = original
