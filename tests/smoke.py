"""Phase-1 smoke: handshake, one turn, clean teardown (run before writing the full suite)."""

from __future__ import annotations

import sys

sys.path.insert(0, "/root/.hermes/plugins")

from agy_acp.tests.support import Harness  # noqa: E402


def main() -> int:
    h = Harness()
    try:
        completion = h.turn("hello mock")
        print("content:", repr(completion.choices[0].message.content))
        print("finish:", completion.choices[0].finish_reason)
        print("usage:", completion.usage)
        print("pool:", h.manager.pool_state())
        print("records:")
        for entry in h.records():
            print(" ", entry.get("event"), {k: v for k, v in entry.items() if k in ("chars", "session", "configId", "value")})
        second = h.turn_messages([
            {"role": "user", "content": "hello mock"},
            {"role": "assistant", "content": completion.choices[0].message.content},
            {"role": "user", "content": "second question"},
        ])
        print("second:", repr(second.choices[0].message.content))
        print("prompt texts:", [p["text"] for p in h.prompts()])
        return 0
    finally:
        h.close()


if __name__ == "__main__":
    sys.exit(main())
