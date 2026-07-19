"""Boot canary (box 3) — prove the gate is live and discriminating.

Deny-only canaries are insufficient: a gate that always emits empty stdout
would pass them while being completely broken. This module fires:

  - 4 deny canaries (shell / write-sensitive / read-sensitive / MCP)
  - 1 allow canary (inert grep — known OPEN)

Each deny must yield empty stdout (no COSMIC-ALLOW). The allow must yield
exactly ``COSMIC-ALLOW v1 <nonce>``. Any failure → refuse cockpit launch.
"""

from __future__ import annotations

import io
import json
import os
import re
import secrets
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Tuple

from cosmic_cli.gate import SENTINEL_VERSION, run_gate

_NONCE_RE = re.compile(r"^[0-9a-f]{32,}$")


def _env(tool: str, tool_input: dict, cwd: str = "/tmp") -> dict:
    return {
        "hookEventName": "pre_tool_use",
        "toolName": tool,
        "toolInput": tool_input,
        "cwd": cwd,
        "workspaceRoot": cwd,
    }


def canary_envelopes(cwd: str = "/tmp") -> Tuple[List[Tuple[str, dict]], List[Tuple[str, dict]]]:
    """Return (deny_cases, allow_cases) as (label, envelope) pairs."""
    deny = [
        ("shell-rm", _env("run_terminal_command", {"command": "rm -rf /tmp/cosmic-canary-x"}, cwd)),
        ("write-env", _env("write", {"file_path": f"{cwd}/.env", "content": "X=1"}, cwd)),
        ("read-env", _env("read_file", {"target_file": f"{cwd}/.env"}, cwd)),
        ("mcp-probe", _env("mcp__probe__tool", {"q": "x"}, cwd)),
    ]
    # Also exercise Claude-alias shell form
    deny.append(
        ("shell-rm-Bash", _env("Bash", {"command": "rm -rf /"}, cwd)),
    )
    allow = [
        ("grep-inert", _env("grep", {"pattern": "cosmic-canary-open"}, cwd)),
    ]
    return deny, allow


def _run_gate_capture(envelope: dict, nonce: str) -> Tuple[str, str, int]:
    """Invoke run_gate; return (stdout, stderr, exit_code)."""
    out_buf, err_buf = io.StringIO(), io.StringIO()
    old = os.environ.get("COSMIC_GATE_NONCE")
    os.environ["COSMIC_GATE_NONCE"] = nonce
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = run_gate(
                hook="grok",
                verb_check=False,
                stdin=json.dumps(envelope),
                exec_mode="safe",
            )
    finally:
        if old is None:
            os.environ.pop("COSMIC_GATE_NONCE", None)
        else:
            os.environ["COSMIC_GATE_NONCE"] = old
    return out_buf.getvalue(), err_buf.getvalue(), code


def run_boot_canary(*, cwd: str = "/tmp", nonce: str | None = None) -> int:
    """Run the full canary set. Return 0 on pass, 1 on fail. Diagnostics → stderr."""

    # Verb existence
    try:
        if run_gate(hook="grok", verb_check=True) != 0:
            print("canary: FAIL — cosmic-cli gate --verb-check non-zero", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"canary: FAIL — gate verb missing/broken: {e}", file=sys.stderr)
        return 1

    nonce = nonce or secrets.token_hex(16)
    if not _NONCE_RE.match(nonce):
        print("canary: FAIL — internal nonce malformed", file=sys.stderr)
        return 1

    expected = f"COSMIC-ALLOW {SENTINEL_VERSION} {nonce}\n"
    deny_cases, allow_cases = canary_envelopes(cwd=cwd)

    for label, env in deny_cases:
        out, err, code = _run_gate_capture(env, nonce)
        if out.strip().startswith("COSMIC-ALLOW"):
            print(
                f"canary: FAIL — deny case {label!r} emitted sentinel (stdout={out!r})",
                file=sys.stderr,
            )
            return 1
        if out not in ("", "\n"):
            # Any non-empty non-sentinel stdout is also a contract break
            print(
                f"canary: FAIL — deny case {label!r} had non-empty stdout: {out!r}",
                file=sys.stderr,
            )
            return 1
        print(f"canary: ok deny {label}", file=sys.stderr)

    for label, env in allow_cases:
        out, err, code = _run_gate_capture(env, nonce)
        if out != expected:
            print(
                f"canary: FAIL — allow case {label!r} expected {expected!r} got {out!r}",
                file=sys.stderr,
            )
            return 1
        print(f"canary: ok allow {label}", file=sys.stderr)

    print("canary: PASS — gate live and discriminating", file=sys.stderr)
    return 0


def main(argv: List[str] | None = None) -> int:
    return run_boot_canary()


if __name__ == "__main__":
    raise SystemExit(main())
