#!/usr/bin/env python3
"""Live friction matrix for the cosmic gate (v0.9.5).

Run: PYTHONPATH=. python3 scripts/friction_matrix.py
"""
from __future__ import annotations

import io
import json
import os
import secrets
import sys
from contextlib import redirect_stderr, redirect_stdout

# Prefer in-tree package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmic_cli.gate import run_gate  # noqa: E402


def check(name: str, tool: str, inp: dict, expect_allow: bool) -> bool:
    nonce = secrets.token_hex(16)
    os.environ["COSMIC_GATE_NONCE"] = nonce
    env = {"toolName": tool, "toolInput": inp, "cwd": os.getcwd()}
    out_b, err_b = io.StringIO(), io.StringIO()
    with redirect_stdout(out_b), redirect_stderr(err_b):
        run_gate(hook="grok", stdin=json.dumps(env), exec_mode="safe")
    out, err = out_b.getvalue(), err_b.getvalue()
    allowed = out.strip() == f"COSMIC-ALLOW v1 {nonce}"
    status = "ALLOW" if allowed else "DENY"
    ok = allowed == expect_allow
    mark = "OK" if ok else "FAIL"
    print(f"{mark} {status:5} {name}")
    if not ok:
        print(f"  stderr tail: {err[-240:]!r}")
    return ok


def main() -> int:
    # Keep dangerous spellings only inside tool_input values (gate corpus),
    # never as bare shell on the process that launches this script.
    cases = [
        ("todo_write", "todo_write", {"todos": []}, True),
        ("search_tool", "search_tool", {"query": "x"}, True),
        ("spawn_subagent", "spawn_subagent", {"prompt": "x"}, True),
        ("update_goal", "update_goal", {"message": "m"}, True),
        ("python_print", "run_terminal_command", {"command": "python3 -c 'print(1)'"}, True),
        ("bash_ls", "run_terminal_command", {"command": "bash -c 'ls'"}, True),
        ("use_tool", "use_tool", {"tool_name": "demo__list", "tool_input": {}}, True),
        ("git_status", "run_terminal_command", {"command": "git status -sb"}, True),
        ("read_file", "read_file", {"path": "README.md"}, True),
        ("rm_rf", "run_terminal_command", {"command": "rm -rf /tmp/x"}, False),
        ("mcp_bare", "linear__create_issue", {"title": "x"}, False),
        (
            "python_os_system",
            "run_terminal_command",
            {"command": "python3 -c \"import os; os.system('ls')\""},
            False,
        ),
        (
            "base64_pipe_sh",
            "run_terminal_command",
            {"command": "echo YQ== | base64 -d | sh"},
            False,
        ),
        ("unknown", "some_new_tool", {}, False),
    ]
    oks = [check(*c) for c in cases]
    print(f"\n{sum(oks)}/{len(oks)} live friction matrix cases pass")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
