#!/usr/bin/env python3
"""End-to-end exercise of the installed cosmic-gate-wrapper.sh."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WRAPPER = Path.home() / ".cosmic-cli" / "hooks" / "cosmic-gate-wrapper.sh"
CWD = str(Path(__file__).resolve().parent.parent)


def call(tool: str, tool_input: dict) -> tuple[int, str]:
    env = json.dumps({"toolName": tool, "toolInput": tool_input, "cwd": CWD})
    r = subprocess.run(
        ["bash", str(WRAPPER)],
        input=env,
        text=True,
        capture_output=True,
    )
    return r.returncode, (r.stdout or "").strip()


def main() -> int:
    if not WRAPPER.is_file():
        print(f"FAIL no wrapper at {WRAPPER}")
        return 2

    cases = [
        ("todo_write", "todo_write", {"todos": []}, True),
        ("search_tool", "search_tool", {"query": "x"}, True),
        ("python_print", "run_terminal_command", {"command": "python3 -c 'print(1)'"}, True),
        ("git_status", "run_terminal_command", {"command": "git status -sb"}, True),
        ("rm_rf", "run_terminal_command", {"command": "rm -rf /tmp/x"}, False),
        ("mcp_bare", "linear__create_issue", {"title": "x"}, False),
        ("unknown", "some_new_tool", {}, False),
    ]
    ok_n = 0
    for name, tool, inp, expect_allow in cases:
        code, out = call(tool, inp)
        # Wrapper: allow => decision allow + exit 0; deny => decision deny + exit 2
        allowed = '"decision": "allow"' in out or '"decision":"allow"' in out
        # also accept compact json
        if not allowed:
            try:
                allowed = json.loads(out).get("decision") == "allow"
            except Exception:
                allowed = False
        good = allowed is expect_allow
        mark = "OK" if good else "FAIL"
        print(f"{mark} allow={allowed} expect={expect_allow} rc={code} {name} out={out[:80]!r}")
        ok_n += int(good)
    print(f"\n{ok_n}/{len(cases)} wrapper e2e cases pass")
    return 0 if ok_n == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
