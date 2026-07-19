"""Privilege ranking — L0 cannot self-approve PAUSE (class close, not patch).

Exact reproducer that previously returned allow:
  model runs accept-pause → stages operator_approval_token → retry → OPEN

Must now return deny at every step of the L0 path. L2 (TTY / COSMIC_L2_OPERATOR
/ explicit COSMIC_APPROVAL_TOKEN in the operator-owned process env) still works.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from cosmic_cli.gate import run_gate
from cosmic_cli.main import cli
from cosmic_cli.ranking import (
    is_approval_token_path,
    is_interactive_l2,
    require_l2_tty,
    touches_approval_surface,
)
from cosmic_cli.secrets import is_sensitive_path
from cosmic_cli.shell_guard import check_shell

NONCE = "a" * 32
SENTINEL = f"COSMIC-ALLOW v1 {NONCE}"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Isolated HOME + COSMIC.md with a PAUSE rule; never touch real token store."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("COSMIC_APPROVAL_TOKEN", raising=False)
    monkeypatch.delenv("COSMIC_L2_OPERATOR", raising=False)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "COSMIC.md").write_text(
        "## Compass Rules\n\n"
        "| ID | Type | Scope | Pattern |\n"
        "|----|------|-------|---------|\n"
        "| pause-deploy | PAUSE | SHELL | deploy |\n",
        encoding="utf-8",
    )
    return root


def _env(command: str, cwd: str) -> dict:
    return {
        "toolName": "run_terminal_command",
        "toolInput": {"command": command},
        "cwd": cwd,
    }


def _run_gate(envelope: dict, capsys, monkeypatch, approval: str | None = None):
    monkeypatch.setenv("COSMIC_GATE_NONCE", NONCE)
    if approval:
        monkeypatch.setenv("COSMIC_APPROVAL_TOKEN", approval)
    else:
        monkeypatch.delenv("COSMIC_APPROVAL_TOKEN", raising=False)
    code = run_gate(hook="grok", stdin=json.dumps(envelope))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def _last_tok() -> str | None:
    p = Path.home() / ".cosmic-cli" / "last_pause_token.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("token")


# ---- pure ranking primitives ----

def test_touches_approval_surface_patterns():
    assert touches_approval_surface("cosmic-cli helix accept-pause")
    assert touches_approval_surface("cat ~/.cosmic-cli/last_pause_token.json")
    assert touches_approval_surface("export COSMIC_APPROVAL_TOKEN=tok-abc")
    assert touches_approval_surface("echo x > operator_approval_token")
    assert touches_approval_surface(None) is None
    assert touches_approval_surface("deploy prod") is None


def test_check_shell_witnesses_approval_surface():
    blocked = check_shell("cosmic-cli helix accept-pause", exec_mode="safe")
    assert blocked and "privilege ranking" in blocked
    # interactive mode still blocks (ranking is not safe-mode-only)
    assert check_shell("show-pause-token", exec_mode="interactive")
    # full = L2/L3 blast opt-in
    assert check_shell("cosmic-cli helix accept-pause", exec_mode="full") is None


def test_token_paths_are_sensitive():
    assert is_approval_token_path("/Users/x/.cosmic-cli/last_pause_token.json")
    assert is_sensitive_path(str(Path.home() / ".cosmic-cli" / "operator_approval_token"))
    assert is_sensitive_path("/tmp/foo.txt") is False or not is_approval_token_path(
        "/tmp/foo.txt"
    )


def test_require_l2_tty_refuses_noninteractive(monkeypatch):
    monkeypatch.delenv("COSMIC_L2_OPERATOR", raising=False)
    # Force non-TTY view regardless of pytest's streams
    monkeypatch.setattr("cosmic_cli.ranking.is_interactive_l2", lambda: False)
    assert require_l2_tty("helix accept-pause")
    monkeypatch.setattr("cosmic_cli.ranking.is_interactive_l2", lambda: True)
    assert require_l2_tty("helix accept-pause") is None


# ---- THE class-closing conformance: exact reproducer must deny ----

def test_l0_accept_pause_then_retry_denies(ws, capsys, monkeypatch):
    """Ranking proof: model runs accept-pause, retries → deny (was allow).

    Steps mirror the self-approval hole:
      1) L0 triggers PAUSE → token minted to last_pause_token.json
      2) L0 shell invokes accept-pause without TTY → refuse, no stage file
      3) L0 retries blocked action without operator env → still deny
    """
    # 1. Mint
    code, out, err = _run_gate(
        _env("deploy prod", str(ws)), capsys, monkeypatch
    )
    assert out == ""
    assert "PAUSE" in err
    tok = _last_tok()
    assert tok and tok.startswith("tok-")

    # 2. L0 accept-pause (no TTY, no COSMIC_L2_OPERATOR)
    monkeypatch.setattr("cosmic_cli.ranking.is_interactive_l2", lambda: False)
    runner = CliRunner()
    result = runner.invoke(cli, ["helix", "accept-pause"])
    assert result.exit_code == 4
    assert "L2-only" in result.output or "privilege ranking" in result.output
    stage = Path.home() / ".cosmic-cli" / "operator_approval_token"
    assert not stage.is_file(), "L0 must not stage operator_approval_token"

    # 3. Retry without L2-presented token → deny (no self-approval)
    code, out, err = _run_gate(
        _env("deploy prod", str(ws)), capsys, monkeypatch
    )
    assert out == "", "L0 retry after failed accept-pause must not OPEN"
    assert "PAUSE" in err or out == ""


def test_l0_shell_accept_pause_denied_at_gate(ws, capsys, monkeypatch):
    """L1 WITNESS: shell tool call that runs accept-pause never mints/opens."""
    code, out, err = _run_gate(
        _env("cosmic-cli helix accept-pause", str(ws)), capsys, monkeypatch
    )
    assert out == ""
    assert "privilege ranking" in err or "check_shell" in err or "BLOCKED" in err
    # Must not have staged anything via side effects of a would-be allow
    assert not (Path.home() / ".cosmic-cli" / "operator_approval_token").is_file()


def test_l0_cat_token_denied_at_gate(ws, capsys, monkeypatch):
    _run_gate(_env("deploy prod", str(ws)), capsys, monkeypatch)
    code, out, err = _run_gate(
        _env("cat ~/.cosmic-cli/last_pause_token.json", str(ws)),
        capsys,
        monkeypatch,
    )
    assert out == ""
    assert "privilege ranking" in err or "BLOCKED" in err


def test_l0_read_token_file_tool_denied(ws, capsys, monkeypatch):
    _run_gate(_env("deploy prod", str(ws)), capsys, monkeypatch)
    tok_path = str(Path.home() / ".cosmic-cli" / "last_pause_token.json")
    env = {
        "toolName": "read_file",
        "toolInput": {"target_file": tok_path},
        "cwd": str(ws),
    }
    code, out, err = _run_gate(env, capsys, monkeypatch)
    assert out == ""
    assert "sensitive" in err.lower() or "BLOCKED" in err


def test_l2_env_token_still_opens(ws, capsys, monkeypatch):
    """Ranking does not break the real L2 path: operator-owned env claim."""
    _run_gate(_env("deploy prod", str(ws)), capsys, monkeypatch)
    tok = _last_tok()
    code, out, err = _run_gate(
        _env("deploy prod", str(ws)), capsys, monkeypatch, approval=tok
    )
    assert out.strip() == SENTINEL


def test_l2_operator_flag_allows_accept_pause(ws, monkeypatch):
    """COSMIC_L2_OPERATOR=1 is the scripted L2 break-glass (still not L0 default)."""
    _run_gate  # ensure module import path warm
    # Mint a token under isolated HOME
    from cosmic_cli.gateway import ApprovalManager

    mgr = ApprovalManager()
    sha = "c" * 64
    tok = mgr.mint_token(sha)
    store = Path.home() / ".cosmic-cli" / "last_pause_token.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps({"token": tok, "channel": "gate-SHELL", "action_sha256": sha}),
        encoding="utf-8",
    )
    monkeypatch.setenv("COSMIC_L2_OPERATOR", "1")
    # Bypass isatty via the env flag path inside is_interactive_l2
    assert is_interactive_l2() is True
    runner = CliRunner()
    result = runner.invoke(cli, ["helix", "accept-pause"])
    assert result.exit_code == 0, result.output
    assert (Path.home() / ".cosmic-cli" / "operator_approval_token").is_file()
