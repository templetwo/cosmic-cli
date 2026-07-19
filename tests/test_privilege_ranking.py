"""Privilege ranking — L0 cannot self-approve PAUSE (class close, not patch).

Hierarchy under test:
  kernel floor (sandbox) → TTY ranking → classification DiD

Reproducers that must deny:
  - plain accept-pause then retry
  - base64|sh wrapping accept-pause (audit live hole)
  - python -c os.system(...) wrapping accept-pause
  - COSMIC_L2_OPERATOR-style env break-glass must not exist
"""

from __future__ import annotations

import base64
import json
import platform
from pathlib import Path

import pytest
from click.testing import CliRunner

from cosmic_cli.gate import run_gate
from cosmic_cli.main import cli
from cosmic_cli.ranking import (
    classification_corpus,
    is_approval_token_path,
    is_interactive_l2,
    is_opaque_shell_wrapper,
    require_l2_tty,
    touches_approval_surface,
)
from cosmic_cli.sandbox import (
    load_deny_paths,
    sandbox_available,
    seatbelt_profile,
    wrap_argv_for_l0_shell,
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


def test_base64_payload_still_hits_surface():
    """DiD: hidden accept-pause in base64 must still classify as surface."""
    payload = "cosmic-cli helix accept-pause"
    b64 = base64.b64encode(payload.encode()).decode()
    corpus = classification_corpus(f"echo {b64} | base64 -d")
    assert "accept-pause" in corpus.lower()
    assert touches_approval_surface(f"echo {b64} | base64 -d | sh")


def test_opaque_shell_wrappers_fail_closed():
    assert is_opaque_shell_wrapper("echo YWJj | base64 -d | sh")
    assert is_opaque_shell_wrapper('python3 -c "os.system(x)"')
    assert is_opaque_shell_wrapper("eval \"$cmd\"")
    assert is_opaque_shell_wrapper("deploy prod") is None


def test_check_shell_witnesses_approval_surface():
    blocked = check_shell("cosmic-cli helix accept-pause", exec_mode="safe")
    assert blocked and "privilege ranking" in blocked
    assert check_shell("show-pause-token", exec_mode="interactive")
    # full = L2/L3 blast opt-in
    assert check_shell("cosmic-cli helix accept-pause", exec_mode="full") is None


def test_check_shell_denies_base64_pipe_sh():
    payload = base64.b64encode(b"cosmic-cli helix accept-pause").decode()
    cmd = f"echo {payload} | base64 -d | sh"
    blocked = check_shell(cmd, exec_mode="safe")
    assert blocked, "base64|sh must not pass classification"
    assert "opaque" in blocked.lower() or "privilege ranking" in blocked.lower()


def test_check_shell_denies_python_os_system():
    cmd = 'python3 -c "import os,base64; os.system(base64.b64decode(\'YQ==\').decode())"'
    blocked = check_shell(cmd, exec_mode="safe")
    assert blocked


def test_token_paths_are_sensitive():
    assert is_approval_token_path("/Users/x/.cosmic-cli/last_pause_token.json")
    assert is_sensitive_path(str(Path.home() / ".cosmic-cli" / "operator_approval_token"))
    assert not is_approval_token_path("/tmp/foo.txt")


def test_require_l2_tty_refuses_noninteractive(monkeypatch):
    monkeypatch.setattr("cosmic_cli.ranking.is_interactive_l2", lambda: False)
    assert require_l2_tty("helix accept-pause")
    monkeypatch.setattr("cosmic_cli.ranking.is_interactive_l2", lambda: True)
    assert require_l2_tty("helix accept-pause") is None


def test_no_env_break_glass(monkeypatch):
    """COSMIC_L2_OPERATOR must NOT elevate L0 — that was the credential hole."""
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: False})())
    monkeypatch.setattr("sys.stdout", type("S", (), {"isatty": lambda self: False})())
    monkeypatch.setattr("sys.stderr", type("S", (), {"isatty": lambda self: False})())
    monkeypatch.setenv("COSMIC_L2_OPERATOR", "1")
    # Force pure function path without relying on real streams: patch isatty checks
    monkeypatch.setattr(
        "cosmic_cli.ranking.is_interactive_l2",
        lambda: False,  # even with env set, implementation must not treat flag as L2
    )
    # Direct: is_interactive_l2 after re-import logic — call real impl with no TTY
    from cosmic_cli import ranking as ranking_mod

    # Restore real is_interactive_l2 and break its streams
    def _no_tty():
        return False  # real function with env must still be false — re-read source contract

    # Contract: env flag is ignored. Prove by reading source path:
    import inspect

    src = inspect.getsource(ranking_mod.is_interactive_l2)
    assert "COSMIC_L2_OPERATOR" not in src
    assert ranking_mod.require_l2_tty("helix accept-pause")  # no TTY in CliRunner below


# ---- gate-level class close ----

def test_l0_accept_pause_then_retry_denies(ws, capsys, monkeypatch):
    code, out, err = _run_gate(_env("deploy prod", str(ws)), capsys, monkeypatch)
    assert out == ""
    assert "PAUSE" in err
    tok = _last_tok()
    assert tok and tok.startswith("tok-")

    monkeypatch.setattr("cosmic_cli.ranking.is_interactive_l2", lambda: False)
    runner = CliRunner()
    result = runner.invoke(cli, ["helix", "accept-pause"])
    assert result.exit_code == 4
    assert "L2-only" in result.output or "privilege ranking" in result.output
    assert not (Path.home() / ".cosmic-cli" / "operator_approval_token").is_file()

    code, out, err = _run_gate(_env("deploy prod", str(ws)), capsys, monkeypatch)
    assert out == ""


def test_l0_shell_accept_pause_denied_at_gate(ws, capsys, monkeypatch):
    code, out, err = _run_gate(
        _env("cosmic-cli helix accept-pause", str(ws)), capsys, monkeypatch
    )
    assert out == ""
    assert "privilege ranking" in err or "check_shell" in err or "BLOCKED" in err


def test_gate_denies_base64_accept_pause_evasion(ws, capsys, monkeypatch):
    """Audit live reproducer: echo b64 | base64 -d | sh must not OPEN."""
    payload = base64.b64encode(b"cosmic-cli helix accept-pause").decode()
    cmd = f"echo {payload} | base64 -d | sh"
    code, out, err = _run_gate(_env(cmd, str(ws)), capsys, monkeypatch)
    assert out == "", f"evasion must deny, got stdout={out!r} err={err!r}"
    assert "BLOCKED" in err or "privilege ranking" in err or "opaque" in err.lower()


def test_gate_denies_python_os_system_evasion(ws, capsys, monkeypatch):
    inner = base64.b64encode(b"cosmic-cli helix accept-pause").decode()
    cmd = (
        "python3 -c \"import os,base64; "
        f"os.system(base64.b64decode('{inner}').decode())\""
    )
    code, out, err = _run_gate(_env(cmd, str(ws)), capsys, monkeypatch)
    assert out == ""
    assert "BLOCKED" in err or "privilege ranking" in err or "opaque" in err.lower()


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
    _run_gate(_env("deploy prod", str(ws)), capsys, monkeypatch)
    tok = _last_tok()
    code, out, err = _run_gate(
        _env("deploy prod", str(ws)), capsys, monkeypatch, approval=tok
    )
    assert out.strip() == SENTINEL


def test_env_flag_does_not_unlock_accept_pause(ws, monkeypatch):
    """Removed break-glass: COSMIC_L2_OPERATOR=1 must not stage a token."""
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
    monkeypatch.setattr("cosmic_cli.ranking.is_interactive_l2", lambda: False)
    runner = CliRunner()
    result = runner.invoke(cli, ["helix", "accept-pause"])
    assert result.exit_code == 4
    assert not (Path.home() / ".cosmic-cli" / "operator_approval_token").is_file()


# ---- kernel floor ----

def test_sandbox_toml_declares_cosmic_cli_deny():
    paths = load_deny_paths()
    assert any(".cosmic-cli" in str(p) for p in paths)
    prof = seatbelt_profile()
    assert "deny file-read" in prof
    assert ".cosmic-cli" in prof or "cosmic-cli" in prof


def test_wrap_argv_uses_seatbelt_on_macos():
    argv = wrap_argv_for_l0_shell("echo hi")
    if platform.system() == "Darwin" and sandbox_available():
        assert argv[0].endswith("sandbox-exec") or "sandbox-exec" in argv[0]
        assert "-f" in argv
    else:
        assert "-c" in argv


@pytest.mark.skipif(
    not (platform.system() == "Darwin" and sandbox_available()),
    reason="Seatbelt floor only enforced on macOS with sandbox-exec",
)
def test_seatbelt_blocks_read_of_token_store(tmp_path, monkeypatch):
    """Live kernel floor: sandboxed process cannot read ~/.cosmic-cli."""
    import subprocess

    home = tmp_path / "home"
    store = home / ".cosmic-cli"
    store.mkdir(parents=True)
    secret = store / "last_pause_token.json"
    secret.write_text('{"token":"tok-secret"}\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    # Re-resolve deny paths under new HOME
    from cosmic_cli import sandbox as sb

    argv = sb.wrap_argv_for_l0_shell(f"cat {secret}")
    r = subprocess.run(argv, capture_output=True, text=True)
    # Seatbelt should deny; exit non-zero and/or no secret in stdout
    assert "tok-secret" not in (r.stdout or "")
    assert r.returncode != 0 or "Operation not permitted" in (r.stderr or "") or (
        r.stdout == "" and r.returncode != 0
    )
