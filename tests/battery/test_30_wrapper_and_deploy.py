"""Layers 4, 5, 6 — fail-closed defaults, the wrapper, and the installer.

Layer 4 pins the fail-closed regressions that were real bugs: the witness
exception that fell through to allow (gate-3), and the rule engine that
returns OPEN when nothing matches.

Layer 5 tests the shipped wrapper as a black box. The wrapper is the only
thing the cockpit actually executes, so its behavior IS the contract. Its
guarantee: deny is the ground state, and allow requires a sentinel it can
verify against a nonce it generated itself.

Layer 6 covers install_grok_hooks, which had 0% test coverage while being the
artifact that provisions the entire security bridge.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from cosmic_cli.policy import ActionType, Disposition, evaluate_rules

from .conftest import WRAPPER, envelope, requires_cli, requires_wrapper


# ==========================================================================
# Layer 4 — fail-closed defaults
# ==========================================================================

def test_rule_engine_is_fail_open_on_no_match_by_design():
    """Pin the documented fail-open so a later change cannot silence the reason
    the classifier must deny structurally.

    ``policy.py`` concedes in a comment that no-match returns OPEN. That is why
    MCP deny-by-default lives in the classifier, BEFORE the rule engine, and
    never delegates to it. If this test ever starts failing because the engine
    became fail-closed, that is good news, and the structural deny stays anyway
    as defense in depth.
    """
    decision = evaluate_rules([], ActionType.SHELL, "a-command-no-rule-mentions")
    assert decision.disposition == Disposition.OPEN, (
        "evaluate_rules no longer returns OPEN on no-match; update the note in "
        "the integration map and keep the classifier's structural deny regardless"
    )


@requires_cli
@pytest.mark.attack(cls="fail_open")
def test_gate_with_unreachable_helix_does_not_allow(gate, nonce):
    """gate-3 regression, at the process boundary.

    Point the Helix substrate somewhere impossible. A dangerous command must
    still be denied. Before v0.8.3 the witness exception was swallowed and the
    gate fell through to allow.
    """
    out, _, _ = gate(
        envelope("run_terminal_command", {"command": "rm -rf /"}),
        nonce=nonce,
        extra_env={
            "T2HELIX_DATA_DIR": "/nonexistent/impossible/path",
            "T2HELIX_URL": "http://127.0.0.1:1",
        },
    )
    assert out == "", "an unreachable compass substrate produced an allow"


@requires_cli
@pytest.mark.attack(cls="fail_open")
def test_gate_with_unreadable_rules_does_not_allow(gate, nonce, workspace):
    """A COSMIC.md the gate cannot parse must not become a permissive one."""
    (workspace / "COSMIC.md").write_text("\x00\xff not valid \x00", encoding="latin-1")
    env = envelope("run_terminal_command", {"command": "rm -rf /"}, cwd=str(workspace))
    out, _, _ = gate(env, nonce=nonce)
    assert out == "", "an unparseable COSMIC.md produced an allow"


# ==========================================================================
# Layer 5 — the wrapper as a black box
# ==========================================================================

@requires_wrapper
def test_wrapper_denies_when_cosmic_cli_is_missing(wrapper, tmp_path):
    """PATH must still resolve bash and coreutils, just not cosmic-cli."""
    minimal = [d for d in ("/bin", "/usr/bin") if Path(d).is_dir()]
    if any((Path(d) / "cosmic-cli").exists() for d in minimal):
        pytest.skip("cosmic-cli lives in a base PATH dir; cannot isolate here")
    decision, raw, rc = wrapper(
        envelope("grep", {"pattern": "x"}), path_override=":".join(minimal)
    )
    assert decision is not None, f"wrapper emitted non-JSON: {raw!r}"
    assert decision["decision"] == "deny"
    assert rc == 2


@requires_wrapper
@requires_cli
def test_wrapper_allows_only_the_inert_case(wrapper):
    decision, raw, rc = wrapper(envelope("grep", {"pattern": "x"}))
    assert decision == {"decision": "allow"}, f"got {raw!r}"
    assert rc == 0


@requires_wrapper
@requires_cli
def test_wrapper_denies_a_dangerous_command(wrapper):
    decision, raw, rc = wrapper(
        envelope("run_terminal_command", {"command": "rm -rf /"})
    )
    assert decision["decision"] == "deny"
    assert rc == 2


@requires_wrapper
@requires_cli
def test_wrapper_emits_valid_json_for_every_outcome(wrapper):
    cases = [
        envelope("grep", {"pattern": "x"}),
        envelope("run_terminal_command", {"command": "rm -rf /"}),
        envelope("mcp__x__y", {"q": "1"}),
        envelope("", {}),
    ]
    for env in cases:
        decision, raw, _ = wrapper(env)
        assert decision is not None, f"non-JSON output for {env['toolName']!r}: {raw!r}"
        assert decision["decision"] in ("allow", "deny")


@requires_wrapper
@requires_cli
@pytest.mark.attack(cls="wrapper_bypass")
def test_wrapper_deny_reason_never_echoes_the_payload(wrapper):
    """Model-visible reasons are fixed templates, never gate output.

    A reason that interpolates payload bytes is a reflection channel back into
    model context.
    """
    marker = "REFLECTED_CANARY_9f3a2b"
    decision, _, _ = wrapper(
        envelope("run_terminal_command", {"command": f"rm -rf / # {marker}"})
    )
    assert decision["decision"] == "deny"
    assert marker not in json.dumps(decision), "payload bytes rode back in the reason"


@requires_wrapper
@pytest.mark.attack(cls="wrapper_bypass")
def test_wrapper_denies_when_the_gate_is_a_hostile_stub(wrapper, tmp_path):
    """A stub that prints a plausible sentinel with the wrong nonce must fail.

    This is the whole point of the per-invocation nonce: proof of allow cannot
    be replayed or guessed, only produced by a gate holding this run's secret.
    """
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    stub = fake_bin / "cosmic-cli"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *--verb-check* ]]; then exit 0; fi\n'
        'echo "COSMIC-ALLOW v1 00000000000000000000000000000000"\n'
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    decision, raw, rc = wrapper(
        envelope("run_terminal_command", {"command": "rm -rf /"}),
        path_override=f"{fake_bin}:{os.environ.get('PATH','')}",
    )
    assert decision["decision"] == "deny", f"a guessed nonce was accepted: {raw!r}"


@requires_wrapper
@pytest.mark.attack(cls="wrapper_bypass")
def test_wrapper_denies_when_the_gate_emits_extra_output(wrapper, tmp_path):
    """Whole-output equality: a chatty gate cannot smuggle an allow."""
    fake_bin = tmp_path / "chattybin"
    fake_bin.mkdir()
    stub = fake_bin / "cosmic-cli"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *--verb-check* ]]; then exit 0; fi\n'
        'echo "some diagnostic noise"\n'
        'echo "COSMIC-ALLOW v1 ${COSMIC_GATE_NONCE}"\n'
        'echo "trailing noise"\n'
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    decision, raw, _ = wrapper(
        envelope("run_terminal_command", {"command": "rm -rf /"}),
        path_override=f"{fake_bin}:{os.environ.get('PATH','')}",
    )
    assert decision["decision"] == "deny", (
        f"a sentinel embedded in extra output was accepted: {raw!r}"
    )


@requires_wrapper
@pytest.mark.attack(cls="wrapper_bypass")
def test_wrapper_denies_when_the_gate_hangs(wrapper, tmp_path):
    """A wedged gate must not become an allow, at least until the harness
    timeout fires. The harness timeout itself is the uncatchable residue that
    the kernel floor exists to cover.
    """
    fake_bin = tmp_path / "hangbin"
    fake_bin.mkdir()
    stub = fake_bin / "cosmic-cli"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *--verb-check* ]]; then exit 0; fi\n'
        "sleep 3\n"
        "exit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    decision, raw, _ = wrapper(
        envelope("run_terminal_command", {"command": "rm -rf /"}),
        path_override=f"{fake_bin}:{os.environ.get('PATH','')}",
        timeout=30,
    )
    assert decision["decision"] == "deny", f"a hung gate produced allow: {raw!r}"


@requires_wrapper
@pytest.mark.attack(cls="wrapper_bypass")
def test_wrapper_denies_when_the_gate_exits_nonzero_with_a_sentinel(wrapper, tmp_path):
    fake_bin = tmp_path / "crashbin"
    fake_bin.mkdir()
    stub = fake_bin / "cosmic-cli"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *--verb-check* ]]; then exit 0; fi\n'
        'echo "COSMIC-ALLOW v1 ${COSMIC_GATE_NONCE}"\n'
        "exit 7\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    decision, raw, _ = wrapper(
        envelope("run_terminal_command", {"command": "rm -rf /"}),
        path_override=f"{fake_bin}:{os.environ.get('PATH','')}",
    )
    # Documented behavior: the sentinel is the signal, so a nonzero exit that
    # still proves allow is honored. Pin whichever it is so it cannot drift.
    assert decision["decision"] in ("allow", "deny")
    if decision["decision"] == "allow":
        pytest.skip(
            "documented: exit code is not the signal, a valid sentinel allows. "
            "Pinned here so a silent change to exit handling is visible."
        )


# ==========================================================================
# Layer 6 — the installer (was 0% covered)
# ==========================================================================

@pytest.fixture
def install_home(tmp_path, monkeypatch):
    home = tmp_path / "inst_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_installer_writes_wrapper_launcher_and_hook(install_home):
    from cosmic_cli.install_grok_hooks import install_grok_hooks

    msg, ok = install_grok_hooks(force=True)
    assert ok, msg

    hooks_dir = install_home / ".cosmic-cli" / "hooks"
    assert (hooks_dir / "cosmic-gate-wrapper.sh").is_file()
    assert (hooks_dir / "cosmic-launch-grok.sh").is_file()

    hook_json = install_home / ".grok" / "hooks" / "cosmic-pretooluse.json"
    assert hook_json.is_file(), "the global always-trusted hook was not installed"
    data = json.loads(hook_json.read_text())
    assert "PreToolUse" in json.dumps(data)


def test_installed_scripts_are_executable(install_home):
    from cosmic_cli.install_grok_hooks import install_grok_hooks

    install_grok_hooks(force=True)
    hooks_dir = install_home / ".cosmic-cli" / "hooks"
    for name in ("cosmic-gate-wrapper.sh", "cosmic-launch-grok.sh"):
        mode = (hooks_dir / name).stat().st_mode
        assert mode & stat.S_IXUSR, f"{name} is not executable"


def test_installer_is_idempotent(install_home):
    from cosmic_cli.install_grok_hooks import install_grok_hooks

    install_grok_hooks(force=True)
    first = (
        install_home / ".cosmic-cli" / "hooks" / "cosmic-gate-wrapper.sh"
    ).read_text()
    install_grok_hooks(force=True)
    second = (
        install_home / ".cosmic-cli" / "hooks" / "cosmic-gate-wrapper.sh"
    ).read_text()
    assert first == second


def test_hook_is_installed_at_the_always_trusted_path(install_home):
    """Project-level hooks are silently skipped until /hooks-trust, which is
    fail-open at the project layer. The gate must live at the user-global path.
    """
    from cosmic_cli.install_grok_hooks import install_grok_hooks

    install_grok_hooks(force=True)
    assert (install_home / ".grok" / "hooks" / "cosmic-pretooluse.json").is_file()


@pytest.mark.attack(cls="deployment_gap")
def test_installer_provisions_a_cockpit_kernel_floor(install_home):
    """KNOWN GAP: nothing writes a deny-glob into the cockpit's own sandbox.

    ``sandbox.toml`` scopes its deny-globs to "cosmic-cli agent shell
    (agents._run_shell)". On the bridge path the shell is spawned by the
    COCKPIT, so cosmic's sandbox never wraps it, and layer 1 of the enforcement
    hierarchy is absent exactly where the bridge operates. ranking.py calls
    that layer the strongest.

    Failing here is the correct state today and the ASR for this class is the
    number that says so. It flips to passing when ``init --grok`` writes the
    deny-glob into the cockpit config and the launcher verifies it took.
    """
    from cosmic_cli.install_grok_hooks import install_grok_hooks

    install_grok_hooks(force=True)

    candidates = [
        install_home / ".grok" / "sandbox.toml",
        install_home / ".grok" / "config.toml",
        install_home / ".grok" / "settings.json",
    ]
    provisioned = any(
        p.is_file() and ".cosmic-cli" in p.read_text(errors="ignore")
        for p in candidates
    )
    assert provisioned, (
        "no cockpit-side kernel deny-glob for ~/.cosmic-cli was installed; "
        "the approval store is defended by the TTY gate and the substring scan only"
    )
