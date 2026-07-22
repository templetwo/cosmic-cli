"""The dashboard auto-start allowlist: which invocations may have side effects.

Mission Control used to be excluded from `gate` by a DENYLIST, and its log used
to land in `~/.cosmic-cli/` — the ranking-protected token store. Both bit us. The
denylist is the interesting half: `gate` is the PreToolUse hook and fires on
every tool call, so the failure mode of a forgotten entry is a server spawn and a
log write on the hot path. Forgetting must fail SAFE, which means the set has to
be an allowlist.

The load-bearing test here is test_a_command_named_nowhere_never_spawns: it
registers a subcommand whose name appears nowhere in main.py, so no membership
list can name it. Under the allowlist it cannot spawn. Under a denylist it spawns
by default and the test fails. That is the by-construction guarantee — a test
that only pins `gate` would pass happily against a re-inverted denylist.

Nothing here starts a real dashboard: the spawn boundary is always patched.
"""

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from cosmic_cli import main as main_module
from cosmic_cli.main import _DASHBOARD_SUBCOMMANDS, _dashboard_log_path, cli

# A valid gate nonce so `gate` reaches its real body instead of bailing early.
NONCE = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"


@pytest.fixture
def spawns(monkeypatch):
    """Spy on the one function that can start a dashboard. Returns the call list."""
    calls: list = []
    monkeypatch.setattr(main_module, "_ensure_dashboard", lambda *a, **k: calls.append((a, k)))
    return calls


@pytest.fixture
def sealed_home(tmp_path, monkeypatch):
    """Redirect HOME so nothing in this module can touch the real token store."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


# ---- the gate path -----------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["gate"],
        ["gate", "--verb-check"],
        ["gate", "--hook", "claude"],
        ["gate", "--hook", "grok", "--mode", "full"],
    ],
)
def test_gate_never_spawns_the_dashboard(argv, spawns, sealed_home, monkeypatch):
    """The ledger's acceptance line: no gate invocation spawns the dashboard."""
    monkeypatch.setenv("COSMIC_GATE_NONCE", NONCE)
    CliRunner().invoke(cli, argv, input='{"tool_name": "Read", "tool_input": {}}')
    assert not spawns, f"{argv} spawned a dashboard on the gate path"


def test_gate_is_not_in_the_allowlist():
    assert "gate" not in _DASHBOARD_SUBCOMMANDS
    assert "version" not in _DASHBOARD_SUBCOMMANDS


# ---- the by-construction guarantee (fails under a re-inverted denylist) ------


def test_a_command_named_nowhere_never_spawns(spawns, sealed_home):
    """A brand-new subcommand must default to NOT spawning.

    `quiescent-nonesuch` is named in no membership list anywhere — the assertion
    below proves that about main.py rather than trusting it. So the only way it
    can spawn is if the callback's default for unknown commands is "spawn",
    i.e. if someone re-inverted the allowlist back into a denylist.
    """
    name = "quiescent-nonesuch"
    assert name not in Path(main_module.__file__).read_text(encoding="utf-8")

    @cli.command(name, hidden=True)
    def _probe() -> None:
        pass

    try:
        result = CliRunner().invoke(cli, [name])
        assert result.exit_code == 0, result.output
        assert not spawns, (
            "an unlisted subcommand spawned the dashboard — the allowlist has "
            "been inverted back into a denylist and new commands now fail OPEN"
        )
    finally:
        cli.commands.pop(name, None)


def test_every_command_outside_the_allowlist_is_silent(spawns, sealed_home, monkeypatch):
    """Sweep the whole live command surface, not a hand-picked pair."""
    monkeypatch.setenv("COSMIC_GATE_NONCE", NONCE)
    ctx = click.Context(cli)
    unlisted = [n for n in cli.list_commands(ctx) if n not in _DASHBOARD_SUBCOMMANDS]
    assert unlisted, "sanity: there should be commands outside the allowlist"
    for name in unlisted:
        spawns.clear()
        # --help stops each command before its body while still running the
        # group callback, which is the only thing under test here.
        CliRunner().invoke(cli, [name, "--help"])
        assert not spawns, f"`{name}` is not allowlisted but spawned a dashboard"


# ---- the allowlisted side still works ----------------------------------------


def test_an_allowlisted_subcommand_does_spawn(spawns, sealed_home):
    """`stargazer` is allowlisted and prints its own help with no side effects."""
    result = CliRunner().invoke(cli, ["stargazer"])
    assert "Usage:" in result.output
    assert len(spawns) == 1


def test_the_allowlist_only_names_real_commands():
    """A typo'd entry is a silently dead entry; pin the names to the surface."""
    ctx = click.Context(cli)
    assert _DASHBOARD_SUBCOMMANDS <= set(cli.list_commands(ctx))


def test_explicit_dashboard_command_still_starts_it(spawns, sealed_home, monkeypatch):
    """`dashboard` is deliberately NOT allowlisted — it opts in at its own call
    site so that --port is honoured. That opt-in must keep working."""
    monkeypatch.setattr(main_module, "_dashboard_running", lambda port=0: False)
    monkeypatch.setattr(main_module.time, "sleep", lambda _s: None)
    result = CliRunner().invoke(cli, ["dashboard", "--no-open", "--port", "4999"])
    assert result.exit_code == 0
    assert spawns == [((4999,), {})], "dashboard did not start on the requested port"


# ---- bare `cosmic-cli` -------------------------------------------------------


def test_bare_invocation_does_not_spawn(spawns, sealed_home):
    """No subcommand => `invoked_subcommand` is None => not in the allowlist.

    Click's no_args_is_help prints help and exits before the group body today, so
    the callback is driven directly to pin the None branch itself rather than
    click's current short-circuit.
    """
    assert None not in _DASHBOARD_SUBCOMMANDS
    with click.Context(cli) as ctx:
        ctx.ensure_object(dict)
        ctx.invoked_subcommand = None
        cli.callback(verbose=False)
    assert not spawns

    CliRunner().invoke(cli, [])
    assert not spawns


# ---- the env opt-out ---------------------------------------------------------


@pytest.fixture
def popen_calls(monkeypatch):
    """Patch the process boundary: _ensure_dashboard is the unit under test now."""
    calls: list = []
    monkeypatch.setattr(main_module, "_dashboard_running", lambda port=0: False)
    monkeypatch.setattr(main_module.subprocess, "Popen",
                        lambda *a, **k: calls.append((a, k)))
    return calls


def test_cosmic_no_dashboard_suppresses(popen_calls, sealed_home, monkeypatch):
    monkeypatch.setenv("COSMIC_NO_DASHBOARD", "1")
    main_module._ensure_dashboard()
    assert not popen_calls


def test_without_the_opt_out_it_would_have_spawned(popen_calls, sealed_home, monkeypatch):
    """The control: proves the suppression test above is not vacuous."""
    monkeypatch.delenv("COSMIC_NO_DASHBOARD", raising=False)
    main_module._ensure_dashboard()
    assert len(popen_calls) == 1


# ---- the log path stays out of the protected tree ----------------------------


def test_dashboard_log_is_outside_the_protected_tree(sealed_home):
    """Assert the relationship, not a string: no hardcoded path can drift here."""
    protected = Path.home() / ".cosmic-cli"
    log = _dashboard_log_path()
    assert log != protected
    assert protected not in log.parents
    assert not str(log).startswith(f"{protected}/")


def test_failure_message_points_at_the_real_log(sealed_home, spawns, monkeypatch):
    """The old message sent operators to ~/.cosmic-cli/dashboard.log, which had
    already been relocated out of the protected tree."""
    monkeypatch.setattr(main_module, "_dashboard_running", lambda port=0: False)
    monkeypatch.setattr(main_module.time, "sleep", lambda _s: None)
    # Capture what the command emits, not what Rich renders: a tmp HOME makes the
    # path long enough that terminal-width truncation would eat the assertion.
    said: list = []
    monkeypatch.setattr(main_module.console, "print", lambda *a, **k: said.extend(a))
    result = CliRunner().invoke(cli, ["dashboard", "--no-open"])
    assert result.exit_code == 0
    line = " ".join(said)
    assert "failed to start" in line
    assert str(_dashboard_log_path()) in line
    assert str(Path.home() / ".cosmic-cli" / "dashboard.log") not in line
