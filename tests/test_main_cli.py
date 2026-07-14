"""CLI surface smoke tests (audit H6 partial)."""

from click.testing import CliRunner

from cosmic_cli.main import cli


def test_cli_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["--help"])
    assert r.exit_code == 0
    assert "do" in r.output
    assert "helix" in r.output
    assert "review" in r.output


def test_doctor_runs():
    runner = CliRunner()
    r = runner.invoke(cli, ["doctor"])
    # may fail api without key in isolated env — still should not crash import
    assert r.exit_code in (0, 1, 2)
    assert "Cosmic doctor" in r.output or "doctor" in r.output.lower() or r.exit_code == 0


def test_sessions_empty_ok(tmp_path, monkeypatch):
    monkeypatch.setattr("cosmic_cli.main.SESSION_DIR", tmp_path / "sessions")
    runner = CliRunner()
    r = runner.invoke(cli, ["sessions"])
    assert r.exit_code == 0


def test_init_writes(tmp_path):
    runner = CliRunner()
    r = runner.invoke(cli, ["init", str(tmp_path)])
    assert r.exit_code == 0
    assert (tmp_path / "COSMIC.md").is_file()


def test_helix_status_command():
    runner = CliRunner()
    r = runner.invoke(cli, ["helix", "status"])
    # available or not — command should return structured output
    assert r.exit_code == 0
    assert "T2HELIX" in r.output or "available" in r.output
