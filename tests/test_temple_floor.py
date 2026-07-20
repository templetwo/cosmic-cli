"""Temple floor: provision + canary (default install is fail-closed)."""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

from cosmic_cli.temple_floor import (
    ensure_temple_default_profile,
    ensure_temple_sandbox_toml,
    provision_temple_floor,
    run_floor_canary,
    token_deny_paths,
)


def test_token_deny_paths_are_files_not_whole_tree(tmp_path):
    paths = token_deny_paths(tmp_path)
    assert all(p.name in {
        "last_pause_token.json",
        "operator_approval_token",
        "local_approvals.json",
        "local_approvals.json.lock",
    } for p in paths)
    assert not any(p.name == ".cosmic-cli" for p in paths)


def test_provision_writes_temple_and_default(tmp_path):
    msg = provision_temple_floor(home=tmp_path)
    assert "temple floor provisioned" in msg
    sb = (tmp_path / ".grok" / "sandbox.toml").read_text(encoding="utf-8")
    assert "[profiles.temple]" in sb
    assert "last_pause_token.json" in sb
    assert 'extends = "workspace"' in sb
    # never deny whole tree as a lone deny of the directory without files
    assert str(tmp_path / ".cosmic-cli") + '"' not in sb or "last_pause" in sb
    cfg = (tmp_path / ".grok" / "config.toml").read_text(encoding="utf-8")
    assert 'profile = "temple"' in cfg


def test_provision_idempotent_preserves_other_profiles(tmp_path):
    grok = tmp_path / ".grok"
    grok.mkdir()
    (grok / "sandbox.toml").write_text(
        '[profiles.other]\nextends = "workspace"\ndeny = ["/tmp/x"]\n',
        encoding="utf-8",
    )
    ensure_temple_sandbox_toml(home=tmp_path)
    ensure_temple_sandbox_toml(home=tmp_path)  # twice
    text = (grok / "sandbox.toml").read_text(encoding="utf-8")
    assert "[profiles.other]" in text
    assert "[profiles.temple]" in text
    assert text.count("[profiles.temple]") == 1


def test_floor_canary_fails_without_provision(tmp_path):
    assert run_floor_canary(home=tmp_path) == 1


def test_floor_canary_passes_after_provision(tmp_path):
    provision_temple_floor(home=tmp_path)
    # On macOS live Seatbelt runs; on Linux skips live with toml ok
    code = run_floor_canary(home=tmp_path)
    assert code == 0


@pytest.mark.skipif(platform.system() != "Darwin", reason="Seatbelt live probe")
def test_floor_canary_live_seatbelt_blocks_token(tmp_path):
    provision_temple_floor(home=tmp_path)
    assert run_floor_canary(home=tmp_path) == 0
