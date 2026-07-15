"""Local policy gateway wired into StargazerAgent._compass_gate."""

from pathlib import Path

from cosmic_cli.agents import StargazerAgent


def _agent(tmp_path: Path, **kwargs) -> StargazerAgent:
    return StargazerAgent(
        "t",
        api_key="test",
        quiet=True,
        use_helix=False,
        show_progress=False,
        work_dir=str(tmp_path),
        **kwargs,
    )


def test_local_witness_blocks_rm(tmp_path: Path):
    (tmp_path / "COSMIC.md").write_text(
        """
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| destructive-rm | WITNESS | SHELL,CODE | rm -rf |
"""
    )
    agent = _agent(tmp_path)
    out = agent._compass_gate("rm -rf /tmp/x", kind="SHELL")
    assert out is not None
    assert "BLOCKED" in out
    assert "local policy" in out


def test_local_pause_mints_token_then_allows_with_token(tmp_path: Path, monkeypatch):
    store = tmp_path / "approvals.json"
    (tmp_path / "COSMIC.md").write_text(
        """
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| outbound-net | PAUSE | SHELL | curl |
"""
    )
    agent = _agent(tmp_path)
    agent._approval_mgr = type(agent._approval_mgr)(store_path=store)
    agent._gateway.approval_manager = agent._approval_mgr

    blocked = agent._compass_gate("curl https://example.com", kind="SHELL")
    assert blocked is not None
    assert "PAUSE" in blocked
    assert "tok-" in blocked

    # Extract token from block message
    tok = None
    for part in blocked.split():
        if part.startswith("tok-"):
            tok = part.rstrip(".")
            break
    assert tok

    agent2 = _agent(tmp_path, approval_token_id=tok)
    agent2._approval_mgr = type(agent._approval_mgr)(store_path=store)
    agent2._gateway.approval_manager = agent2._approval_mgr
    # Same action + valid token → local policy allows (check_shell may still run)
    out = agent2._compass_gate("curl https://example.com", kind="SHELL")
    # curl is blocked by check_shell in safe mode — so we expect blocklist, not PAUSE
    assert out is None or "safe mode" in out or "network" in out.lower()
    if out:
        assert "PAUSE" not in out or "local policy PAUSE" not in out
