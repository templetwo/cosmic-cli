"""Local policy gateway wired into StargazerAgent (Claude live-fire fixes)."""

from pathlib import Path

from cosmic_cli.agents import StargazerAgent
from cosmic_cli.gateway import ApprovalManager, ApprovalStoreError
from cosmic_cli.policy import ActionType, PolicyValidationError, evaluate_rules
from cosmic_cli.rules import load_rules_from_markdown


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


def test_pause_token_not_burned_when_check_shell_blocks(tmp_path: Path):
    """Token-burn deadlock fix: check_shell must not consume PAUSE token."""
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
    agent._approval_mgr = ApprovalManager(store_path=store)
    agent._gateway.approval_manager = agent._approval_mgr

    # Capture human-only token from stderr ui path: mint by driving gate once
    blocked = agent._compass_gate("curl https://example.com", kind="SHELL")
    assert blocked and "PAUSE" in blocked
    # Model-visible reason must not carry the raw token (confused-deputy lock)
    assert "tok-" not in blocked
    assert "COSMIC_APPROVAL_TOKEN=" not in blocked

    rules = load_rules_from_markdown(tmp_path / "COSMIC.md")
    d = evaluate_rules(rules, ActionType.SHELL, "curl https://example.com")
    # Token was minted into store — recover any unused token for this action
    import json

    store_data = json.loads(store.read_text())
    tok = next(
        tid
        for tid, meta in store_data.items()
        if meta.get("action_sha256") == d.evaluated_input_sha256
        and not meta.get("used")
    )

    agent2 = _agent(tmp_path, approval_token_id=tok)
    agent2._approval_mgr = ApprovalManager(store_path=store)
    agent2._gateway.approval_manager = agent2._approval_mgr
    out = agent2._compass_gate("curl https://example.com", kind="SHELL")
    # safe mode blocks network after policy — token must still be unused
    assert out is not None
    assert "network" in out.lower() or "safe mode" in out or "BLOCKED" in out
    assert agent2._approval_mgr.validate(tok, d.evaluated_input_sha256)


def test_write_scope_rule_blocks_mutation(tmp_path: Path):
    """Mutation door must enforce WRITE-scoped rules (was silent dead law)."""
    (tmp_path / "COSMIC.md").write_text(
        """
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| no-secrets-write | WITNESS | WRITE,EDIT | secret_key |
"""
    )
    agent = _agent(tmp_path)
    out = agent._execute_step("CREATE: notes.txt|||secret_key=hunter2")
    assert "BLOCKED" in out
    assert "local policy" in out
    assert not (tmp_path / "notes.txt").exists()


def test_edit_scope_rule_blocks(tmp_path: Path):
    (tmp_path / "COSMIC.md").write_text(
        """
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| no-evil-edit | WITNESS | EDIT | evil |
"""
    )
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    agent = _agent(tmp_path)
    agent.files_seen.add("a.py")
    agent.files_read["a.py"] = "x = 1\n"
    out = agent._execute_step("EDIT: a.py|||x = 1|||x = evil")
    assert "BLOCKED" in out
    assert f.read_text() == "x = 1\n"


def test_bad_disposition_row_fails_loud(tmp_path: Path):
    md = tmp_path / "COSMIC.md"
    md.write_text(
        """
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| bad | BLOCK | SHELL | rm |
"""
    )
    try:
        load_rules_from_markdown(md)
        assert False, "expected PolicyValidationError"
    except PolicyValidationError as e:
        assert "BLOCK" in str(e) or "disposition" in str(e).lower()


def test_approval_persist_fail_closed(tmp_path: Path):
    """Store write failure must not silently leave replayable tokens."""
    store_as_dir = tmp_path / "approvals.json"
    store_as_dir.mkdir()
    mgr = ApprovalManager(store_path=store_as_dir)
    try:
        mgr.mint_token("abc" * 10)
        assert False, "expected ApprovalStoreError"
    except ApprovalStoreError:
        pass


def test_cosmic_md_self_edit_blocked(tmp_path: Path):
    (tmp_path / "COSMIC.md").write_text(
        "## Compass Rules\n\n| ID | Type | Scope | Pattern |\n"
        "|----|------|-------|---------|\n"
        "| x | OPEN | SHELL | echo |\n"
    )
    agent = _agent(tmp_path)
    agent.files_seen.add("COSMIC.md")
    agent.files_read["COSMIC.md"] = (tmp_path / "COSMIC.md").read_text()
    out = agent._execute_step("WRITE: COSMIC.md|||" + "# wiped\n")
    assert "BLOCKED" in out
    assert "COSMIC.md" in out
    assert "Compass Rules" in (tmp_path / "COSMIC.md").read_text()


def test_pause_without_approval_manager_refused(tmp_path: Path):
    from cosmic_cli.gateway import ActionGateway
    from cosmic_cli.policy import ActionType, evaluate_rules, validate_and_load_rules

    rules = validate_and_load_rules(
        [{"id": "net", "type": "PAUSE", "scope": "SHELL", "pattern": "curl"}]
    )
    gw = ActionGateway(evaluate_rules, approval_manager=None)
    try:
        gw.authorize(
            ActionType.SHELL,
            "curl x",
            rules,
            approval_token_id="tok-garbage",
        )
        assert False, "expected PermissionError"
    except PermissionError as e:
        assert "ApprovalManager" in str(e)
