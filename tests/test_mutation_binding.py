"""Mutation binding: full-content hash + post-exec verify (Claude v3 bugs)."""

from pathlib import Path

from cosmic_cli.agents import StargazerAgent
from cosmic_cli.gateway import ActionGateway, ApprovalManager
from cosmic_cli.policy import ActionType, Disposition, PolicyDecision, evaluate_rules


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


def test_action_input_hashes_full_content_not_prefix(tmp_path: Path):
    """Token bound to benign header must NOT validate for different tail."""
    store = tmp_path / "approvals.json"
    header = "x" * 800
    benign = header + "\nSAFE_TAIL"
    evil = header + "\nEVIL_SECRET_PAYLOAD"

    (tmp_path / "COSMIC.md").write_text(
        """
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| pause-write | PAUSE | WRITE | xxxx |
"""
    )
    # Force PAUSE on anything with many x's — both payloads match
    agent = _agent(tmp_path)
    agent._approval_mgr = ApprovalManager(store_path=store)
    agent._gateway.approval_manager = agent._approval_mgr

    # Mint against full benign action_input
    rules = agent._load_policy_rules()
    d_benign = evaluate_rules(
        rules, ActionType.WRITE, f"WRITE notes.txt\n{benign}"
    )
    assert d_benign.disposition == Disposition.PAUSE
    tok = agent._approval_mgr.mint_token(d_benign.evaluated_input_sha256)

    d_evil = evaluate_rules(rules, ActionType.WRITE, f"WRITE notes.txt\n{evil}")
    # Different full content → different sha → token must not validate
    assert d_benign.evaluated_input_sha256 != d_evil.evaluated_input_sha256
    assert not agent._approval_mgr.validate(tok, d_evil.evaluated_input_sha256)
    assert agent._approval_mgr.validate(tok, d_benign.evaluated_input_sha256)


def test_post_exec_content_mismatch_rolls_back(tmp_path: Path):
    """If executor writes different bytes than approved, refuse + restore."""
    target = tmp_path / "f.txt"
    target.write_text("original")

    def open_eval(rules, action_type, action_input):
        from cosmic_cli.policy import _compute_sha256

        return PolicyDecision(
            disposition=Disposition.OPEN,
            matches=(),
            default_used=True,
            policy_sha256="p",
            evaluated_input_sha256=_compute_sha256(action_input.encode()),
        )

    from cosmic_cli.checkpoint import CheckpointManager

    mgr = CheckpointManager(tmp_path)
    gw = ActionGateway(open_eval, checkpoint_manager=mgr)
    receipt = gw.authorize(
        ActionType.WRITE,
        "WRITE f.txt\napproved-body",
        [],
        intended_paths=[target],
        executor_name="t",
    )

    def evil_writer():
        target.write_text("NOT-THE-APPROVED-BODY")
        return "ok", True

    try:
        gw.execute_with_receipt(
            receipt,
            evil_writer,
            expected_content=b"approved-body",
            verify_path=target,
        )
        assert False, "expected PermissionError"
    except PermissionError as e:
        assert "mismatch" in str(e).lower()

    # Rolled back to pre-image
    assert target.read_text() == "original"


def test_pause_reason_has_no_token(tmp_path: Path):
    store = tmp_path / "a.json"
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
    out = agent._compass_gate("curl https://example.com", kind="SHELL")
    assert out and "BLOCKED" in out and "PAUSE" in out
    assert "tok-" not in out
    assert "COSMIC_APPROVAL_TOKEN=" not in out
    assert "helix confirm" not in out
