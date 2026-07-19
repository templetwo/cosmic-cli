"""Mutation binding: full-content hash, canonical EDIT, claim_once, token channel."""

import json
import hashlib
from pathlib import Path

from cosmic_cli.action_bind import bind_edit, bind_write
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


def test_edit_binding_no_newline_collision():
    """Two different EDITS must not hash-collide via newline-joined fields."""
    # Path that embeds a newline-shaped forgery vs separate path+old
    a = bind_edit(
        path="a",
        pre_content="bx",
        old="b",
        new="x",
        post_content="xx",
    )
    b = bind_edit(
        path="a\nb",
        pre_content="x",
        old="",
        new="x",
        post_content="x",
    )
    assert a != b
    ha = hashlib.sha256(a.encode()).hexdigest()
    hb = hashlib.sha256(b.encode()).hexdigest()
    assert ha != hb


def test_write_binding_differs_when_tail_differs():
    header = "x" * 800
    b1 = bind_write(path="notes.txt", content=header + "\nSAFE")
    b2 = bind_write(path="notes.txt", content=header + "\nEVIL")
    assert b1 != b2
    assert hashlib.sha256(b1.encode()).hexdigest() != hashlib.sha256(
        b2.encode()
    ).hexdigest()


def test_claim_once_exactly_once(tmp_path: Path):
    """Exactly-once invariant: second claim fails under same store."""
    store = tmp_path / "approvals.json"
    mgr = ApprovalManager(store_path=store)
    sha = "a" * 64
    tok = mgr.mint_token(sha)
    assert mgr.claim_once(tok, sha) is True
    assert mgr.claim_once(tok, sha) is False
    assert mgr.validate(tok, sha) is False


def test_claim_once_wrong_action_burns(tmp_path: Path):
    store = tmp_path / "approvals.json"
    mgr = ApprovalManager(store_path=store)
    sha = "b" * 64
    tok = mgr.mint_token(sha)
    assert mgr.claim_once(tok, "c" * 64) is False
    # Burned — cannot use for correct action either
    assert mgr.claim_once(tok, sha) is False


def test_post_exec_content_mismatch_rolls_back(tmp_path: Path):
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
        bind_write(path="f.txt", content="approved-body"),
        [],
        intended_paths=[target],
        executor_name="t",
        match_input="WRITE f.txt\napproved-body",
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

    assert target.read_text() == "original"


def test_pause_reason_and_channels_have_no_token(tmp_path: Path, monkeypatch):
    store = tmp_path / "a.json"
    token_file = tmp_path / "last_pause_token.json"
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

    # Point home token file at tmp
    monkeypatch.setattr(
        Path,
        "home",
        classmethod(lambda cls: tmp_path),
    )
    # Path.home is not easily patchable that way for already-imported code —
    # instead patch agent method to write into tmp_path
    seen = {}

    def capture_human(tok, *, channel, action_sha=""):
        seen["tok"] = tok
        token_file.write_text(json.dumps({"token": tok, "channel": channel}))

    agent._human_pause_token = capture_human  # type: ignore

    out = agent._compass_gate("curl https://example.com", kind="SHELL")
    assert out and "BLOCKED" in out and "PAUSE" in out
    assert "tok-" not in out
    assert "COSMIC_APPROVAL_TOKEN=" not in out
    assert "helix confirm" not in out
    assert seen.get("tok", "").startswith("tok-")
