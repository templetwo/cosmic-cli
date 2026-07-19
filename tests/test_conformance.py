"""Conformance suite — the terminal reviewer for the single-use token invariant.

Milestone: TERMINAL REVIEWER SEATED. Exercises the FROZEN v0.8.5 avionics
(action_bind, gateway.ApprovalManager, gateway.ActionGateway, checkpoint).

Box 4a/4b were strict-xfail receipts that legally unfroze gateway.py; both
are now green (fcntl refuse + seal-after-verify). Authored by the code seat
2026-07-19; helm commit + box-4 fixes 2026-07-19.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from cosmic_cli import gateway as gateway_mod
from cosmic_cli.action_bind import action_sha256, bind_edit, bind_write
from cosmic_cli.checkpoint import CheckpointManager
from cosmic_cli.gateway import ActionGateway, ApprovalManager
from cosmic_cli.policy import ActionType, evaluate_rules

SHA = "a" * 64
WRONG = "b" * 64


def test_collision_edit_serialization():
    a = action_sha256(
        bind_edit(
            path="f",
            pre_content="ORIG\nMAL\nSAFE",
            old="ORIG\nMAL",
            new="SAFE",
            post_content="SAFE\nSAFE",
        )
    )
    b = action_sha256(
        bind_edit(
            path="f",
            pre_content="ORIG\nMAL\nSAFE",
            old="ORIG",
            new="MAL\nSAFE",
            post_content="ORIG\nMAL\nSAFE",
        )
    )
    assert a != b, "colliding EDIT pair must bind to different action_sha256"


def _claim(args):
    store, tok, sha = args
    return ApprovalManager(store_path=Path(store)).claim_once(tok, sha)


def test_exactly_once_concurrent(tmp_path):
    store = tmp_path / "approvals.json"
    tok = ApprovalManager(store_path=store).mint_token(SHA)
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_claim, [(str(store), tok, SHA)] * 8))
    assert sum(1 for r in results if r) == 1, f"exactly one claim must win: {results}"


def test_fcntl_absent_refuses(tmp_path, monkeypatch):
    """Box 4a: without fcntl, claim_once refuses (never races unlocked)."""
    store = tmp_path / "approvals.json"
    # Mint while fcntl is available
    tok = ApprovalManager(store_path=store).mint_token(SHA)
    # Then remove the lock primitive — claim must refuse, not unlock
    monkeypatch.setattr(gateway_mod, "fcntl", None)
    mgr = ApprovalManager(store_path=store)
    assert mgr.claim_once(tok, SHA) is False


def test_replay_denies(tmp_path):
    store = tmp_path / "approvals.json"
    mgr = ApprovalManager(store_path=store)
    tok = mgr.mint_token(SHA)
    assert mgr.claim_once(tok, SHA) is True
    assert mgr.claim_once(tok, SHA) is False
    assert ApprovalManager(store_path=store).claim_once(tok, SHA) is False


def test_mismatch_burns(tmp_path):
    store = tmp_path / "approvals.json"
    mgr = ApprovalManager(store_path=store)
    tok = mgr.mint_token(SHA)
    assert mgr.claim_once(tok, WRONG) is False
    assert mgr.claim_once(tok, SHA) is False


def test_post_exec_binding_rolls_back(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("PRE")
    gw = ActionGateway(evaluate_rules, checkpoint_manager=CheckpointManager(tmp_path))
    record = bind_write(path="t.txt", content="APPROVED")
    receipt = gw.authorize(
        ActionType.WRITE,
        record,
        [],
        intended_paths=[f],
        executor_name="test",
    )

    def bad_executor():
        f.write_text("EVIL-DIFFERENT")
        return "wrote"

    with pytest.raises(PermissionError):
        gw.execute_with_receipt(
            receipt,
            bad_executor,
            expected_content=b"APPROVED",
            verify_path=f,
        )
    assert f.read_text() == "PRE", "post-exec mismatch must roll back to the pre-image"


def test_quiescence_post_verify_write(tmp_path):
    """Box 4b: after verify+seal, a late write must not stick."""
    f = tmp_path / "t.txt"
    f.write_text("PRE")
    gw = ActionGateway(evaluate_rules, checkpoint_manager=CheckpointManager(tmp_path))
    record = bind_write(path="t.txt", content="APPROVED")
    receipt = gw.authorize(
        ActionType.WRITE,
        record,
        [],
        intended_paths=[f],
        executor_name="test",
    )

    def executor():
        f.write_text("APPROVED")
        return "wrote"

    gw.execute_with_receipt(
        receipt,
        executor,
        expected_content=b"APPROVED",
        verify_path=f,
    )
    # Seal drops write bits — late mutation raises; content stays approved.
    with pytest.raises(OSError):
        f.write_text("EVIL-LATE")
    assert f.read_text() == "APPROVED"


def test_sequential_mutation_survives_seal(tmp_path):
    """Write-then-revise: seal after #1 must not block authorized mutation #2.

    authorize() un-seals intended_paths; writers also chmod u+w. Late
    unauthorized writes still fail (test_quiescence_post_verify_write).
    """
    from cosmic_cli import tools

    f = tmp_path / "note.txt"
    f.write_text("v0")

    def mutate(content):
        gw = ActionGateway(evaluate_rules, checkpoint_manager=CheckpointManager(tmp_path))
        rec = gw.authorize(
            ActionType.WRITE,
            bind_write(path="note.txt", content=content),
            [],
            intended_paths=[f],
            executor_name="stargazer",
        )
        return gw.execute_with_receipt(
            rec,
            lambda: tools.tool_write(tmp_path, "note.txt", content),
            expected_content=content.encode(),
            verify_path=f,
        )

    mutate("first draft")
    mutate("revised draft")
    assert f.read_text() == "revised draft"
    # Still sealed after last authorized write — unauthorized late write fails
    with pytest.raises(OSError):
        f.write_text("EVIL-LATE")

