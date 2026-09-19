"""Shared PAUSE helper: bind action_sha256, stage don't claim, no last-file selector."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from cosmic_cli.gateway import ApprovalManager
from cosmic_cli.pause_authority import (
    PauseHandle,
    accept_pause_cli,
    approve_pause,
    decline_pause,
)
from cosmic_cli.tui.screens.pause import PauseApproveScreen

SHA_A = "a" * 64
SHA_B = "b" * 64


def _mgr(tmp_path: Path) -> ApprovalManager:
    return ApprovalManager(store_path=tmp_path / "approvals.json")


def test_handle_has_no_token_fields():
    handle = PauseHandle(action_sha256=SHA_A, action_summary="curl x")
    dumped = asdict(handle)
    assert "token" not in dumped
    assert "approval_token_id" not in dumped
    assert not any("tok-" in str(v) for v in dumped.values())


def test_approve_stages_without_claiming(tmp_path: Path):
    mgr = _mgr(tmp_path)
    tok = mgr.mint_token(SHA_A)
    stage = tmp_path / "operator_approval_token"
    result = approve_pause(
        PauseHandle(action_sha256=SHA_A, action_summary="curl x"),
        manager=mgr,
        stage_path=stage,
        require_tty=False,
    )
    assert result.outcome == "approved"
    assert result.by == "operator"
    assert result.staged_for_gate is True
    assert result.approval_token_id == tok
    assert stage.read_text(encoding="utf-8").strip() == tok
    # Consume is the retry, not approve.
    assert mgr.peek_unused(SHA_A) == tok
    assert mgr.claim_once(tok, SHA_A) is True


def test_two_pending_cli_refuses_without_sha(tmp_path: Path):
    mgr = _mgr(tmp_path)
    mgr.mint_token(SHA_A)
    mgr.mint_token(SHA_B)
    stage = tmp_path / "stage"
    result = accept_pause_cli(
        "",
        manager=mgr,
        stage_path=stage,
        require_tty=False,
    )
    assert result.outcome == "ambiguous"
    assert not stage.exists()
    assert "pending PAUSE" in result.message


def test_cli_sha_selects_a_not_b(tmp_path: Path):
    mgr = _mgr(tmp_path)
    tok_a = mgr.mint_token(SHA_A)
    tok_b = mgr.mint_token(SHA_B)
    stage = tmp_path / "stage"
    result = accept_pause_cli(
        SHA_A,
        manager=mgr,
        stage_path=stage,
        require_tty=False,
    )
    assert result.outcome == "approved"
    assert result.handle.action_sha256 == SHA_A
    assert result.approval_token_id == tok_a
    assert stage.read_text(encoding="utf-8").strip() == tok_a
    assert tok_b not in stage.read_text()
    assert mgr.peek_unused(SHA_B) == tok_b


def test_single_pending_needs_no_query(tmp_path: Path):
    mgr = _mgr(tmp_path)
    tok = mgr.mint_token(SHA_A)
    stage = tmp_path / "stage"
    result = accept_pause_cli("", manager=mgr, stage_path=stage, require_tty=False)
    assert result.outcome == "approved"
    assert result.approval_token_id == tok


def test_decline_burns_without_operator_false_attribution_on_miss(tmp_path: Path):
    mgr = _mgr(tmp_path)
    tok = mgr.mint_token(SHA_A)
    result = decline_pause(
        PauseHandle(action_sha256=SHA_A, action_summary="curl x"),
        manager=mgr,
        require_tty=False,
    )
    assert result.outcome == "declined"
    assert result.by == "operator"
    assert mgr.peek_unused(SHA_A) is None
    assert mgr.claim_once(tok, SHA_A) is False
    miss = decline_pause(
        PauseHandle(action_sha256=SHA_B),
        manager=mgr,
        require_tty=False,
    )
    assert miss.outcome == "not_found"
    assert miss.by is None


def test_ranking_denied_without_tty(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("cosmic_cli.pause_authority.require_l2_tty", lambda _a: "L2-only")
    mgr = _mgr(tmp_path)
    mgr.mint_token(SHA_A)
    stage = tmp_path / "stage"
    result = approve_pause(
        PauseHandle(action_sha256=SHA_A),
        manager=mgr,
        stage_path=stage,
        require_tty=True,
    )
    assert result.outcome == "ranking_denied"
    assert not stage.exists()
    assert result.by is None


def test_modal_does_not_hold_a_token():
    handle = PauseHandle(
        action_sha256=SHA_A,
        action_summary="curl https://example.com",
        rule_id="outbound-net",
    )
    screen = PauseApproveScreen(handle)
    assert screen.handle is handle
    dumped = asdict(screen.handle)
    assert "token" not in dumped
    assert SHA_A == screen.handle.action_sha256


def test_modal_y_approves_and_does_not_render_tok():
    from textual.widgets import Static

    from cosmic_cli.ui import DirectivesUI
    from tests.test_pilot_board import _run_pilot

    handle = PauseHandle(
        action_sha256=SHA_A,
        action_summary="curl https://example.com",
        rule_id="outbound-net",
    )
    app = DirectivesUI(testing=True)
    seen: list = []

    async def body(app, pilot):
        app.push_screen(PauseApproveScreen(handle), seen.append)
        await pilot.pause()
        parts = []
        for node in app.screen.query(Static):
            rendered = node.render()
            parts.append(getattr(rendered, "plain", str(rendered)))
        blob = "\n".join(parts)
        assert "tok-" not in blob
        assert "token never shown" in blob
        await pilot.click("#pause_approve")
        await pilot.pause()

    _run_pilot(app, body=body)
    assert seen == ["approved"]
