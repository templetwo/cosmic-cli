"""Integration: staged accept-pause is consumed on a fresh agent; Helix PAUSE
reaches confirm_pending from the board — not the local token store.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from cosmic_cli.agents import StargazerAgent
from cosmic_cli.bus import LocalMissionBus
from cosmic_cli.gateway import ApprovalManager
from cosmic_cli.pause_authority import accept_pause_cli, load_staged_token
from cosmic_cli.policy import ActionType, evaluate_rules
from cosmic_cli.rules import load_rules_from_markdown
from cosmic_cli.tui.app import PilotApp
from cosmic_cli.tui.state import apply_event


def _write_pause_rule(tmp_path: Path, pattern: str = "echo hi") -> None:
    (tmp_path / "COSMIC.md").write_text(
        f"""
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| echo-pause | PAUSE | SHELL | {pattern} |
"""
    )


def _agent(tmp_path: Path, store: Path, bus=None, **kwargs):
    kwargs.setdefault("write_echo", False)
    kwargs.setdefault("use_helix", False)
    kwargs.setdefault("session_id", "seat-retry")
    agent = StargazerAgent(
        "retry wiring",
        api_key="test",
        quiet=True,
        show_progress=False,
        work_dir=str(tmp_path),
        bus=bus or LocalMissionBus(),
        **kwargs,
    )
    mgr = ApprovalManager(store_path=store)
    agent._approval_mgr = mgr
    agent._gateway.approval_manager = mgr
    return agent


def test_accept_pause_then_fresh_do_session_consumes_once(tmp_path, monkeypatch):
    """do --session constructs a new StargazerAgent with no approval_token_id.

    Consume must happen via load_staged_token → claim_once on the retry gate,
    not a manual claim_once in the test.
    """
    monkeypatch.delenv("COSMIC_APPROVAL_TOKEN", raising=False)
    import cosmic_cli.pause_authority as pause_authority

    stage = tmp_path / "operator_approval_token"
    monkeypatch.setattr(pause_authority, "STAGE_PATH", stage)

    store = tmp_path / "approvals.json"
    _write_pause_rule(tmp_path)
    agent1 = _agent(tmp_path, store)
    blocked = agent1._compass_gate("echo hi", kind="SHELL")
    assert blocked and "PAUSE" in blocked

    rules = load_rules_from_markdown(tmp_path / "COSMIC.md")
    sha = evaluate_rules(rules, ActionType.SHELL, "echo hi").evaluated_input_sha256
    staged = accept_pause_cli(sha, manager=agent1._approval_mgr, stage_path=stage, require_tty=False)
    assert staged.outcome == "approved"
    assert stage.is_file()
    assert load_staged_token(stage) == staged.approval_token_id
    assert agent1._approval_mgr.peek_unused(sha) == staged.approval_token_id

    bus2 = LocalMissionBus()
    tape2 = []
    bus2.subscribe(tape2.append)
    agent2 = _agent(tmp_path, store, bus=bus2, session_id="seat-retry")
    assert agent2.approval_token_id == staged.approval_token_id
    assert agent2.session_id == "seat-retry"
    out2 = agent2._compass_gate("echo hi", kind="SHELL")
    assert out2 is None
    resolved = [e for e in tape2 if e.get("event") == "gate.pause_resolved"]
    assert len(resolved) == 1
    assert resolved[0]["decision"] == "approved"
    assert resolved[0]["by"] == "operator"
    assert resolved[0]["action_sha256"] == sha
    assert "token" not in resolved[0]
    assert agent2._approval_mgr.peek_unused(sha) is None

    bus3 = LocalMissionBus()
    tape3 = []
    bus3.subscribe(tape3.append)
    agent3 = _agent(tmp_path, store, bus=bus3, session_id="seat-retry")
    out3 = agent3._compass_gate("echo hi", kind="SHELL")
    assert out3 is not None and "BLOCKED" in out3
    assert not any(
        e.get("event") == "gate.pause_resolved" and e.get("by") == "operator"
        for e in tape3
    )


def test_helix_origin_board_approve_calls_confirm_pending(tmp_path, monkeypatch):
    monkeypatch.delenv("COSMIC_APPROVAL_TOKEN", raising=False)
    helix_token = "cafef00ddeadbeef"
    pending_id = 42
    confirms = []

    def fake_witness(**_kwargs):
        return {
            "ok": True,
            "result": {
                "classification": "PAUSE",
                "blocked": True,
                "pending_token": helix_token,
                "pending_id": pending_id,
                "reason": "needs confirmation",
                "action_summary": "echo hi",
            },
        }

    def fake_confirm(token):
        confirms.append(token)
        return {"ok": True, "result": {"ok": True}}

    import cosmic_cli.agents as agents_mod

    monkeypatch.setattr(agents_mod.helix_bridge, "witness", fake_witness)
    monkeypatch.setattr(agents_mod.helix_bridge, "confirm_pending", fake_confirm)
    monkeypatch.setattr(
        "cosmic_cli.helix_bridge.confirm_pending", fake_confirm
    )

    store = tmp_path / "approvals.json"
    bus = LocalMissionBus()
    tape = []
    bus.subscribe(tape.append)
    agent = _agent(tmp_path, store, bus=bus, use_helix=True)
    agent._human_pause_token = lambda *a, **k: None

    blocked = agent._compass_gate("echo hi", kind="SHELL")
    assert blocked and "Helix compass PAUSE" in blocked
    assert helix_token not in blocked
    minted = [e for e in tape if e["event"] == "gate.pause_minted"]
    assert len(minted) == 1
    rec = minted[0]
    assert rec["channel"] == "helix"
    assert rec["pending_id"] == pending_id
    assert helix_token not in json.dumps(rec)
    assert rec.get("action_sha256")
    assert agent._approval_mgr.unused_action_shas() == []

    from dataclasses import replace

    app = PilotApp(testing=True)
    app.agents_by_mission[agent.mission_id] = agent
    for event in tape:
        app.board = apply_event(app.board, event)
    app.board = replace(app.board, selected_key=agent.mission_id)
    pauses = app.board.pending_pauses
    assert len(pauses) == 1
    assert pauses[0].channel == "helix"
    handle = app._pause_handle(pauses[0])
    assert handle.channel == "helix"
    assert handle.pending_id == pending_id

    app._apply_pause_choice("approved", handle)
    assert confirms == [helix_token]
    assert agent._approval_mgr.unused_action_shas() == []
    resolved = [e for e in tape if e.get("event") == "gate.pause_resolved"]
    assert any(e.get("decision") == "approved" and e.get("by") == "operator" for e in resolved)
    blob = json.dumps(tape)
    assert helix_token not in blob
    assert "tok-" not in blob


def test_helix_board_approve_does_not_use_local_peek(tmp_path, monkeypatch):
    """Wrong store: a local unused token for some other sha must not be staged."""
    monkeypatch.delenv("COSMIC_APPROVAL_TOKEN", raising=False)
    helix_token = "aabbccddeeff0011"
    confirms = []

    def fake_witness(**_kwargs):
        return {
            "ok": True,
            "result": {
                "classification": "PAUSE",
                "blocked": True,
                "pending_token": helix_token,
                "pending_id": 7,
                "reason": "needs confirmation",
            },
        }

    import cosmic_cli.agents as agents_mod

    monkeypatch.setattr(agents_mod.helix_bridge, "witness", fake_witness)
    monkeypatch.setattr(
        "cosmic_cli.helix_bridge.confirm_pending",
        lambda tok: confirms.append(tok) or {"ok": True, "result": {"ok": True}},
    )

    store = tmp_path / "approvals.json"
    decoy = ApprovalManager(store_path=store)
    decoy_sha = "b" * 64
    decoy.mint_token(decoy_sha)

    bus = LocalMissionBus()
    agent = _agent(tmp_path, store, bus=bus, use_helix=True)
    agent._human_pause_token = lambda *a, **k: None
    agent._compass_gate("echo hi", kind="SHELL")

    app = PilotApp(testing=True)
    app.agents_by_mission[agent.mission_id] = agent
    from dataclasses import replace
    from cosmic_cli.tui.state import apply_event as _apply

    # Replay only pause_minted from this agent
    import json as _json

    rows = [
        _json.loads(ln)
        for ln in agent.session_path.read_text().splitlines()
        if ln.strip()
    ]
    for event in rows:
        app.board = _apply(app.board, event)
    app.board = replace(app.board, selected_key=agent.mission_id)
    handle = app._pause_handle(app.board.pending_pauses[0])
    app._apply_pause_choice("approved", handle)
    assert confirms == [helix_token]
    assert decoy.peek_unused(decoy_sha) is not None
