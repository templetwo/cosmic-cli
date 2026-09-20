"""Gate identity and stage recovery, using only temporary operator stores."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmic_cli.gateway import ApprovalManager
from cosmic_cli.pause_authority import (
    PauseHandle, decline_pause, load_staged_token, retire_staged_token,
    write_stage_file,
)
from cosmic_cli.tui.app import PilotApp
from cosmic_cli.tui.screens.pause import PauseApproveScreen
from cosmic_cli.tui.state import BoardState, PendingPause, apply_event
from tests.test_pause_retry_wiring import _agent, _write_pause_rule
from tests.test_pilot_board import _run_pilot

SHA = "a" * 64


@pytest.fixture(autouse=True)
def isolated_stage(tmp_path, monkeypatch):
    from cosmic_cli import pause_authority
    from cosmic_cli.agents import StargazerAgent

    monkeypatch.delenv("COSMIC_APPROVAL_TOKEN", raising=False)
    stage = tmp_path / "stage"
    monkeypatch.setattr(pause_authority, "STAGE_PATH", stage)
    monkeypatch.setattr(StargazerAgent, "_human_pause_token", lambda *a, **kw: None)
    return stage


def test_helix_decline_preserves_local_token_and_pending_row(tmp_path):
    manager = ApprovalManager(store_path=tmp_path / "approvals.json")
    token = manager.mint_token(SHA)
    pause = PendingPause("echo hi", "M", pending_id=10,
                         action_sha256=SHA, channel="helix")
    app = PilotApp(testing=True)
    app.board = BoardState(pending_pauses=[pause], selected_key="M")
    emit = Mock()
    app.agents_by_mission["M"] = SimpleNamespace(
        _approval_mgr=manager, _emit_pause_resolved=emit,
    )
    notify = Mock()
    app.notify = notify
    app._decide_selected_pause("declined")
    assert app.board.pending_pauses == [pause]
    assert manager.peek_unused(SHA) == token
    emit.assert_not_called()
    assert "Helix decline is unavailable" in notify.call_args.args[0]
    result = decline_pause(app._pause_handle(pause), manager=manager, require_tty=False)
    assert result.outcome == "unsupported"
    assert result.by is None


def test_helix_modal_disables_decline_and_keeps_escape():
    app = PilotApp(testing=True)
    choices = []

    async def body(app, pilot):
        screen = PauseApproveScreen(PauseHandle(action_sha256=SHA, channel="helix"))
        app.push_screen(screen, choices.append)
        await pilot.pause()
        assert screen.query_one("#pause_decline").disabled
        await pilot.press("n")
        await pilot.pause()
        assert app.screen is screen
        assert choices == []
        await pilot.press("escape")
        await pilot.pause()
        assert choices == [None]

    _run_pilot(app, body=body)


def test_gate_ids_override_identical_hashes_in_reducer_and_modal_selection():
    first = PendingPause("echo hi", "M", pending_id=10,
                         action_sha256=SHA, channel="helix")
    second = replace(first, pending_id=20)
    other_mission = replace(second, mission_key="N")
    local = replace(second, pending_id=None, channel="local")
    state = BoardState(pending_pauses=[first, second, other_mission, local])
    event = {"event": "gate.pause_resolved", "mission": "M", "pending_id": 20,
             "action_sha256": SHA, "channel": "helix", "decision": "approved"}
    resolved = apply_event(state, event)
    assert resolved.pending_pauses == [first, other_mission, local]
    app = PilotApp(testing=True)
    app.board = state
    assert app._pause_matching_event(event) == second
    assert app._pause_matching_event(dict(event, pending_id=99)) is None
    assert apply_event(state, dict(event, pending_id=99)).pending_pauses == state.pending_pauses
    local_event = dict(event, channel="local", pending_id=None)
    assert apply_event(state, local_event).pending_pauses == [first, second, other_mission]


def test_helix_token_lookup_does_not_fall_back_from_unknown_id(tmp_path):
    agent = _agent(tmp_path, tmp_path / "approvals.json")
    agent._remember_helix_pause("test-only", pending_id=10, action_sha256=SHA)
    assert agent.helix_pause_token(pending_id=99, action_sha256=SHA) is None
    assert agent.helix_pause_token(pending_id=10, action_sha256=SHA) == "test-only"


def test_retirement_preserves_newer_stage_and_private_permissions(tmp_path):
    stage = tmp_path / "stage"
    write_stage_file("test-old", stage)
    write_stage_file("test-new", stage)
    assert not retire_staged_token("test-old", stage)
    assert load_staged_token(stage) == "test-new"
    assert stage.stat().st_mode & 0o777 == 0o600
    assert retire_staged_token("test-new", stage)
    assert not stage.exists()


def _staged_agent(tmp_path, stage, *, expired=False):
    import hashlib

    _write_pause_rule(tmp_path)
    store = tmp_path / "approvals.json"
    manager = ApprovalManager(store_path=store)
    sha = hashlib.sha256(b"echo hi").hexdigest()
    token = manager.mint_token(sha, ttl_seconds=-1 if expired else 300)
    write_stage_file(token, stage)
    return _agent(tmp_path, store), manager, token, store


def test_success_retires_stage_but_explicit_replay_stays_blocked(tmp_path, isolated_stage):
    agent, manager, token, store = _staged_agent(tmp_path, isolated_stage)
    assert agent._compass_gate("echo hi") is None
    assert not isolated_stage.exists()
    assert manager.token_is_spent(token)
    replay = _agent(tmp_path, store, approval_token_id=token)
    tape = []
    replay._bus.subscribe(tape.append)
    assert "BLOCKED" in replay._compass_gate("echo hi")
    assert not any(e["event"] == "gate.pause_minted" for e in tape)
    fresh = _agent(tmp_path, store)
    assert fresh.approval_token_id is None
    assert "Human approval required" in fresh._compass_gate("echo hi")


@pytest.mark.parametrize("spent", ["expired", "used"])
def test_stale_stage_blocks_current_attempt_then_allows_fresh_request(
    tmp_path, isolated_stage, spent,
):
    agent, manager, token, store = _staged_agent(
        tmp_path, isolated_stage, expired=spent == "expired",
    )
    if spent == "used":
        import hashlib
        assert manager.claim_once(token, hashlib.sha256(b"echo hi").hexdigest())
    tape = []
    agent._bus.subscribe(tape.append)
    blocked = agent._compass_gate("echo hi")
    assert "BLOCKED" in blocked and "Stale staged approval cleared" in blocked
    assert not isolated_stage.exists()
    assert not any(e["event"] == "gate.pause_minted" for e in tape)
    fresh = _agent(tmp_path, store)
    assert "Human approval required" in fresh._compass_gate("echo hi")


def test_wrong_action_does_not_retire_valid_stage(tmp_path, isolated_stage):
    agent, manager, token, _ = _staged_agent(tmp_path, isolated_stage)
    _write_pause_rule(tmp_path, pattern="echo")
    assert "BLOCKED" in agent._compass_gate("echo other")
    assert load_staged_token(isolated_stage) == token
    assert not manager.token_is_spent(token)
    assert agent._compass_gate("echo hi") is None


def test_retirement_does_not_remove_concurrently_replaced_approval(tmp_path, isolated_stage):
    agent, manager, token, _ = _staged_agent(tmp_path, isolated_stage)
    replacement = manager.mint_token("b" * 64)
    write_stage_file(replacement, isolated_stage)
    assert agent._compass_gate("echo hi") is None
    assert load_staged_token(isolated_stage) == replacement
    assert manager.token_is_spent(token)


def test_unknown_token_is_not_assumed_spent(tmp_path):
    manager = ApprovalManager(store_path=tmp_path / "approvals.json")
    assert not manager.token_is_spent("unknown")


@pytest.mark.parametrize("expired", [False, True])
def test_mutation_retires_stage_without_replaying_or_reminting(tmp_path, isolated_stage, expired):
    import hashlib
    from cosmic_cli.action_bind import bind_write

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "COSMIC.md").write_text(
        "## Compass Rules\n\n| ID | Type | Scope | Pattern |\n"
        "|----|------|-------|---------|\n| write-pause | PAUSE | WRITE | panel.txt |\n"
    )
    store = tmp_path / "approvals.json"
    manager = ApprovalManager(store_path=store)
    binding = bind_write(path="panel.txt", content="green")
    sha = hashlib.sha256(binding.encode()).hexdigest()
    token = manager.mint_token(sha, ttl_seconds=-1 if expired else 300)
    write_stage_file(token, isolated_stage)
    agent = _agent(workspace, store)
    tape = []
    agent._bus.subscribe(tape.append)
    result = agent._execute_step("WRITE: panel.txt|||green")
    assert not isolated_stage.exists()
    assert not any(e["event"] == "gate.pause_minted" for e in tape)
    if expired:
        assert "BLOCKED" in result and "Stale staged approval cleared" in result
        assert not (workspace / "panel.txt").exists()
    else:
        assert (workspace / "panel.txt").read_text() == "green", result
        assert manager.token_is_spent(token)
        # Retaining the token in this agent must not silently authorize or mint again.
        agent.files_seen.add("panel.txt")
        assert "BLOCKED" in agent._execute_step("WRITE: panel.txt|||green")
        assert not any(e["event"] == "gate.pause_minted" for e in tape)


def test_stage_writer_waits_for_retirement_compare(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from cosmic_cli import pause_authority

    stage = tmp_path / "stage"
    write_stage_file("test-old", stage)
    comparing, release = Event(), Event()
    original_load = pause_authority.load_staged_token

    def paused_load(path):
        value = original_load(path)
        comparing.set()
        assert release.wait(5)
        return value

    monkeypatch.setattr(pause_authority, "load_staged_token", paused_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        retired = pool.submit(retire_staged_token, "test-old", stage)
        try:
            assert comparing.wait(5)
            written = pool.submit(write_stage_file, "test-new", stage)
        finally:
            release.set()
        assert retired.result(timeout=5)
        written.result(timeout=5)
    assert original_load(stage) == "test-new"
