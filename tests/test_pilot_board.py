"""Pilot Board layout, bus subscribe, and selected-mission tape isolation."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from dataclasses import replace

from cosmic_cli import __version__, theme
from cosmic_cli.bus import LocalMissionBus
from cosmic_cli.events import make_event
from cosmic_cli.tui.format import identity_line
from cosmic_cli.ui import APIKeyScreen, DirectivesUI


def _evt(event: str, mission: str, seq: int, **payload):
    return make_event(
        event,
        session="S",
        mission=mission,
        seq=seq,
        ts="2026-09-19T21:03:00+00:00",
        **payload,
    )


def _start(mission: str, directive: str, seq: int = 0, **payload):
    payload.setdefault("model", "grok-4.5")
    payload.setdefault("exec_mode", "safe")
    payload.setdefault("max_steps", 30)
    payload.setdefault("root", "/tmp/work")
    payload.setdefault("helix", True)
    return _evt("mission.start", mission, seq, directive=directive, **payload)


def _step(mission: str, n: int, seq: int, action: str, path: str):
    raw = f"{action}: {path}"
    return _evt(
        "step.proposed",
        mission,
        seq,
        n=n,
        action=action,
        raw=raw,
        head=raw,
    )


def _run_pilot(app, size=(120, 40), body=None):
    async def _inner():
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            if body is not None:
                return await body(app, pilot)
            return None

    return asyncio.run(_inner())


def test_identity_unknown_floor_is_not_ok():
    line = identity_line(
        version="0.9.5",
        commit="de0a403",
        model="grok-4.5",
        helix_on=False,
        floor_ok=None,
    )
    assert "floor:ok" not in line
    assert "floor:unknown" in line
    assert theme.GOOD not in line.split("floor")[-1]


def test_two_mocked_agents_stay_isolated():
    ui = DirectivesUI(testing=True)
    with patch.dict(os.environ, {"XAI_API_KEY": "test_key"}), patch(
        "cosmic_cli.ui.StargazerAgent"
    ) as mock_agent_cls, patch.object(ui, "_refresh_panel"):
        a, b = Mock(), Mock()
        a.status = "ready"
        a.logs = []
        a.mission_id = "S__A"
        a._bus = LocalMissionBus()
        b.status = "ready"
        b.logs = []
        b.mission_id = "S__B"
        b._bus = LocalMissionBus()
        mock_agent_cls.side_effect = [a, b]
        ui.add_directive("alpha")
        ui.add_directive("beta")
        assert set(ui.agents) == {"alpha", "beta"}
        assert ui.agents_by_mission["S__A"] is a
        assert ui.agents_by_mission["S__B"] is b
        assert ui.board.missions["S__A"].directive == "alpha"
        assert ui.board.missions["S__B"].directive == "beta"


def test_subscribe_before_run():
    ui = DirectivesUI(testing=True)
    order = []

    class FakeBus:
        def subscribe(self, fn):
            order.append("subscribe")
            return lambda: None

    class FakeAgent:
        mission_id = "M1"
        status = "ready"
        logs = []
        model = "grok-4.5"
        max_steps = 30
        _bus = FakeBus()

        def run(self):
            order.append("run")

    with patch.dict(os.environ, {"XAI_API_KEY": "test_key"}), patch(
        "cosmic_cli.ui.StargazerAgent", return_value=FakeAgent()
    ), patch.object(ui, "_refresh_panel"):
        ui.add_directive("first")
    assert order == ["subscribe", "run"]


def test_layout_ids_at_acceptance_sizes():
    async def body(app, pilot):
        for wid in (
            "identity",
            "mission_table",
            "step_tape",
            "compass_pulse",
            "pending",
            "session_meta",
            "directive_input",
            "deploy_btn",
            "diff_peek",
            "diff_body",
        ):
            app.query_one(f"#{wid}")
        peek = app.query_one("#diff_peek")
        assert peek.has_class("-hidden")
        ident = str(app.query_one("#identity").render())
        assert __version__ in ident or __version__ in app._identity_text()
        assert "floor:ok" not in app._identity_text()
        assert "floor:unknown" in app._identity_text()
        table = app.query_one("#mission_table")
        assert [str(c.label) for c in table.columns.values()] == [
            "STATUS",
            "STEPS",
            "DIRECTIVE",
            "BASIS",
        ]

    _run_pilot(DirectivesUI(testing=True), size=(120, 40), body=body)
    _run_pilot(DirectivesUI(testing=True), size=(100, 30), body=body)


def test_selected_tape_isolation():
    async def body(app, pilot):
        app.apply_bus_event(_start("S__A", "alpha"))
        app.apply_bus_event(_start("S__B", "beta"))
        app.apply_bus_event(_step("S__A", 1, 1, "READ", "a.py"))
        app.apply_bus_event(_step("S__A", 2, 2, "READ", "a2.py"))
        app.apply_bus_event(_step("S__B", 1, 1, "EDIT", "b.py"))
        await pilot.pause()
        assert [s.kind for s in app.board.missions["S__A"].steps] == ["READ", "READ"]
        assert [s.kind for s in app.board.missions["S__B"].steps] == ["EDIT"]
        app.board = replace(app.board, selected_key="S__A")
        app._paint()
        await pilot.pause()
        tape = app.query_one("#step_tape")
        plain = "\n".join(getattr(line, "text", str(line)) for line in tape.lines)
        assert "READ" in plain
        assert "EDIT" not in plain
        app.board = replace(app.board, selected_key="S__B")
        app._paint()
        await pilot.pause()
        tape = app.query_one("#step_tape")
        plain = "\n".join(getattr(line, "text", str(line)) for line in tape.lines)
        assert "EDIT" in plain
        assert "READ" not in plain

    _run_pilot(DirectivesUI(testing=True), body=body)


def test_typing_q_does_not_quit():
    async def body(app, pilot):
        inp = app.query_one("#directive_input")
        inp.focus()
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert app.is_running
        assert inp.value == "q"
        peek = app.query_one("#diff_peek")
        await pilot.press("D")
        await pilot.pause()
        assert peek.has_class("-hidden")
        assert "D" in inp.value or "d" in inp.value

    _run_pilot(DirectivesUI(testing=True), body=body)


def test_d_toggles_diff_when_input_not_focused():
    async def body(app, pilot):
        table = app.query_one("#mission_table")
        table.focus()
        await pilot.pause()
        peek = app.query_one("#diff_peek")
        assert peek.has_class("-hidden")
        await pilot.press("D")
        await pilot.pause()
        assert not peek.has_class("-hidden")
        await pilot.press("D")
        await pilot.pause()
        assert peek.has_class("-hidden")

    _run_pilot(DirectivesUI(testing=True), body=body)


def test_ctrl_k_opens_api_key():
    async def body(app, pilot):
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(app.screen, APIKeyScreen)

    _run_pilot(DirectivesUI(testing=True), body=body)


def test_unmount_unsubscribes():
    ui = DirectivesUI(testing=True)
    hits = []

    class FakeBus:
        def subscribe(self, fn):
            return lambda: hits.append("unsub")

        def unsubscribe(self, fn):
            hits.append("unsubscribe")

    class FakeAgent:
        mission_id = "M1"
        status = "ready"
        logs = []
        model = "grok-4.5"
        max_steps = 30
        _bus = FakeBus()

        def run(self):
            return None

    with patch.dict(os.environ, {"XAI_API_KEY": "test_key"}), patch(
        "cosmic_cli.ui.StargazerAgent", return_value=FakeAgent()
    ), patch.object(ui, "_refresh_panel"):
        ui.add_directive("first")
    ui._detach_bus()
    assert "unsub" in hits
    assert "unsubscribe" in hits


def test_late_event_after_shutdown_does_not_crash():
    app = DirectivesUI(testing=True)

    async def body(app, pilot):
        app.apply_bus_event(_start("S__A", "alpha"))

    _run_pilot(app, body=body)
    app._on_bus_event(_step("S__A", 1, 1, "READ", "late.py"))
    app.apply_bus_event(_step("S__A", 1, 1, "READ", "late.py"))


@pytest.mark.parametrize("decision", ["approved", "declined"])
def test_operator_resolution_clears_worker_minted_pause(tmp_path, decision):
    from cosmic_cli.gateway import ApprovalManager
    from cosmic_cli.pause_authority import approve_pause

    bus = LocalMissionBus()
    manager = ApprovalManager(store_path=tmp_path / "approvals.json")
    sha = "a" * 64
    token = manager.mint_token(sha)
    resolutions = []

    def emit_resolution(choice, summary, action_sha256, **payload):
        resolutions.append(choice)
        bus.publish(_evt(
            "gate.pause_resolved", "S__A", 3, decision=choice,
            action_summary=summary, action_sha256=action_sha256, **payload,
        ))

    agent = SimpleNamespace(
        _bus=bus, _approval_mgr=manager, _emit_pause_resolved=emit_resolution,
    )

    async def body(app, pilot):
        app.agents_by_mission["S__A"] = agent
        app._subscribe_agent(agent)
        # Agent events arrive on a worker; the operator resolves on the UI thread.
        for event in (
            _start("S__A", "approval exercise"),
            _evt("gate.pause_minted", "S__A", 1, action_sha256=sha,
                 action_summary="printf exercise", channel="local"),
            _evt("mission.end", "S__A", 2, status="blocked"),
        ):
            await asyncio.to_thread(bus.publish, event)
        await pilot.pause()
        assert len(app.board.pending_pauses) == 1
        assert "printf exercise" in str(app.query_one("#pending").render())

        # Keep the real helper, but stage only inside this test's temporary store.
        with patch("cosmic_cli.tui.app.approve_pause", side_effect=lambda *a, **kw:
                   approve_pause(*a, stage_path=tmp_path / "stage", **kw)):
            app._decide_selected_pause(decision)
            await pilot.pause()
            assert app.board.pending_pauses == []
            assert "printf exercise" not in str(app.query_one("#pending").render())
            assert app.board.missions["S__A"].status == "blocked"
            app._decide_selected_pause(decision)
            assert resolutions == [decision]
        assert manager.claim_once(token, sha) is (decision == "approved")
        assert manager.claim_once(token, sha) is False

    _run_pilot(DirectivesUI(testing=True), body=body)
