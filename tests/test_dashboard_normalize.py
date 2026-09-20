"""Dashboard readers accept MissionBus v1 and legacy session/echo tapes.

Isolated tmp fixtures only. Does not read ~/.cosmic_echo.jsonl, live
sessions, or chronicle.db. Does not call state() against a live DB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HOME = Path.home()
LIVE_ECHO = HOME / ".cosmic_echo.jsonl"
LIVE_SESSIONS = HOME / ".cosmic-cli" / "sessions"
LIVE_DB = (
    HOME / ".claude/plugins/data/t2helix-templetwo-t2helix/chronicle.db"
)


@pytest.fixture(autouse=True)
def dashboard_mod(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dashboard"])
    from cosmic_cli import dashboard

    echo = tmp_path / "echo.jsonl"
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(dashboard, "ECHO", echo)
    monkeypatch.setattr(dashboard, "SESSIONS", sessions)
    monkeypatch.setattr(dashboard, "DB", tmp_path / "no-chronicle.db")
    return dashboard


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _legacy_tape():
    return [
        {
            "event": "start",
            "directive": "fix tests",
            "model": "m",
            "ts": "t0",
            "session": "OLD",
        },
        {"event": "step", "n": 1, "action": "READ: tests/test_x.py"},
        {"event": "step", "n": 2, "action": "EDIT: foo.py|||a|||b"},
        {"event": "end", "status": "complete", "steps": 2, "model": "m"},
    ]


def _v1_dual_write():
    start = {
        "v": 1,
        "event": "mission.start",
        "ts": "t0",
        "session": "S",
        "mission": "S__T",
        "seq": 0,
        "directive": "fix tests",
        "model": "grok-4.5",
    }
    start_alias = dict(start, event="start", compat=True, alias_of=0)
    status = {
        "v": 1,
        "event": "mission.status",
        "ts": "t1",
        "session": "S",
        "mission": "S__T",
        "seq": 1,
        "status": "running",
    }
    step = {
        "v": 1,
        "event": "step.proposed",
        "ts": "t2",
        "session": "S",
        "mission": "S__T",
        "seq": 2,
        "n": 1,
        "action": "READ",
        "raw": "READ: a.py",
        "head": "READ: a.py",
    }
    step_alias = dict(
        step, event="step", compat=True, alias_of=2, action="READ: a.py"
    )
    step2 = {
        "v": 1,
        "event": "step.proposed",
        "ts": "t3",
        "session": "S",
        "mission": "S__T",
        "seq": 3,
        "n": 2,
        "action": "EDIT",
        "raw": "EDIT: b.py|||old|||new",
        "head": "EDIT: b.py|||old|||new",
    }
    step2_alias = dict(
        step2,
        event="step",
        compat=True,
        alias_of=3,
        action="EDIT: b.py|||old|||new",
    )
    declared = {
        "v": 1,
        "event": "finish.declared",
        "ts": "t4",
        "session": "S",
        "mission": "S__T",
        "seq": 4,
        "n": 2,
        "status": "needs_review",
        "finish_basis": "model_declared",
        "synthesized": False,
        "text": "done",
    }
    end = {
        "v": 1,
        "event": "mission.end",
        "ts": "t5",
        "session": "S",
        "mission": "S__T",
        "seq": 5,
        "status": "needs_review",
        "finish_basis": "model_declared",
        "steps": 2,
        "edited": ["b.py"],
        "warnings": [],
        "model": "grok-4.5",
        "outcome": "done",
    }
    end_alias = dict(end, event="end", compat=True, alias_of=5)
    return [
        start,
        start_alias,
        status,
        step,
        step_alias,
        step2,
        step2_alias,
        declared,
        end,
        end_alias,
    ]


def test_isolated_from_live_stores(dashboard_mod):
    assert dashboard_mod.ECHO != LIVE_ECHO
    assert LIVE_ECHO not in dashboard_mod.ECHO.parents
    assert dashboard_mod.SESSIONS != LIVE_SESSIONS
    assert LIVE_SESSIONS not in dashboard_mod.SESSIONS.parents
    assert dashboard_mod.DB != LIVE_DB


def test_v1_dual_write_one_row_per_step(dashboard_mod):
    rows = dashboard_mod.session_step_rows(_v1_dual_write())
    heads = [r["head"] for r in rows]
    assert heads == ["READ: a.py", "EDIT: b.py|||old|||new"]
    assert all(r["event"] == "step.proposed" for r in rows)
    assert all("finish_basis" not in r for r in rows)


@pytest.mark.parametrize("decision", ["approved", "declined"])
def test_gate_resolution_after_end_preserves_terminal(dashboard_mod, decision):
    tape = _v1_dual_write()
    tape = [e for e in tape if e["event"] != "finish.declared"]
    for event in tape:
        if event["event"] in ("mission.end", "end"):
            event["status"] = "blocked"
            event.pop("finish_basis", None)
    before = dashboard_mod.session_terminal(tape)
    steps = dashboard_mod.session_step_rows(tape)
    tape.append({
        "v": 1, "event": "gate.pause_resolved", "ts": "t6",
        "session": "S", "mission": "S__T", "seq": 6,
        "action_sha256": "a" * 64, "decision": decision, "by": "operator",
    })
    assert dashboard_mod.session_terminal(tape) == before
    assert before["status"] == "blocked"
    assert "finish_basis" not in before
    assert dashboard_mod.session_step_rows(tape) == steps


def test_dual_written_step_proposed_and_compat_step_is_one(dashboard_mod):
    rows = dashboard_mod.session_step_rows(
        [
            {
                "event": "step.proposed",
                "seq": 2,
                "mission": "S__T",
                "head": "READ: a.py",
                "action": "READ",
            },
            {
                "event": "step",
                "compat": True,
                "alias_of": 2,
                "seq": 2,
                "mission": "S__T",
                "action": "READ: a.py",
            },
        ]
    )
    assert len(rows) == 1
    assert rows[0]["head"] == "READ: a.py"


def test_legacy_without_v_readable(dashboard_mod):
    tape = _legacy_tape()
    assert all("v" not in rec for rec in tape)
    rows = dashboard_mod.session_step_rows(tape)
    assert [r["head"] for r in rows] == [
        "READ: tests/test_x.py",
        "EDIT: foo.py|||a|||b",
    ]
    term = dashboard_mod.session_terminal(tape)
    assert term == {"status": "complete"}
    assert "finish_basis" not in term


def test_prefer_head_else_legacy_action(dashboard_mod):
    with_head = dashboard_mod.session_step_rows(
        [
            {
                "event": "step.proposed",
                "head": "READ: a.py",
                "action": "READ",
            }
        ]
    )
    assert with_head[0]["head"] == "READ: a.py"
    legacy = dashboard_mod.session_step_rows(
        [{"event": "step", "n": 1, "action": "READ: x.py"}]
    )
    assert legacy[0]["head"] == "READ: x.py"
    verb_only = dashboard_mod.session_step_rows(
        [{"event": "step.proposed", "action": "READ"}]
    )
    assert verb_only[0]["head"] == "READ"


def test_lifecycle_events_not_on_step_tape(dashboard_mod):
    rows = dashboard_mod.session_step_rows(_v1_dual_write())
    names = {r["event"] for r in rows}
    assert names == {"step.proposed"}
    forbidden = {
        "start",
        "end",
        "mission.start",
        "mission.end",
        "mission.status",
        "finish.declared",
    }
    assert not names & forbidden


def test_session_terminal_reads_basis_from_end_or_mission_end(dashboard_mod):
    v1 = dashboard_mod.session_terminal(_v1_dual_write())
    assert v1 == {
        "status": "needs_review",
        "finish_basis": "model_declared",
    }
    legacy_named_end = dashboard_mod.session_terminal(
        [
            {
                "event": "end",
                "status": "verified",
                "finish_basis": "verifier",
            }
        ]
    )
    assert legacy_named_end == {
        "status": "verified",
        "finish_basis": "verifier",
    }


def test_does_not_invent_finish_basis(dashboard_mod):
    blocked = dashboard_mod.session_terminal(
        [{"event": "end", "status": "blocked", "steps": 1}]
    )
    assert blocked == {"status": "blocked"}
    assert "finish_basis" not in blocked
    needs = dashboard_mod.session_terminal(
        [{"event": "mission.end", "status": "needs_review"}]
    )
    assert needs == {"status": "needs_review"}
    assert "finish_basis" not in needs


def test_session_steps_mixed_dir_no_duplicates(dashboard_mod, tmp_path):
    sessions = dashboard_mod.SESSIONS
    _write_jsonl(sessions / "OLD__legacy.jsonl", _legacy_tape())
    _write_jsonl(
        sessions / "S__T.jsonl",
        _v1_dual_write(),
    )
    steps = dashboard_mod.session_steps(n_files=3, n_steps=20)
    heads = [s["head"] for s in steps]
    assert heads.count("READ: a.py") == 1
    assert heads.count("EDIT: b.py|||old|||new") == 1
    assert heads.count("READ: tests/test_x.py") == 1
    assert heads.count("EDIT: foo.py|||a|||b") == 1
    assert len(heads) == 4
    stamps = {s["_file"] for s in steps}
    assert stamps == {"OLD", "S"}


def test_file_stamp_from_filename(dashboard_mod):
    stem = "20260919T210300Z__20260919T210300123456Z_ab.jsonl"
    assert dashboard_mod.session_file_stamp(stem) == "20260919"


def test_mission_counts_keep_complete_separate(dashboard_mod):
    _write_jsonl(
        dashboard_mod.ECHO,
        [
            {"status": "verified", "finish_basis": "verifier"},
            {"status": "needs_review", "finish_basis": "model_declared"},
            {"status": "needs_review", "finish_basis": "synthesized"},
            {"status": "complete"},
            {"status": "blocked"},
        ],
    )
    counts = dashboard_mod.mission_counts()
    assert counts == {
        "missions_verified": 1,
        "missions_needs_review": 2,
        "missions_complete": 1,
        "missions_blocked": 1,
    }


def test_old_echo_without_v_remains_readable(dashboard_mod):
    _write_jsonl(
        dashboard_mod.ECHO,
        [{"status": "complete", "directive": "old", "model": "m", "steps": 3}],
    )
    counts = dashboard_mod.mission_counts()
    assert counts["missions_complete"] == 1
    assert counts["missions_verified"] == 0
