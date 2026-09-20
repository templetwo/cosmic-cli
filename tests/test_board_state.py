"""Pure BoardState reducer: MissionBus events -> board, no I/O."""

from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from cosmic_cli.events import (
    FORBIDDEN_TOKEN_KEYS,
    make_compat_alias,
    make_event,
)
from cosmic_cli.tui import BoardState, Mission, apply_event
from cosmic_cli.tui.state import PendingPause


def _evt(event: str, mission: str = "S__T", seq: int = 0, **payload):
    return make_event(
        event,
        session="S",
        mission=mission,
        seq=seq,
        ts="2026-09-19T21:03:00+00:00",
        **payload,
    )


def _start(mission: str, seq: int = 0, **payload):
    payload.setdefault("directive", f"work {mission}")
    payload.setdefault("model", "grok-4.5")
    payload.setdefault("exec_mode", "safe")
    payload.setdefault("max_steps", 30)
    return _evt("mission.start", mission, seq, **payload)


def _step(mission: str, n: int, seq: int, action: str = "READ", path: str = "f.py"):
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


class TestIsolation:
    def test_two_missions_isolated(self):
        s0 = BoardState()
        s1 = apply_event(s0, _start("S__A", seq=0))
        s2 = apply_event(s1, _start("S__B", seq=0))
        assert s0.missions == {}
        assert s2.selected_key == "S__A"
        a_steps_before = s2.missions["S__A"].steps
        b_steps_before = s2.missions["S__B"].steps

        s3 = apply_event(s2, _step("S__A", n=1, seq=1))
        s4 = apply_event(s3, _step("S__A", n=2, seq=2))
        s5 = apply_event(s4, _step("S__B", n=1, seq=1))

        assert a_steps_before == []
        assert b_steps_before == []
        assert s2.missions["S__A"].steps is a_steps_before
        assert s2.missions["S__B"].steps is b_steps_before
        assert [st.kind for st in s5.missions["S__A"].steps] == ["READ", "READ"]
        assert [st.kind for st in s5.missions["S__B"].steps] == ["READ"]
        assert s5.missions["S__A"].steps_taken == 2
        assert s5.missions["S__B"].steps_taken == 1
        assert s5.missions["S__A"].steps is not s5.missions["S__B"].steps
        assert s5.selected_key == "S__A"

        compass = _evt(
            "compass.verdict",
            "S__A",
            seq=3,
            n=2,
            classification="PAUSE",
            tool_name="SHELL",
            action_summary="rm -rf build",
            rule_matched="destructive_rm",
        )
        s6 = apply_event(s5, compass)
        assert any(st.compass == "PAUSE" for st in s6.missions["S__A"].steps)
        assert all(st.compass is None for st in s6.missions["S__B"].steps)
        assert s6.compass_today.get("PAUSE") == 1
        assert s6.compass_total == {}


class TestFinish:
    def test_verified_iff_verifier_on_end(self):
        ready = apply_event(BoardState(), _start("S__T"))
        verified = apply_event(
            ready,
            _evt(
                "mission.end",
                seq=3,
                status="verified",
                finish_basis="verifier",
                steps=7,
                edited=[],
                warnings=[],
                model="grok-4.5",
                outcome="ok",
            ),
        )
        assert verified.missions["S__T"].status == "verified"
        assert verified.missions["S__T"].finish_basis == "verifier"

        declared = apply_event(
            ready,
            _evt(
                "finish.declared",
                seq=2,
                n=7,
                status="verified",
                finish_basis="verifier",
                synthesized=False,
                text="done",
            ),
        )
        assert declared.missions["S__T"].status == "verified"
        assert declared.missions["S__T"].finish_basis == "verifier"

        needs = apply_event(
            ready,
            _evt(
                "mission.end",
                seq=3,
                status="needs_review",
                finish_basis="model_declared",
                steps=1,
                edited=[],
                warnings=[],
                model="grok-4.5",
                outcome="x",
            ),
        )
        assert needs.missions["S__T"].status == "needs_review"
        assert needs.missions["S__T"].finish_basis == "model_declared"

        # Illegal pair: never paint verified without basis=verifier.
        fake = {
            "v": 1,
            "event": "mission.end",
            "ts": "2026-09-19T21:03:00+00:00",
            "session": "S",
            "mission": "S__T",
            "seq": 3,
            "status": "verified",
            "finish_basis": "model_declared",
        }
        coerced = apply_event(ready, fake)
        assert coerced.missions["S__T"].status != "verified"
        assert coerced.missions["S__T"].status == "needs_review"
        assert coerced.missions["S__T"].finish_basis == "model_declared"

        no_basis = {
            "v": 1,
            "event": "mission.end",
            "ts": "2026-09-19T21:03:00+00:00",
            "session": "S",
            "mission": "S__T",
            "seq": 3,
            "status": "verified",
        }
        refused = apply_event(ready, no_basis)
        assert refused.missions["S__T"].status != "verified"
        assert refused.missions["S__T"].finish_basis is None

    def test_blocked_has_no_basis(self):
        ready = apply_event(BoardState(), _start("S__T"))
        ended = apply_event(
            ready,
            _evt(
                "mission.end",
                seq=4,
                status="blocked",
                steps=3,
                edited=[],
                warnings=[],
                model="grok-4.5",
                outcome="[BLOCKED] x",
            ),
        )
        assert ended.missions["S__T"].status == "blocked"
        assert ended.missions["S__T"].finish_basis is None

        smuggled = {
            "v": 1,
            "event": "mission.end",
            "ts": "2026-09-19T21:03:00+00:00",
            "session": "S",
            "mission": "S__T",
            "seq": 4,
            "status": "blocked",
            "finish_basis": "model_declared",
        }
        dropped = apply_event(ready, smuggled)
        assert dropped.missions["S__T"].status == "blocked"
        assert dropped.missions["S__T"].finish_basis is None

    @pytest.mark.parametrize("status", ["max_steps", "passed", "error"])
    def test_non_finish_terminals_omit_basis(self, status):
        ready = apply_event(BoardState(), _start("S__T"))
        ended = apply_event(
            ready,
            _evt(
                "mission.end",
                seq=4,
                status=status,
                steps=1,
                edited=[],
                warnings=[],
                model="grok-4.5",
                outcome="x",
            ),
        )
        assert ended.missions["S__T"].status == status
        assert ended.missions["S__T"].finish_basis is None


class TestPause:
    def test_pause_minted_then_resolved(self):
        s = apply_event(BoardState(), _start("S__T"))
        minted = _evt(
            "gate.pause_minted",
            seq=9,
            action_summary="rm -rf build",
            action_sha256="c" * 64,
            pending_id=42,
            expires_at="2026-09-19T21:08:00+00:00",
        )
        before = list(s.pending_pauses)
        s1 = apply_event(s, minted)
        assert s.pending_pauses == before
        assert len(s1.pending_pauses) == 1
        pause = s1.pending_pauses[0]
        assert pause.pending_id == 42
        assert pause.action_sha256 == "c" * 64
        assert pause.mission_key == "S__T"
        assert pause.action_summary == "rm -rf build"

        resolved = _evt(
            "gate.pause_resolved",
            seq=10,
            decision="declined",
            by="operator",
            action_summary="rm -rf build",
            pending_id=42,
        )
        s2 = apply_event(s1, resolved)
        assert s2.pending_pauses == []
        assert len(s1.pending_pauses) == 1

        by_sha = apply_event(
            s1,
            _evt(
                "gate.pause_resolved",
                seq=11,
                decision="expired",
                action_summary="rm -rf build",
                action_sha256="c" * 64,
            ),
        )
        assert by_sha.pending_pauses == []

    def test_token_key_on_pause_event_is_not_copied(self):
        s = apply_event(BoardState(), _start("S__T"))
        raw = {
            "v": 1,
            "event": "gate.pause_minted",
            "ts": "2026-09-19T21:03:00+00:00",
            "session": "S",
            "mission": "S__T",
            "seq": 9,
            "action_summary": "rm -rf build",
            "action_sha256": "d" * 64,
            "pending_id": 7,
            "expires_at": "2026-09-19T21:08:00+00:00",
            "token": "tok-deadbeefdeadbeef",
            "token_id": "tok-aabbccddeeff0011",
            "approval_token": "secret",
            "token_id_prefix": "deadbeef",
        }
        s1 = apply_event(s, raw)
        assert len(s1.pending_pauses) == 1
        pause = s1.pending_pauses[0]
        assert isinstance(pause, PendingPause)
        names = {f.name for f in fields(pause)}
        assert names.isdisjoint(FORBIDDEN_TOKEN_KEYS)
        assert "token" not in names
        assert "token_id_prefix" not in names
        dumped = asdict(pause)
        for key in FORBIDDEN_TOKEN_KEYS:
            assert key not in dumped
            assert not hasattr(pause, key)
        assert "token" not in dumped
        assert "token_id_prefix" not in dumped
        assert pause.pending_id == 7
        assert pause.action_summary == "rm -rf build"


class TestCompat:
    def test_compat_alias_does_not_double_count_a_step(self):
        s = apply_event(BoardState(), _start("S__T"))
        canonical = _step("S__T", n=1, seq=2)
        alias = make_compat_alias(canonical)
        assert alias is not None
        s1 = apply_event(s, canonical)
        s2 = apply_event(s1, alias)
        assert s2 is s1
        assert len(s1.missions["S__T"].steps) == 1
        assert s1.missions["S__T"].steps_taken == 1

        # Dual-write the other way still counts once.
        s_alias_first = apply_event(s, alias)
        s_then_canon = apply_event(s_alias_first, canonical)
        assert s_alias_first is s
        assert len(s_then_canon.missions["S__T"].steps) == 1


class TestHealth:
    def test_unknown_floor_ok_stays_none(self):
        s = BoardState()
        assert s.floor_ok is None
        s = apply_event(s, _start("S__T"))
        s = apply_event(s, _step("S__T", n=1, seq=1))
        s = apply_event(
            s,
            _evt(
                "mission.status",
                seq=2,
                status="running",
            ),
        )
        assert s.floor_ok is None
        assert s.floor_ok is not True
        assert s.floor_ok is not False


class TestMutate:
    def test_fs_mutate_records_path_checkpoint_and_optional_diff(self):
        s = apply_event(BoardState(), _start("S__T"))
        s = apply_event(
            s,
            _evt(
                "fs.mutate",
                seq=6,
                n=2,
                op="EDIT",
                path="foo.py",
                checkpoint_id="ck_1",
                receipt_id="r1",
            ),
        )
        mission = s.missions["S__T"]
        assert mission.last_mutation_path == "foo.py"
        assert mission.last_checkpoint == "ck_1"
        assert mission.last_diff is None
        s = apply_event(
            s,
            _evt(
                "fs.mutate",
                seq=7,
                n=3,
                op="WRITE",
                path="foo.py",
                checkpoint_id="ck_2",
                diff="@@ -1 +1 @@\n",
            ),
        )
        assert s.missions["S__T"].last_diff == "@@ -1 +1 @@\n"
        assert s.missions["S__T"].last_checkpoint == "ck_2"


class TestUnknown:
    def test_unknown_event_is_noop(self):
        s = apply_event(BoardState(), _start("S__T"))
        out = apply_event(
            s,
            _evt("steer", seq=9, kind="repeat", message="loop"),
        )
        assert out is s
        assert isinstance(s.missions["S__T"], Mission)
