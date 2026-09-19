"""MissionBus v1 envelope, finish combinations, redaction, subscriber isolation."""

from __future__ import annotations

import logging

import pytest

from cosmic_cli.agents import FINISHED_STATUSES
from cosmic_cli.bus import LocalMissionBus
from cosmic_cli.events import (
    EventValidationError,
    SCHEMA_VERSION,
    action_verb,
    apply_redaction,
    finish_basis_for_status,
    iter_canonical,
    make_compat_alias,
    make_event,
    normalize_legacy,
    unique_mission_stem,
    validate_event,
)


def _env(**payload):
    return make_event(
        payload.pop("event", "mission.status"),
        session="S",
        mission="S__T",
        seq=payload.pop("seq", 0),
        ts="2026-09-19T21:03:00+00:00",
        **payload,
    )


class TestEnvelope:
    def test_required_fields(self):
        rec = _env(event="mission.start", directive="fix tests", status="running")
        assert rec["v"] == SCHEMA_VERSION
        for key in ("v", "event", "ts", "session", "mission", "seq"):
            assert key in rec

    def test_payload_cannot_override_envelope(self):
        with pytest.raises(EventValidationError, match="envelope"):
            make_event(
                "mission.status",
                session="S",
                mission="S__T",
                seq=0,
                v=2,
            )

    def test_missing_envelope_field_rejected(self):
        rec = _env(status="running")
        del rec["mission"]
        with pytest.raises(EventValidationError, match="mission"):
            validate_event(rec)

    def test_seq_must_be_non_negative_int(self):
        rec = _env(status="running")
        rec["seq"] = -1
        with pytest.raises(EventValidationError, match="seq"):
            validate_event(rec)


class TestFinishCombinations:
    def test_verified_iff_verifier(self):
        rec = make_event(
            "mission.end",
            session="S",
            mission="S__T",
            seq=3,
            status="verified",
            finish_basis="verifier",
            steps=7,
            edited=[],
            warnings=[],
            model="grok-4.5",
            outcome="ok",
        )
        assert rec["status"] == "verified"
        assert rec["finish_basis"] == "verifier"
        with pytest.raises(EventValidationError):
            finish_basis_for_status("verified", "model_declared")

    def test_needs_review_bases(self):
        for basis in ("model_declared", "synthesized", "verifier_blocked"):
            finish_basis_for_status("needs_review", basis)
        with pytest.raises(EventValidationError):
            finish_basis_for_status("needs_review", "verifier")

    def test_blocked_and_max_steps_omit_basis(self):
        for status in ("blocked", "max_steps", "passed", "error"):
            rec = make_event(
                "mission.end",
                session="S",
                mission="S__T",
                seq=4,
                status=status,
                steps=1,
                edited=[],
                warnings=[],
                model="grok-4.5",
                outcome="x",
            )
            assert "finish_basis" not in rec
            with pytest.raises(EventValidationError):
                finish_basis_for_status(status, "model_declared")

    def test_complete_is_rejected_for_new_writers(self):
        with pytest.raises(EventValidationError, match="complete"):
            make_event(
                "mission.end",
                session="S",
                mission="S__T",
                seq=1,
                status="complete",
                steps=1,
                edited=[],
                warnings=[],
                model="x",
                outcome="x",
            )

    def test_finished_statuses_single_source(self):
        assert FINISHED_STATUSES == ("verified", "needs_review")
        assert "complete" not in FINISHED_STATUSES


class TestSensitive:
    def test_token_key_rejected(self):
        with pytest.raises(EventValidationError, match="forbidden"):
            make_event(
                "gate.pause_minted",
                session="S",
                mission="S__T",
                seq=1,
                action_summary="rm -rf build",
                action_sha256="a" * 64,
                token="tok-deadbeefdeadbeef",
            )

    def test_nested_approval_token_rejected(self):
        with pytest.raises(EventValidationError, match="forbidden"):
            make_event(
                "log",
                session="S",
                mission="S__T",
                seq=1,
                level="info",
                message="x",
                extra={"approval_token_id": "tok-aabbccddeeff0011"},
            )

    def test_token_id_prefix_too_long_rejected(self):
        with pytest.raises(EventValidationError, match="token_id_prefix"):
            make_event(
                "gate.pause_minted",
                session="S",
                mission="S__T",
                seq=1,
                action_summary="rm",
                action_sha256="b" * 64,
                token_id_prefix="tok-deadbe",  # 9 chars and credential-shaped
            )

    def test_opaque_pending_id_allowed(self):
        rec = make_event(
            "gate.pause_minted",
            session="S",
            mission="S__T",
            seq=1,
            action_summary="rm -rf build",
            action_sha256="c" * 64,
            pending_id=42,
            expires_at="2026-09-19T21:08:00+00:00",
        )
        assert rec["pending_id"] == 42
        assert "token" not in rec


class TestRedaction:
    def test_redact_then_truncate(self):
        fake = "ghp_" + "A1b2" * 9
        out = apply_redaction({"outcome": fake + ("x" * 3000), "head": "READ: a.py"})
        assert fake not in out["outcome"]
        assert len(out["outcome"]) <= 2000
        assert out["head"] == "READ: a.py"


class TestNormalizeLegacy:
    def test_start_step_end_aliases(self):
        start = normalize_legacy({"event": "start", "directive": "fix", "model": "m"})
        assert start["event"] == "mission.start"
        step = normalize_legacy(
            {"event": "step", "n": 3, "action": "EDIT: path|||old|||new"}
        )
        assert step["event"] == "step.proposed"
        assert step["action"] == "EDIT"
        assert step["raw"] == "EDIT: path|||old|||new"
        assert step["head"].startswith("EDIT:")
        end = normalize_legacy({"event": "end", "status": "blocked"})
        assert end["event"] == "mission.end"

    def test_does_not_invent_finish_basis(self):
        end = normalize_legacy({"event": "end", "status": "blocked"})
        assert "finish_basis" not in end
        old = normalize_legacy(
            {"event": "end", "status": "needs_review"}  # pre-split-ish, no basis
        )
        assert "finish_basis" not in old

    def test_action_verb_helper(self):
        assert action_verb("EDIT: foo.py|||a|||b") == "EDIT"
        assert action_verb("FINISH: done") == "FINISH"


class TestCompatDedupe:
    def test_alias_shares_seq_and_iter_canonical_drops_it(self):
        canonical = make_event(
            "mission.end",
            session="S",
            mission="S__T",
            seq=11,
            status="blocked",
            steps=3,
            edited=[],
            warnings=[],
            model="grok-4.5",
            outcome="[BLOCKED] x",
        )
        alias = make_compat_alias(canonical)
        assert alias is not None
        assert alias["event"] == "end"
        assert alias["compat"] is True
        assert alias["alias_of"] == 11
        assert alias["seq"] == 11
        tape = list(iter_canonical([canonical, alias]))
        assert len(tape) == 1
        assert tape[0]["event"] == "mission.end"


class TestMissionStem:
    def test_two_stems_in_the_same_second_differ(self):
        from datetime import datetime, timezone

        t = datetime(2026, 9, 19, 21, 3, 0, tzinfo=timezone.utc)
        a = unique_mission_stem("S", when=t)
        b = unique_mission_stem("S", when=t, nonce="ab12")
        assert a.startswith("S__")
        assert a != b
        assert "T210300" in a


class TestBus:
    def test_subscriber_exception_does_not_raise(self, caplog):
        bus = LocalMissionBus()

        def boom(_event):
            raise RuntimeError("subscriber died")

        seen = []
        bus.subscribe(boom)
        bus.subscribe(seen.append)
        with caplog.at_level(logging.ERROR):
            bus.publish({"event": "log", "message": "hi"})
        assert seen == [{"event": "log", "message": "hi"}]

    def test_subscriber_mutation_does_not_corrupt_others(self):
        bus = LocalMissionBus()
        first = []
        second = []

        def mutate(event):
            event["event"] = "mutated"
            event["payload"] = "changed"
            first.append(event)

        bus.subscribe(mutate)
        bus.subscribe(second.append)
        original = {"event": "mission.status", "status": "running"}
        bus.publish(original)
        assert first[0]["event"] == "mutated"
        assert second[0]["event"] == "mission.status"
        assert original["event"] == "mission.status"

    def test_unsubscribe_is_idempotent(self):
        bus = LocalMissionBus()
        hits = []
        unsub = bus.subscribe(hits.append)
        unsub()
        unsub()
        bus.unsubscribe(hits.append)
        bus.publish({"event": "log"})
        assert hits == []
