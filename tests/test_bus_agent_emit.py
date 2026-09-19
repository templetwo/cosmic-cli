"""Agent MissionBus emission: ordered tape, dual-write, unique stems."""

from __future__ import annotations

import itertools
import json

from cosmic_cli.bus import LocalMissionBus
from cosmic_cli.events import ENVELOPE_KEYS

from tests.test_finish_line import make_agent, run


def _subscribe(bus):
    tape = []
    bus.subscribe(tape.append)
    return tape


def _jsonl(agent):
    return [
        json.loads(ln)
        for ln in agent.session_path.read_text().splitlines()
        if ln.strip()
    ]


class TestAgentEmit:
    def test_scripted_execute_emits_ordered_canonical_tape(self):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = make_agent(bus=bus)
        run(agent, ["FINISH: done"])
        names = [e["event"] for e in tape]
        assert names[:3] == ["mission.start", "mission.status", "step.proposed"]
        assert tape[1]["status"] == "running"
        assert names[-1] == "mission.end"
        seqs = [e["seq"] for e in tape]
        assert seqs[0] == 0
        assert seqs == list(range(len(seqs)))
        assert all(not e.get("compat") for e in tape)
        assert all("alias_of" not in e for e in tape)

    def test_synthesized_finish_declared_immediately_before_end(self):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = make_agent(bus=bus)
        run(agent, itertools.repeat("READ: f.py"))
        names = [e["event"] for e in tape]
        assert names[-2] == "finish.declared"
        assert names[-1] == "mission.end"
        declared = tape[-2]
        assert declared["synthesized"] is True
        assert declared["status"] == "needs_review"
        assert declared["finish_basis"] == "synthesized"
        assert tape[-1]["status"] == "needs_review"
        assert tape[-1]["finish_basis"] == "synthesized"

    def test_jsonl_contains_canonical_and_compat_end(self):
        bus = LocalMissionBus()
        agent = make_agent(bus=bus)
        run(agent, ["FINISH: done"])
        rows = _jsonl(agent)
        assert any(e.get("event") == "mission.end" for e in rows)
        assert any(e.get("event") == "end" for e in rows)
        canon = [e for e in rows if e.get("event") == "mission.end"][-1]
        alias = [e for e in rows if e.get("event") == "end"][-1]
        assert alias.get("compat") is True
        assert alias.get("alias_of") == canon["seq"]
        assert alias["seq"] == canon["seq"]

    def test_two_agents_have_distinct_session_paths(self):
        a = make_agent(session_id="seat-1")
        b = make_agent(session_id="seat-1")
        assert a.session_id == b.session_id == "seat-1"
        assert a.session_path.name != b.session_path.name
        assert a.mission_id != b.mission_id

    def test_envelope_fields_present(self):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = make_agent(bus=bus)
        run(agent, ["FINISH: done"])
        for rec in tape:
            for key in ENVELOPE_KEYS:
                assert key in rec
            assert rec["v"] == 1
            assert rec["mission"] == agent.mission_id
            assert rec["session"] == agent.session_id

    def test_blocked_known_class_emits_compass_verdict(self):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = make_agent(bus=bus)
        run(
            agent,
            ["SHELL: echo hi"],
            shell=["[BLOCKED] compass PAUSE: approval required"],
        )
        names = [e["event"] for e in tape]
        assert "compass.verdict" in names
        assert "finish.declared" not in names
        verdict = next(e for e in tape if e["event"] == "compass.verdict")
        assert verdict["classification"] == "PAUSE"
        assert verdict["n"] == 1
        assert names[-1] == "mission.end"
        assert tape[-1]["status"] == "blocked"
        assert "finish_basis" not in tape[-1]
        assert all(not e.get("compat") for e in tape)

    def test_blocked_unknown_class_skips_compass_verdict(self):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = make_agent(bus=bus)
        run(
            agent,
            ["SHELL: echo hi"],
            shell=["[BLOCKED] dangerous pattern in safe mode: 'rm'"],
        )
        names = [e["event"] for e in tape]
        assert "compass.verdict" not in names
        assert "finish.declared" not in names
        assert names[-1] == "mission.end"
        assert tape[-1]["status"] == "blocked"
        assert "finish_basis" not in tape[-1]
