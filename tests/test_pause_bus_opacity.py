"""PAUSE mint/claim bus opacity: no token bodies on MissionBus or JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmic_cli.agents import StargazerAgent
from cosmic_cli.bus import LocalMissionBus
from cosmic_cli.events import FORBIDDEN_TOKEN_KEYS
from cosmic_cli.gateway import ApprovalManager
from cosmic_cli.policy import ActionType, evaluate_rules
from cosmic_cli.rules import load_rules_from_markdown

FAKE_TOKEN = "tok-deadbeefdeadbeef"
HOME_PAUSE = Path.home() / ".cosmic-cli" / "last_pause_token.json"


def _write_rule(tmp_path: Path, *, rule_id: str, disposition: str, pattern: str) -> None:
    (tmp_path / "COSMIC.md").write_text(
        f"""
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| {rule_id} | {disposition} | SHELL | {pattern} |
"""
    )


def _subscribe(bus: LocalMissionBus):
    tape = []
    bus.subscribe(tape.append)
    return tape


def _jsonl(agent: StargazerAgent):
    if not agent.session_path.is_file():
        return []
    return [
        json.loads(ln)
        for ln in agent.session_path.read_text().splitlines()
        if ln.strip()
    ]


def _agent(tmp_path: Path, bus: LocalMissionBus, store: Path, **kwargs) -> StargazerAgent:
    kwargs.setdefault("write_echo", False)
    kwargs.setdefault("use_helix", False)
    agent = StargazerAgent(
        "t",
        api_key="test",
        quiet=True,
        show_progress=False,
        work_dir=str(tmp_path),
        bus=bus,
        **kwargs,
    )
    mgr = ApprovalManager(store_path=store)
    agent._approval_mgr = mgr
    agent._gateway.approval_manager = mgr
    return agent


def _assert_opaque(records) -> None:
    for rec in records:
        for key in FORBIDDEN_TOKEN_KEYS:
            assert key not in rec
        assert "token" not in rec
        blob = json.dumps(rec)
        assert "tok-" not in blob
        assert FAKE_TOKEN not in blob
        assert "pending_token" not in blob


def _unused_token(store: Path, action_sha256: str) -> str:
    data = json.loads(store.read_text())
    return next(
        tid
        for tid, meta in data.items()
        if meta.get("action_sha256") == action_sha256 and not meta.get("used")
    )


@pytest.fixture(autouse=True)
def _redirect_pause_token_file(tmp_path, monkeypatch):
    """Mint emit stays on the agent; token file never touches the real home."""
    dest = tmp_path / "last_pause_token.json"

    def _write(self, tok, *, channel, action_sha=""):
        dest.write_text(
            json.dumps(
                {
                    "token": tok,
                    "channel": channel,
                    "action_sha256": action_sha,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(StargazerAgent, "_human_pause_token", _write)
    return dest


def test_local_pause_mint_is_opaque_on_bus_and_jsonl(tmp_path, _redirect_pause_token_file):
    home_before = HOME_PAUSE.stat().st_mtime_ns if HOME_PAUSE.exists() else None
    store = tmp_path / "approvals.json"
    _write_rule(tmp_path, rule_id="outbound-net", disposition="PAUSE", pattern="curl")
    bus = LocalMissionBus()
    tape = _subscribe(bus)
    agent = _agent(tmp_path, bus, store)

    blocked = agent._compass_gate("curl https://example.com", kind="SHELL")
    assert blocked and "BLOCKED" in blocked and "PAUSE" in blocked
    assert "tok-" not in blocked
    assert FAKE_TOKEN not in blocked
    assert "COSMIC_APPROVAL_TOKEN=" not in blocked

    minted = [e for e in tape if e["event"] == "gate.pause_minted"]
    assert len(minted) == 1
    rec = minted[0]
    assert rec["action_sha256"]
    assert rec["action_summary"]
    assert "expires_at" in rec
    assert "pending_id" not in rec
    assert "token_id_prefix" not in rec
    _assert_opaque(tape)
    _assert_opaque(_jsonl(agent))

    verdicts = [e for e in tape if e["event"] == "compass.verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["classification"] == "PAUSE"
    assert not any(e["event"] == "gate.pause_resolved" for e in tape)

    home_after = HOME_PAUSE.stat().st_mtime_ns if HOME_PAUSE.exists() else None
    assert home_after == home_before
    assert _redirect_pause_token_file.is_file()


def test_seeded_token_does_not_appear_on_bus(tmp_path):
    store = tmp_path / "approvals.json"
    _write_rule(
        tmp_path,
        rule_id=FAKE_TOKEN,
        disposition="PAUSE",
        pattern="echo",
    )
    bus = LocalMissionBus()
    tape = _subscribe(bus)
    agent = _agent(tmp_path, bus, store)

    blocked = agent._compass_gate(f"echo hi {FAKE_TOKEN}", kind="SHELL")
    assert blocked and "PAUSE" in blocked
    _assert_opaque(tape)
    jsonl = _jsonl(agent)
    _assert_opaque(jsonl)
    blob = json.dumps(tape) + json.dumps(jsonl)
    assert FAKE_TOKEN not in blob
    assert "tok-" not in blob


def test_claim_once_emits_approved_by_operator_exactly_once(tmp_path):
    store = tmp_path / "approvals.json"
    _write_rule(tmp_path, rule_id="echo-pause", disposition="PAUSE", pattern="echo hi")
    bus1 = LocalMissionBus()
    tape1 = _subscribe(bus1)
    agent1 = _agent(tmp_path, bus1, store)

    blocked = agent1._compass_gate("echo hi", kind="SHELL")
    assert blocked and "PAUSE" in blocked
    assert "tok-" not in blocked
    assert any(e["event"] == "gate.pause_minted" for e in tape1)
    assert not any(e["event"] == "gate.pause_resolved" for e in tape1)

    rules = load_rules_from_markdown(tmp_path / "COSMIC.md")
    decision = evaluate_rules(rules, ActionType.SHELL, "echo hi")
    tok = _unused_token(store, decision.evaluated_input_sha256)

    bus2 = LocalMissionBus()
    tape2 = _subscribe(bus2)
    agent2 = _agent(tmp_path, bus2, store, approval_token_id=tok)
    out2 = agent2._compass_gate("echo hi", kind="SHELL")
    assert out2 is None
    resolved = [e for e in tape2 if e["event"] == "gate.pause_resolved"]
    assert len(resolved) == 1
    assert resolved[0]["decision"] == "approved"
    assert resolved[0]["by"] == "operator"
    assert resolved[0]["action_sha256"] == decision.evaluated_input_sha256
    assert not any(e["event"] == "gate.pause_minted" for e in tape2)
    _assert_opaque(tape2)
    _assert_opaque(_jsonl(agent2))

    bus3 = LocalMissionBus()
    tape3 = _subscribe(bus3)
    agent3 = _agent(tmp_path, bus3, store, approval_token_id=tok)
    out3 = agent3._compass_gate("echo hi", kind="SHELL")
    assert out3 is not None
    assert "BLOCKED" in out3
    assert not any(e["event"] == "gate.pause_resolved" for e in tape3)
    assert not any(e.get("by") == "operator" for e in tape3)
    _assert_opaque(tape3)


def test_failed_claim_emits_invalid_not_operator(tmp_path):
    store = tmp_path / "approvals.json"
    _write_rule(tmp_path, rule_id="echo-pause", disposition="PAUSE", pattern="echo hi")
    bus = LocalMissionBus()
    agent = _agent(tmp_path, bus, store)
    assert agent._compass_gate("echo hi", kind="SHELL")

    bus2 = LocalMissionBus()
    tape2 = _subscribe(bus2)
    agent2 = _agent(tmp_path, bus2, store, approval_token_id=FAKE_TOKEN)
    agent2._approval_mgr.validate = lambda *a, **k: True  # type: ignore
    out = agent2._compass_gate("echo hi", kind="SHELL")
    assert out is not None and "BLOCKED" in out
    assert FAKE_TOKEN not in out
    resolved = [e for e in tape2 if e["event"] == "gate.pause_resolved"]
    assert len(resolved) == 1
    assert resolved[0]["decision"] == "invalid"
    assert "by" not in resolved[0]
    _assert_opaque(tape2)
    _assert_opaque(_jsonl(agent2))


def test_witness_emits_verdict_without_pause_mint(tmp_path):
    store = tmp_path / "approvals.json"
    _write_rule(tmp_path, rule_id="echo-witness", disposition="WITNESS", pattern="echo hi")
    bus = LocalMissionBus()
    tape = _subscribe(bus)
    agent = _agent(tmp_path, bus, store)
    out = agent._compass_gate("echo hi", kind="SHELL")
    assert out is not None and "WITNESS" in out
    names = [e["event"] for e in tape]
    assert "compass.verdict" in names
    assert "gate.pause_minted" not in names
    assert "gate.pause_resolved" not in names
    verdict = next(e for e in tape if e["event"] == "compass.verdict")
    assert verdict["classification"] == "WITNESS"
    _assert_opaque(tape)


def test_open_allow_does_not_invent_open_verdict(tmp_path):
    store = tmp_path / "approvals.json"
    _write_rule(tmp_path, rule_id="echo-open", disposition="OPEN", pattern="echo hi")
    bus = LocalMissionBus()
    tape = _subscribe(bus)
    agent = _agent(tmp_path, bus, store)
    out = agent._compass_gate("echo hi", kind="SHELL")
    assert out is None
    assert "compass.verdict" not in [e["event"] for e in tape]
    assert "gate.pause_minted" not in [e["event"] for e in tape]
