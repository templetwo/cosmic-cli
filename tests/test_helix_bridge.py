from unittest.mock import patch

from cosmic_cli import helix_bridge


def test_format_boot_offline():
    text = helix_bridge.format_boot_context({"ok": False, "error": "no node"})
    assert "offline" in text


def test_format_boot_online():
    text = helix_bridge.format_boot_context(
        {
            "ok": True,
            "result": {
                "sessionId": "s1",
                "memoryCount": 2,
                "dataDir": "/tmp/x",
                "memory": [{"layer": "hypothesis", "content": "hello helix"}],
            },
        }
    )
    assert "hello helix" in text
    assert "s1" in text


def test_call_when_unavailable():
    with patch.object(helix_bridge, "available", return_value=False):
        r = helix_bridge.call("health")
    assert r["ok"] is False


def test_witness_sends_structured_bash():
    """Shell must not go as a plain string (Grok tag skips Bash rules)."""
    captured = {}

    def fake_call(op, **kwargs):
        captured["op"] = op
        captured["kwargs"] = kwargs
        return {
            "ok": True,
            "result": {
                "classification": "PAUSE",
                "blocked": True,
                "pending_token": "deadbeef",
                "reason": "test",
            },
        }

    with patch.object(helix_bridge, "call", side_effect=fake_call):
        r = helix_bridge.witness(
            tool_name="Bash",
            tool_input={"command": "echo sk-test"},
            session_id="s1",
        )
    assert captured["op"] == "witness"
    action = captured["kwargs"]["action"]
    assert action["tool_name"] == "Bash"
    assert action["tool_input"]["command"] == "echo sk-test"
    d = helix_bridge.parse_witness(r)
    assert d["classification"] == "PAUSE"
    assert d["pending_token"] == "deadbeef"


def test_parse_witness_pause_and_witness():
    pause = helix_bridge.parse_witness(
        {
            "ok": True,
            "result": {
                "classification": "PAUSE",
                "blocked": True,
                "pending_token": "abc123",
                "reason": "credential",
            },
        }
    )
    assert pause["classification"] == "PAUSE"
    assert pause["blocked"] is True
    assert pause["pending_token"] == "abc123"

    wit = helix_bridge.parse_witness(
        {
            "ok": True,
            "result": {"classification": "WITNESS", "blocked": True},
        }
    )
    assert wit["blocked"] is True

    open_ = helix_bridge.parse_witness(
        {"ok": True, "result": {"classification": "OPEN", "blocked": False}}
    )
    assert open_["blocked"] is False
