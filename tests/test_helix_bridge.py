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
