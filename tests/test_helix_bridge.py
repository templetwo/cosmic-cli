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


def test_code_path_gated_like_shell():
    """CODE: must not bypass compass (Claude CODE-path finding)."""
    from cosmic_cli.agents import StargazerAgent

    agent = StargazerAgent(
        "t", api_key="test", quiet=True, use_helix=False, show_progress=False
    )
    # Local blocklist catches rm inside Python without needing Helix
    out = agent._run_code('import os; os.system("rm -rf /tmp/x")')
    assert "BLOCKED" in out

    # Credential-shaped constant in Python — needs Helix; if offline still
    # should not crash
    agent2 = StargazerAgent(
        "t",
        api_key="test",
        quiet=True,
        use_helix=True,
        show_progress=False,
        session_id="code-gate-test",
    )
    out2 = agent2._run_code(
        'print("sk-abcdefghijklmnopqrstuvwxyz0123456789")'
    )
    # Either PAUSE from Helix or (if helix down) may run — only assert no crash
    assert isinstance(out2, str)
    if "BLOCKED" in out2:
        assert "PAUSE" in out2 or "WITNESS" in out2


def test_blocked_shell_stops_without_thrash():
    """PAUSE block must surface to caller, not thrash-retry (Claude #2)."""
    from cosmic_cli.agents import StargazerAgent

    agent = StargazerAgent(
        "run secret shell",
        api_key="test",
        quiet=True,
        use_helix=False,
        max_steps=6,
        show_progress=False,
    )
    # Force model to keep emitting the same SHELL
    steps = [
        'SHELL: export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789 && echo hi',
        'SHELL: export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789 && echo hi',
        'SHELL: export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789 && echo hi',
    ]
    # Without helix, shell_guard may not PAUSE — simulate BLOCKED output
    with patch.object(
        agent, "_ask_grok_for_next_step", side_effect=steps
    ), patch.object(
        agent,
        "_execute_step",
        return_value=(
            "[BLOCKED] Helix compass PAUSE: test. "
            "Approve: cosmic-cli helix confirm deadbeef then re-run."
        ),
    ):
        result = agent.execute()
    assert result["status"] == "blocked"
    assert "PAUSE" in result.get("block_message", "")
    # Only one execute of the blocked step, not three thrash retries
    # (one next_step call before return)
    assert result["steps_taken"] == 1


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


def test_parse_witness_fail_closed_on_error_and_unknown():
    """Unknown/malformed envelopes must not OPEN (Claude exercise)."""
    err = helix_bridge.parse_witness({"ok": False, "error": "rpc down"})
    assert err["blocked"] is True
    assert err["classification"] == "WITNESS"
    assert err.get("fail_closed") is True

    bad = helix_bridge.parse_witness(
        {"ok": True, "result": {"classification": "BANANA", "blocked": False}}
    )
    assert bad["blocked"] is True
    assert bad["classification"] == "WITNESS"

    empty = helix_bridge.parse_witness({"ok": True, "result": {}})
    assert empty["blocked"] is True


def test_compass_gate_fail_closed_when_witness_raises():
    """Raised witness must not fall through to allow (integration map step 0)."""
    from cosmic_cli.agents import StargazerAgent

    agent = StargazerAgent(
        "t",
        api_key="test",
        quiet=True,
        use_helix=True,
        show_progress=False,
        work_dir="/tmp",
    )
    # Force helix path: use_helix True and mock bridge
    import cosmic_cli.agents as agents_mod

    class BoomBridge:
        @staticmethod
        def witness(**kwargs):
            raise NotADirectoryError("simulated data-dir failure")

        @staticmethod
        def parse_witness(w):
            return {}

    with patch.object(agents_mod, "helix_bridge", BoomBridge):
        agent.use_helix = True
        out = agent._compass_gate("echo hello", kind="SHELL")
    assert out is not None
    assert "BLOCKED" in out
    assert "fail-closed" in out

