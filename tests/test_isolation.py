"""The test suite must not write into the operator's live stores.

Before this file existed, running `pytest tests/` appended to three real
places on the machine: ~/.cosmic_echo.jsonl (which Mission Control counts),
~/.cosmic-cli/sessions (which Mission Control and any session reader tail),
and the T2Helix chronicle (which recall surfaces to every seat). 611 of 638
"COSMIC mission" rows in one live chronicle were unit-test artifacts.

The isolation lives in tests/conftest.py as an autouse fixture, so a test
that forgets to redirect a path is safe by default.
"""

from pathlib import Path
from unittest.mock import patch

import cosmic_cli.agents as agents
from cosmic_cli.agents import StargazerAgent

HOME = Path.home()


def test_echo_file_is_not_the_operators():
    assert agents.ECHO_FILE != HOME / ".cosmic_echo.jsonl"
    assert HOME / ".cosmic_echo.jsonl" not in agents.ECHO_FILE.parents


def test_session_dir_is_not_the_operators():
    real = HOME / ".cosmic-cli" / "sessions"
    assert agents.SESSION_DIR != real
    assert real not in agents.SESSION_DIR.parents


def test_an_agent_left_on_defaults_writes_nothing_live():
    # Refuse to run the probe at all unless the stores are already redirected:
    # a failing run of this test must not itself write a live row.
    assert getattr(agents.helix_bridge.record, "is_test_stub", False)
    assert agents.SESSION_DIR != HOME / ".cosmic-cli" / "sessions"
    assert agents.ECHO_FILE != HOME / ".cosmic_echo.jsonl"

    # Defaults on purpose: write_echo=True, use_helix=True. This is the shape
    # of the tests that leaked.
    agent = StargazerAgent(
        "isolation probe", api_key="test_key", quiet=True, show_progress=False, max_steps=2
    )
    with patch.object(agent, "_ask_grok_for_next_step", side_effect=["FINISH: ok"]):
        agent.execute()
    assert agent.session_path.parent == agents.SESSION_DIR
    assert agents.ECHO_FILE.exists()
    # The stub saw the mission; the live chronicle did not.
    assert any("isolation probe" in c for c in agents.helix_bridge.record.calls)
