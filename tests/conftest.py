"""Suite-wide isolation from the operator's live stores.

A StargazerAgent on its defaults appends to ~/.cosmic_echo.jsonl, writes a
session log under ~/.cosmic-cli/sessions, and records to the T2Helix chronicle.
Those are the operator's real memory and what Mission Control reads. Redirect
all three for every test, so a test that forgets is safe by default. A test
that needs its own path can still monkeypatch over these.
"""

import pytest


class _HelixRecordStub:
    """Stands in for helix_bridge.record: remembers calls, writes nothing."""

    is_test_stub = True

    def __init__(self):
        self.calls = []

    def __call__(self, content, **kwargs):
        self.calls.append(content)
        return {"ok": True, "result": {"id": None, "stub": True}}


@pytest.fixture(autouse=True)
def _isolate_live_stores(tmp_path_factory, monkeypatch):
    import cosmic_cli.agents as agents
    import cosmic_cli.pause_authority as pause_authority

    root = tmp_path_factory.mktemp("cosmic_live_stores")
    monkeypatch.setattr(agents, "ECHO_FILE", root / "echo.jsonl")
    monkeypatch.setattr(agents, "SESSION_DIR", root / "sessions")
    monkeypatch.setattr(
        pause_authority, "STAGE_PATH", root / "operator_approval_token"
    )
    if agents.helix_bridge is not None:
        monkeypatch.setattr(agents.helix_bridge, "record", _HelixRecordStub())
    yield
