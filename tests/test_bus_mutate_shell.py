"""Commit 4: fs.mutate, shell.exec, verify.result from existing result markers.

Auto_verify success is not mission verification. FINISH basis assignment is
untouched; these tests only read the tape.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from unittest.mock import patch

from cosmic_cli.bus import LocalMissionBus
from cosmic_cli.events import TEXT_CAPS
from cosmic_cli.secrets import redact

from tests.test_finish_line import make_agent, run


def _subscribe(bus: LocalMissionBus):
    tape = []
    bus.subscribe(tape.append)
    return tape


def _named(tape, name: str):
    return [e for e in tape if e.get("event") == name]


def _agent(tmp_path: Path, bus: LocalMissionBus, **kwargs):
    kwargs.setdefault("use_helix", False)
    kwargs.setdefault("write_echo", False)
    kwargs.setdefault("work_dir", str(tmp_path))
    kwargs.setdefault("auto_verify", False)
    agent = make_agent(bus=bus, **kwargs)
    agent.steps_taken = 1
    return agent


class TestFsMutate:
    def test_full_mode_write_emits_path_without_diff(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus, exec_mode="full")
        out = agent._execute_step("WRITE: notes.txt|||hello")
        assert "WRITE ok" in out
        ev = _named(tape, "fs.mutate")
        assert len(ev) == 1
        rec = ev[0]
        assert rec["op"] == "WRITE"
        assert rec["path"] == "notes.txt"
        assert rec["rel"] == "notes.txt"
        assert rec["abs"].endswith("notes.txt")
        assert "diff" not in rec
        assert "diff_truncated" not in rec
        assert "receipt_id" not in rec
        assert rec["n"] == 1

    def test_full_mode_edit_emits_fs_mutate(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
        agent = _agent(tmp_path, bus, exec_mode="full")
        assert "x = 1" in agent._execute_step("READ: f.py")
        out = agent._execute_step("EDIT: f.py|||x = 1|||x = 2")
        assert "EDIT ok" in out
        ev = _named(tape, "fs.mutate")
        assert len(ev) == 1
        assert ev[0]["op"] == "EDIT"
        assert ev[0]["path"] == "f.py"

    def test_full_mode_create_and_mkdir_emit(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus, exec_mode="full")
        assert "CREATE ok" in agent._execute_step("CREATE: a/b.txt|||hi")
        assert "MKDIR ok" in agent._execute_step("MKDIR: nest")
        ops = [e["op"] for e in _named(tape, "fs.mutate")]
        assert ops == ["CREATE", "MKDIR"]
        paths = [e["path"] for e in _named(tape, "fs.mutate")]
        assert paths == ["a/b.txt", "nest"]

    def test_gateway_write_carries_receipt_id(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus)  # safe: authorize + execute_with_receipt
        out = agent._execute_step("WRITE: x.txt|||y")
        assert "WRITE ok" in out
        rec = _named(tape, "fs.mutate")[0]
        assert rec["op"] == "WRITE"
        assert rec["path"] == "x.txt"
        assert rec["receipt_id"].startswith("rcpt-")
        assert rec.get("checkpoint_id")

    def test_blocked_or_error_does_not_emit(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus, exec_mode="full")
        err = agent._execute_step("EDIT: missing.py|||a|||b")
        assert "READ-before-EDIT" in err
        (tmp_path / "COSMIC.md").write_text(
            """
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| no-secret | WITNESS | WRITE | secret_key |
"""
        )
        blocked_agent = _agent(tmp_path, bus)
        blocked = blocked_agent._execute_step("WRITE: notes.txt|||secret_key=hunter2")
        assert "BLOCKED" in blocked
        assert _named(tape, "fs.mutate") == []


class TestShellExec:
    def test_blocked_shell_is_not_exit_zero(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus)
        blocked = "[BLOCKED] compass PAUSE: approval required"
        with patch.object(agent, "_run_shell", return_value=blocked) as sh:
            out = agent._execute_step("SHELL: echo hi")
        sh.assert_called_once_with("echo hi")
        assert out == blocked
        rec = _named(tape, "shell.exec")[0]
        assert rec["kind"] == "SHELL"
        assert rec["blocked"] is True
        assert rec["exit_code"] is None
        assert rec["exit_code"] != 0
        assert rec["cmd"] == "echo hi"
        assert rec["output_head"].startswith("[BLOCKED]")

    def test_exit_zero_marker(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus)
        with patch.object(agent, "_run_shell", return_value="[exit 0]\nok"):
            agent._execute_step("SHELL: echo hi")
        rec = _named(tape, "shell.exec")[0]
        assert rec["exit_code"] == 0
        assert rec["blocked"] is False
        assert rec["output_head"].startswith("[exit 0]")

    def test_exit_one_marker(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus)
        with patch.object(agent, "_run_shell", return_value="[exit 1]\nfail"):
            agent._execute_step("SHELL: false")
        rec = _named(tape, "shell.exec")[0]
        assert rec["exit_code"] == 1
        assert rec["blocked"] is False

    def test_missing_marker_is_not_success(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus)
        with patch.object(agent, "_run_shell", return_value="1 passed"):
            agent._execute_step("SHELL: pytest -q")
        rec = _named(tape, "shell.exec")[0]
        assert rec["exit_code"] is None
        assert rec["blocked"] is False

    def test_blocked_wins_over_embedded_exit_zero(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus)
        with patch.object(
            agent, "_run_shell", return_value="[BLOCKED] no\n[exit 0]\nok"
        ):
            agent._execute_step("SHELL: echo hi")
        rec = _named(tape, "shell.exec")[0]
        assert rec["blocked"] is True
        assert rec["exit_code"] is None

    def test_test_and_code_kinds(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus)
        with patch.object(agent, "_run_shell", return_value="[exit 0]\n1 passed"):
            agent._execute_step("TEST: tests/ -q")
        with patch.object(agent, "_run_code", return_value="[exit 0]\n"):
            agent._execute_step("CODE: print(1)")
        kinds = [e["kind"] for e in _named(tape, "shell.exec")]
        assert kinds == ["TEST", "CODE"]
        test_ev = _named(tape, "shell.exec")[0]
        assert test_ev["cmd"] == "python -m pytest tests/ -q"

    def test_output_head_capped_and_cmd_redacted(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = _agent(tmp_path, bus)
        fake = "ghp_" + "A1b2" * 9
        blob = "[exit 0]\n" + ("x" * 800)
        with patch.object(agent, "_run_shell", return_value=blob):
            agent._execute_step(f"SHELL: echo {fake}")
        rec = _named(tape, "shell.exec")[0]
        assert len(rec["output_head"]) <= TEXT_CAPS["output_head"]
        assert fake not in rec["cmd"]
        assert fake not in rec["output_head"]
        assert redact(fake) in rec["cmd"]


class TestVerifyResult:
    def test_verify_cmd_role_distinct_and_before_declared(self):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = make_agent(
            bus=bus, verify_cmd="pytest -q", auto_verify=False, use_helix=False
        )
        result, sh = run(agent, ["FINISH: fixed"], shell=["[exit 0]\n1 passed"])
        assert result["status"] == "verified"
        assert result["finish_basis"] == "verifier"
        sh.assert_called_once_with("pytest -q")
        verify = _named(tape, "verify.result")
        assert len(verify) == 1
        rec = verify[0]
        assert rec["role"] == "verify_cmd"
        assert rec["cmd"] == "pytest -q"
        assert rec["exit_code"] == 0
        assert rec["ok"] is True
        assert rec["blocked"] is False
        names = [e["event"] for e in tape]
        assert names.index("verify.result") < names.index("finish.declared")
        assert tape[-1]["event"] == "mission.end"
        assert tape[-1]["status"] == "verified"
        assert _named(tape, "shell.exec") == []

    def test_auto_verify_success_does_not_verify_mission(self, tmp_path: Path):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = make_agent(
            bus=bus,
            work_dir=str(tmp_path),
            exec_mode="full",
            auto_verify=True,
            max_steps=8,
            use_helix=False,
        )
        result, _ = run(
            agent,
            ["WRITE: foo.py|||print(1)\n"] + ["READ: foo.py"] * 5,
            shell=["[exit 0]\nok"],
        )
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "synthesized"
        end = _named(tape, "mission.end")[-1]
        assert end["status"] == "needs_review"
        assert end["finish_basis"] == "synthesized"
        verify = _named(tape, "verify.result")
        assert verify
        assert all(v["role"] == "auto_verify" for v in verify)
        assert any(v.get("ok") is True and v.get("exit_code") == 0 for v in verify)
        assert all(v["role"] != "verify_cmd" for v in verify)
        assert "python -m py_compile" in verify[0]["cmd"]
        assert _named(tape, "shell.exec") == []

    def test_blocked_verify_cmd_is_not_ok(self):
        bus = LocalMissionBus()
        tape = _subscribe(bus)
        agent = make_agent(
            bus=bus, verify_cmd="pytest -q", auto_verify=False, use_helix=False
        )
        result, _ = run(
            agent,
            ["FINISH: fixed"],
            shell=["[BLOCKED] compass PAUSE: approval required"],
        )
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "verifier_blocked"
        rec = _named(tape, "verify.result")[0]
        assert rec["role"] == "verify_cmd"
        assert rec["blocked"] is True
        assert rec["ok"] is False
        assert rec["exit_code"] is None

    def test_existing_synthesized_finish_never_verified(self):
        agent = make_agent(verify_cmd="pytest -q")
        result, _ = run(
            agent,
            itertools.repeat("READ: f.py"),
            shell=itertools.repeat("[exit 0]\nok"),
        )
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "synthesized"


def test_jsonl_carries_canonical_mutate_without_alias(tmp_path: Path):
    bus = LocalMissionBus()
    agent = _agent(tmp_path, bus, exec_mode="full")
    agent._execute_step("WRITE: notes.txt|||hello")
    rows = [
        json.loads(ln)
        for ln in agent.session_path.read_text().splitlines()
        if ln.strip()
    ]
    mutate = [e for e in rows if e.get("event") == "fs.mutate"]
    assert len(mutate) == 1
    assert mutate[0]["v"] == 1
    assert "compat" not in mutate[0]
    assert not any(e.get("event") == "mutate" for e in rows)
