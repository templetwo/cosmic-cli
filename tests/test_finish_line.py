"""The finish line: a mission's terminal status must say what was established.

Before this cut, status "complete" covered three different endings:
  - the model said FINISH and py_compile passed (a syntax check),
  - the model said FINISH and nothing was checked,
  - the harness synthesized a FINISH because the model was looping.

A synthesized FINISH is an actor approving its own level. These tests pin the
split: "verified" only when an operator-supplied verifier exits 0 on a FINISH
the model itself declared; everything else that reaches the finish path is
"needs_review", with the basis named.

The verifier is stubbed at `_run_shell` because the real one needs the kernel
floor. That stub is also the assertion that the verifier takes the gated shell
path: if the agent ever ran it through a bare subprocess, `_run_shell` would
not be called and these tests go red.
"""

import itertools
from unittest.mock import patch

import pytest

from cosmic_cli.agents import FINISHED_STATUSES, StargazerAgent


def make_agent(directive="count lines only", **kw):
    kw.setdefault("max_steps", 6)
    return StargazerAgent(
        directive,
        api_key="test_key",
        quiet=True,
        show_progress=False,
        write_echo=False,
        use_helix=False,
        **kw,
    )


def run(agent, steps, shell=None):
    """Drive execute() with a scripted model and, optionally, a scripted shell."""
    with patch.object(agent, "_ask_grok_for_next_step", side_effect=steps), patch.object(
        agent.context_manager, "read_file", return_value="line1\nline2\n"
    ):
        if shell is None:
            return agent.execute(), None
        with patch.object(agent, "_run_shell", side_effect=shell) as sh:
            return agent.execute(), sh


class TestNoVerifier:
    def test_model_declared_finish_is_needs_review(self):
        result, _ = run(make_agent(), ["FINISH: 2 lines"])
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "model_declared"

    def test_complete_is_never_reported(self):
        result, _ = run(make_agent(), ["FINISH: 2 lines"])
        assert result["status"] != "complete"
        assert "complete" not in FINISHED_STATUSES


class TestSynthesizedFinish:
    def test_loop_breaker_finish_is_needs_review(self):
        # The model never says FINISH. The harness stops the loop for it.
        result, _ = run(make_agent(), itertools.repeat("READ: f.py"))
        assert result["results"][-1]["step"] == "FINISH"
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "synthesized"

    def test_synthesized_finish_is_never_verified_even_with_a_passing_verifier(self):
        agent = make_agent(verify_cmd="pytest -q")
        result, _ = run(
            agent, itertools.repeat("READ: f.py"), shell=itertools.repeat("[exit 0]\nok")
        )
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "synthesized"


class TestVerifier:
    def test_passing_verifier_on_a_declared_finish_is_verified(self):
        agent = make_agent(verify_cmd="pytest -q")
        result, sh = run(agent, ["FINISH: fixed"], shell=["[exit 0]\n1 passed"])
        assert result["status"] == "verified"
        assert result["finish_basis"] == "verifier"
        sh.assert_called_once_with("pytest -q")

    def test_failing_verifier_does_not_finish_and_is_handed_back_to_the_model(self):
        agent = make_agent(verify_cmd="pytest -q")
        result, sh = run(
            agent,
            ["FINISH: fixed", "FINISH: fixed for real"],
            shell=["[exit 1]\n1 failed", "[exit 0]\n1 passed"],
        )
        assert sh.call_count == 2
        assert result["status"] == "verified"
        failures = [r for r in result["results"] if r["step"] == "VERIFY:cmd"]
        assert failures and "1 failed" in failures[0]["result"]

    def test_verifier_that_never_passes_ends_as_max_steps_not_a_finish(self):
        agent = make_agent(verify_cmd="pytest -q", max_steps=3)
        result, _ = run(
            agent,
            itertools.repeat("FINISH: fixed"),
            shell=itertools.repeat("[exit 1]\n1 failed"),
        )
        assert result["status"] == "max_steps"
        assert result["status"] not in FINISHED_STATUSES

    @pytest.mark.parametrize(
        "shell_output",
        [
            "[BLOCKED] compass PAUSE: approval required",
            "[SKIPPED] declined",
            "",
            "1 passed",  # no exit marker: success cannot be established
        ],
    )
    def test_verifier_that_could_not_run_fails_closed_to_needs_review(self, shell_output):
        agent = make_agent(verify_cmd="pytest -q")
        result, _ = run(agent, ["FINISH: fixed"], shell=[shell_output])
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "verifier_blocked"


class TestRedaction:
    def test_unrunnable_verifier_output_is_redacted_in_the_finish_text(self):
        from cosmic_cli.secrets import redact

        # Shaped like a credential, assembled at runtime, never a real one.
        fake = "ghp_" + "A1b2" * 9
        assert redact(fake) != fake, "precondition: the funnel masks this shape"
        agent = make_agent(verify_cmd="pytest -q")
        result, _ = run(agent, ["FINISH: fixed"], shell=[f"[BLOCKED] leaked {fake}"])
        finish_text = result["results"][-1]["result"]
        assert "could not run" in finish_text
        assert fake not in finish_text


class TestConsumers:
    def test_theme_styles_every_finished_status(self):
        from cosmic_cli.theme import STATUS_STYLE

        for status in FINISHED_STATUSES:
            assert status in STATUS_STYLE

    def test_dashboard_counts_the_two_outcomes_separately(self, tmp_path, monkeypatch):
        import json
        import sys

        # dashboard.py reads its port from sys.argv at import time; under
        # pytest that would be the test path.
        monkeypatch.setattr(sys, "argv", ["dashboard"])
        from cosmic_cli import dashboard

        echo = tmp_path / "echo.jsonl"
        rows = [
            {"status": "verified"},
            {"status": "needs_review"},
            {"status": "needs_review"},
            {"status": "complete"},  # a record written before the split
            {"status": "blocked"},
        ]
        echo.write_text("".join(json.dumps(r) + "\n" for r in rows))
        monkeypatch.setattr(dashboard, "ECHO", echo)
        counts = dashboard.mission_counts()
        assert counts == {
            "missions_verified": 1,
            "missions_needs_review": 2,
            "missions_complete": 1,
            "missions_blocked": 1,
        }


class TestCli:
    def _invoke(self, monkeypatch, status, args):
        from click.testing import CliRunner

        from cosmic_cli import main as main_module

        seen = {}

        def fake_run(directive, **kw):
            seen.update(kw)
            return {"status": status, "session": "s", "edited": []}

        monkeypatch.setattr(main_module, "_run_stargazer", fake_run)
        return CliRunner().invoke(main_module.cli, args), seen

    def test_do_passes_the_verifier_through(self, monkeypatch):
        _, seen = self._invoke(
            monkeypatch, "verified", ["do", "fix it", "--verify-cmd", "pytest -q"]
        )
        assert seen["verify_cmd"] == "pytest -q"

    @pytest.mark.parametrize(
        "status,code",
        [("verified", 0), ("needs_review", 0), ("max_steps", 1), ("blocked", 4)],
    )
    def test_do_exit_codes(self, monkeypatch, status, code):
        result, _ = self._invoke(monkeypatch, status, ["do", "fix it"])
        assert result.exit_code == code
