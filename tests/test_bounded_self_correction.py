"""BoundedSelfCorrection — attempt accounting + clean ACCEPT (exercise findings)."""

from pathlib import Path

from cosmic_cli.checkpoint import CheckpointManager
from cosmic_cli.gateway import ActionGateway
from cosmic_cli.policy import ActionType, Disposition, PolicyDecision
from cosmic_cli.self_correction import BoundedSelfCorrection, CorrectionState


def _open_evaluator(rules, action_type, action_input):
    return PolicyDecision(
        disposition=Disposition.OPEN,
        matches=(),
        default_used=True,
        policy_sha256="p",
        evaluated_input_sha256="a",
    )


def test_happy_path_accepts_without_cap_error(tmp_path: Path):
    """INTAKE→…→DONE must finish clean; executor mutation must survive."""
    target = tmp_path / "work.txt"
    target.write_text("v1")
    mgr = CheckpointManager(tmp_path)
    gw = ActionGateway(_open_evaluator, checkpoint_manager=mgr)
    loop = BoundedSelfCorrection(gw, mgr)

    # Wrap authorize so EXECUTE-equivalent mutation happens under the checkpoint
    real_auth = gw.authorize

    def auth_and_mutate(*a, **k):
        receipt = real_auth(*a, **k)
        target.write_text("v2-mutated-by-executor")
        return receipt

    gw.authorize = auth_and_mutate  # type: ignore

    ctx = loop.run("safe plan", [target], action_type=ActionType.SHELL)

    assert ctx.state == CorrectionState.DONE
    assert ctx.error is None, f"unexpected error: {ctx.error}"
    assert ctx.result == "Executed: safe plan"
    assert ctx.attempts == 1  # one REVIEW cycle, not 6 bookkeeping ticks
    assert ctx.safety_steps >= 6
    assert target.read_text() == "v2-mutated-by-executor"


def test_attempt_budget_not_bookkeeping(tmp_path: Path):
    """max_attempts=1 still allows the happy path (one REVIEW), not 6 ticks."""
    target = tmp_path / "work.txt"
    target.write_text("v1")
    mgr = CheckpointManager(tmp_path)
    gw = ActionGateway(_open_evaluator, checkpoint_manager=mgr)
    loop = BoundedSelfCorrection(gw, mgr)

    ctx = loop.run("plan", [target], action_type=ActionType.SHELL, max_attempts=1)
    assert ctx.state == CorrectionState.DONE
    assert ctx.error is None
    assert ctx.attempts == 1
