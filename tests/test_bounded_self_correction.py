"""BoundedSelfCorrection happy path — iteration-cap fix (adoption 2026-07-15)."""

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
    """INTAKE→…→DONE must finish clean — not via max-iterations + rollback."""
    target = tmp_path / "work.txt"
    target.write_text("v1")
    mgr = CheckpointManager(tmp_path)
    gw = ActionGateway(_open_evaluator, checkpoint_manager=mgr)
    loop = BoundedSelfCorrection(gw, mgr)

    ctx = loop.run("safe plan", [target], action_type=ActionType.SHELL)

    assert ctx.state == CorrectionState.DONE
    assert ctx.error is None, f"unexpected error: {ctx.error}"
    assert ctx.result == "Executed: safe plan"
    assert ctx.checkpoint is not None
    # Original content still present (no erroneous post-success rollback)
    assert target.read_text() == "v1"
