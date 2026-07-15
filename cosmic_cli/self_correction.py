"""Bounded self-correction state machine (avionics Phase 6).

Attempt budget counts WORK cycles (each entry to REVIEW), not bookkeeping
transitions. Bookkeeping (INTAKE→PLAN→REVIEW, ACCEPT→DONE) does not burn the
cap — so retry edges can be added later without the old cap-raise bug shape.
A separate safety_steps ceiling bounds infinite loops.
EXECUTE remains a stub until wired through gateway.execute_with_receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

from cosmic_cli.checkpoint import CheckpointManager, CheckpointManifest
from cosmic_cli.gateway import ActionGateway
from cosmic_cli.policy import ActionType


class CorrectionState(Enum):
    INTAKE = auto()
    PLAN = auto()
    REVIEW = auto()
    EXECUTE = auto()
    VERIFY = auto()
    ACCEPT = auto()
    ROLLBACK = auto()
    DONE = auto()


@dataclass
class CorrectionContext:
    state: CorrectionState = CorrectionState.INTAKE
    plan: Optional[str] = None
    intended_paths: List[Path] = field(default_factory=list)
    checkpoint: Optional[CheckpointManifest] = None
    result: Optional[str] = None
    error: Optional[str] = None
    # Work attempts: incremented only when entering REVIEW (authorize cycle).
    attempts: int = 0
    max_attempts: int = 8
    # Total state transitions — hard ceiling against infinite loops.
    safety_steps: int = 0
    max_safety_steps: int = 64
    # Back-compat alias used by older tests/callers
    @property
    def iteration(self) -> int:
        return self.attempts

    @property
    def max_iterations(self) -> int:
        return self.max_attempts


class BoundedSelfCorrection:
    """Bounded self-correction state machine. Uses gateway + checkpoint layers."""

    def __init__(self, gateway: ActionGateway, checkpoint_manager: CheckpointManager):
        self.gateway = gateway
        self.checkpoint_manager = checkpoint_manager

    def run(
        self,
        initial_plan: str,
        intended_paths: List[Path],
        action_type: ActionType = ActionType.SHELL,
        *,
        max_attempts: Optional[int] = None,
    ) -> CorrectionContext:
        ctx = CorrectionContext(plan=initial_plan, intended_paths=intended_paths)
        if max_attempts is not None:
            ctx.max_attempts = max_attempts

        while (
            ctx.state != CorrectionState.DONE
            and ctx.safety_steps < ctx.max_safety_steps
        ):
            ctx.safety_steps += 1

            if ctx.state == CorrectionState.INTAKE:
                ctx.state = CorrectionState.PLAN

            elif ctx.state == CorrectionState.PLAN:
                ctx.state = CorrectionState.REVIEW

            elif ctx.state == CorrectionState.REVIEW:
                # One work attempt per authorize/execute cycle
                ctx.attempts += 1
                if ctx.attempts > ctx.max_attempts:
                    ctx.error = (
                        f"Max correction attempts ({ctx.max_attempts}) reached "
                        "— bounded safety triggered"
                    )
                    ctx.state = CorrectionState.ROLLBACK
                    continue
                try:
                    receipt = self.gateway.authorize(
                        action_type=action_type,
                        action_input=ctx.plan or "",
                        policy_rules=[],
                        intended_paths=ctx.intended_paths,
                    )
                    if receipt.checkpoint_manifest:
                        ctx.checkpoint = receipt.checkpoint_manifest
                    ctx.state = CorrectionState.EXECUTE
                except PermissionError as e:
                    ctx.error = str(e)
                    ctx.state = CorrectionState.ROLLBACK

            elif ctx.state == CorrectionState.EXECUTE:
                try:
                    # Stub executor — real path will use gateway.execute_with_receipt.
                    ctx.result = f"Executed: {ctx.plan}"
                    ctx.state = CorrectionState.VERIFY
                except Exception as e:
                    ctx.error = str(e)
                    ctx.state = CorrectionState.ROLLBACK

            elif ctx.state == CorrectionState.VERIFY:
                if ctx.checkpoint and self.checkpoint_manager:
                    current = [
                        self.checkpoint_manager.workspace_root / snap.relative_path
                        for snap in ctx.checkpoint.files
                    ]
                    unexpected = self.checkpoint_manager.detect_unrelated_changes(
                        ctx.checkpoint, current
                    )
                    if unexpected:
                        ctx.error = f"Unrelated changes: {list(unexpected)}"
                        ctx.state = CorrectionState.ROLLBACK
                    else:
                        ctx.state = CorrectionState.ACCEPT
                else:
                    ctx.state = CorrectionState.ACCEPT

            elif ctx.state == CorrectionState.ACCEPT:
                ctx.state = CorrectionState.DONE

            elif ctx.state == CorrectionState.ROLLBACK:
                if ctx.checkpoint and self.checkpoint_manager:
                    try:
                        self.checkpoint_manager.rollback(ctx.checkpoint)
                    except Exception:
                        pass
                ctx.state = CorrectionState.DONE

        if ctx.state != CorrectionState.DONE:
            ctx.error = ctx.error or "Safety step ceiling reached — bounded safety triggered"
            if ctx.checkpoint and self.checkpoint_manager:
                try:
                    self.checkpoint_manager.rollback(ctx.checkpoint)
                except Exception:
                    pass
            ctx.state = CorrectionState.DONE

        return ctx
