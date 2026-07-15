"""Bounded self-correction state machine (avionics Phase 6).

Assembled from phase6_complete, ported to hardened kernel/gateway/checkpoint
APIs. Iteration-cap bug fixed on adoption (2026-07-15): happy path is 6
transitions; the bound is a safety ceiling, not the happy-path length.
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
    iteration: int = 0
    # Happy path needs 6 transitions; 32 is a thrash bound, not a goal.
    max_iterations: int = 32


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
    ) -> CorrectionContext:
        ctx = CorrectionContext(plan=initial_plan, intended_paths=intended_paths)

        while ctx.state != CorrectionState.DONE and ctx.iteration < ctx.max_iterations:
            ctx.iteration += 1

            if ctx.state == CorrectionState.INTAKE:
                ctx.state = CorrectionState.PLAN

            elif ctx.state == CorrectionState.PLAN:
                ctx.state = CorrectionState.REVIEW

            elif ctx.state == CorrectionState.REVIEW:
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
            ctx.error = "Max iterations reached — bounded safety triggered"
            if ctx.checkpoint and self.checkpoint_manager:
                try:
                    self.checkpoint_manager.rollback(ctx.checkpoint)
                except Exception:
                    pass
            ctx.state = CorrectionState.DONE

        return ctx
