"""Mandatory Action Gateway — authorize → receipt → execute.

PAUSE tokens are validated at authorize and consumed only when the action
is fully allowed (execute_with_receipt, or explicit commit after later gates).
Consuming at authorize burned tokens when a later gate (e.g. check_shell)
still blocked — Claude live-fire 2026-07-15.
"""

from __future__ import annotations

import json
import secrets
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from cosmic_cli.checkpoint import CheckpointError, CheckpointManager, CheckpointManifest
from cosmic_cli.policy import ActionType, Disposition, PolicyDecision


class ApprovalStoreError(RuntimeError):
    """Raised when the approval token store cannot be read or written."""


class RollbackFailedError(RuntimeError):
    """Raised when post-failure rollback itself fails (workspace may be dirty)."""


@dataclass(frozen=True)
class AuthorizationReceipt:
    receipt_id: str
    action_sha256: str
    policy_sha256: str
    disposition: Disposition
    matched_rule_ids: tuple[str, ...]
    checkpoint_id: Optional[str] = None
    checkpoint_manifest: Optional[CheckpointManifest] = None
    approved_by: str | None = None
    approval_token_id: str | None = None
    consumed: bool = False
    executor: str = ""
    result: str = ""


class ActionGateway:
    """The single unavoidable seam for authorize → receipt → execute."""

    def __init__(
        self,
        policy_evaluator: Callable,
        checkpoint_manager: Optional[CheckpointManager] = None,
        approval_manager: Optional["ApprovalManager"] = None,
    ):
        self.policy_evaluator = policy_evaluator
        self.checkpoint_manager = checkpoint_manager
        self.approval_manager = approval_manager
        self._receipts: Dict[str, AuthorizationReceipt] = {}

    def _mint_receipt_id(self) -> str:
        return f"rcpt-{uuid.uuid4().hex[:16]}"

    def authorize(
        self,
        action_type: ActionType,
        action_input: str,
        policy_rules: list,
        intended_paths: Optional[List[Path]] = None,
        executor_name: str = "unknown",
        approval_token_id: Optional[str] = None,
    ) -> AuthorizationReceipt:
        decision: PolicyDecision = self.policy_evaluator(
            policy_rules, action_type, action_input
        )

        checkpoint: Optional[CheckpointManifest] = None
        if self.checkpoint_manager and intended_paths:
            checkpoint = self.checkpoint_manager.create_checkpoint(intended_paths)

        if decision.disposition == Disposition.WITNESS:
            raise PermissionError(
                f"WITNESS: action blocked by policy. "
                f"Matched: {[m.rule.rule_id for m in decision.matches]}. "
                f"Checkpoint: {checkpoint.checkpoint_id if checkpoint else 'none'}"
            )

        if decision.disposition == Disposition.PAUSE:
            if not approval_token_id:
                raise PermissionError("PAUSE requires valid approval token")
            if self.approval_manager is not None:
                # Validate only — consume at execute/commit so later gates
                # (check_shell, helix) cannot burn a single-use token.
                if not self.approval_manager.validate(
                    approval_token_id, decision.evaluated_input_sha256
                ):
                    raise PermissionError(
                        "PAUSE approval token invalid, expired, or already used"
                    )

        receipt = AuthorizationReceipt(
            receipt_id=self._mint_receipt_id(),
            action_sha256=decision.evaluated_input_sha256,
            policy_sha256=decision.policy_sha256,
            disposition=decision.disposition,
            matched_rule_ids=tuple(m.rule.rule_id for m in decision.matches),
            checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
            checkpoint_manifest=checkpoint,
            approval_token_id=approval_token_id,
            executor=executor_name,
        )
        self._receipts[receipt.receipt_id] = receipt
        return receipt

    def commit_pause_token(self, receipt: AuthorizationReceipt) -> None:
        """Consume a PAUSE token after all gates have allowed the action."""
        if receipt.disposition != Disposition.PAUSE:
            return
        if not receipt.approval_token_id or self.approval_manager is None:
            return
        if not self.approval_manager.consume(receipt.approval_token_id):
            raise PermissionError(
                "PAUSE approval token could not be consumed (already used or store failed)"
            )

    def execute_with_receipt(
        self,
        receipt: AuthorizationReceipt,
        executor_fn: Callable[[], Any],
        observed_paths: Optional[Iterable[Path]] = None,
    ) -> Any:
        stored = self._receipts.get(receipt.receipt_id)
        if stored is None:
            raise PermissionError("Receipt not minted by this gateway")
        if stored.consumed:
            raise PermissionError("Receipt already consumed")
        if stored.executor != receipt.executor:
            raise PermissionError("Executor identity mismatch")

        # Consume PAUSE only when we are about to run for real.
        self.commit_pause_token(stored)

        checkpoint = stored.checkpoint_manifest

        try:
            result = executor_fn()

            if checkpoint and self.checkpoint_manager:
                if observed_paths is not None:
                    observed = [Path(p) for p in observed_paths]
                else:
                    observed = [
                        self.checkpoint_manager.workspace_root / snap.relative_path
                        for snap in checkpoint.files
                    ]
                unexpected = self.checkpoint_manager.detect_unrelated_changes(
                    checkpoint, observed
                )
                if unexpected:
                    raise PermissionError(
                        f"Unrelated changes detected and escalated: {list(unexpected)}"
                    )

            updated = replace(stored, consumed=True, result=str(result)[:300])
            self._receipts[receipt.receipt_id] = updated
            return result

        except Exception as original:
            if checkpoint and self.checkpoint_manager:
                try:
                    self.checkpoint_manager.rollback(checkpoint)
                except Exception as rb_err:
                    raise RollbackFailedError(
                        f"executor failed ({original!r}) AND rollback failed "
                        f"({rb_err!r}) — workspace may be inconsistent"
                    ) from original
            raise


class ApprovalManager:
    """Single-use, action-bound PAUSE tokens with fail-closed persistence."""

    def __init__(self, store_path: Optional[Path] = None):
        self._store_path = Path(
            store_path
            if store_path is not None
            else Path.home() / ".cosmic-cli" / "local_approvals.json"
        )
        self._tokens: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._store_path.is_file():
            self._tokens = {}
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ApprovalStoreError("approval store is not a JSON object")
            self._tokens = data
        except ApprovalStoreError:
            raise
        except Exception as e:
            raise ApprovalStoreError(
                f"cannot read approval store {self._store_path}: {e}"
            ) from e

    def _persist(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._store_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._tokens, indent=2), encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception as e:
            raise ApprovalStoreError(
                f"cannot write approval store {self._store_path}: {e}"
            ) from e

    def mint_token(self, action_sha256: str, ttl_seconds: int = 300) -> str:
        self._load()
        token_id = f"tok-{secrets.token_hex(8)}"
        self._tokens[token_id] = {
            "action_sha256": action_sha256,
            "expiry": time.time() + ttl_seconds,
            "used": False,
        }
        self._persist()
        return token_id

    def validate(self, token_id: str, current_action_sha256: str) -> bool:
        """Check token is valid for this action without consuming it."""
        self._load()
        tok = self._tokens.get(token_id)
        if not tok:
            return False
        if tok.get("used") or time.time() > float(tok.get("expiry", 0)):
            return False
        if tok.get("action_sha256") != current_action_sha256:
            return False
        return True

    def consume(self, token_id: str) -> bool:
        """Mark token used. Fails closed if store cannot persist."""
        self._load()
        tok = self._tokens.get(token_id)
        if not tok or tok.get("used"):
            return False
        tok["used"] = True
        self._persist()
        return True

    def validate_and_consume(self, token_id: str, current_action_sha256: str) -> bool:
        """Legacy one-shot: validate then consume (fail-closed)."""
        if not self.validate(token_id, current_action_sha256):
            # Mutation / wrong action — burn if present
            self._load()
            if token_id in self._tokens:
                self._tokens[token_id]["used"] = True
                self._persist()
            return False
        return self.consume(token_id)

    def present_for_witness(self, decision: "PolicyDecision", action_input: str) -> dict:
        return {
            "disposition": decision.disposition.value,
            "matched_rules": [m.rule.rule_id for m in decision.matches],
            "action_preview": action_input[:200],
            "requires_human_judgment": True,
        }
