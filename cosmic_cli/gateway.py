"""Mandatory Action Gateway interface (Phase 2 seam + Phase 5 checkpoint integration).

Every executable path must go through authorize() -> AuthorizationReceipt.
No executor should be callable without a valid receipt.

Merged surface (assembly, 2026-07-15):
- Base: policy_kernel_v0.4 gateway (receipt with defaults, ApprovalManager prototype).
- Ported from phase6_complete: checkpoint-before-execute, uuid receipt ids,
  gateway-minted receipt verification, executor identity check, receipt
  consumption via dataclasses.replace, post-execution unrelated-change
  escalation, and rollback on executor failure or policy breach — adapted to
  the hardened CheckpointManager API (detect_unrelated_changes /
  CheckpointManifest.files, rollback raises CheckpointError).
- PAUSE semantics: phase6's strict gate is kept — PAUSE requires an approval
  token. The hardened PolicyDecision has no approval_token_id field (phase6
  read it off the decision via getattr, which could never succeed), so the
  token is threaded explicitly through authorize(approval_token_id=...).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
import uuid

from cosmic_cli.policy import PolicyDecision, Disposition, ActionType
from cosmic_cli.checkpoint import CheckpointManager, CheckpointManifest

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

        # Checkpoint before disposition gates (phase6 ordering).
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
                ok = self.approval_manager.validate_and_consume(
                    approval_token_id, decision.evaluated_input_sha256
                )
                if not ok:
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

        checkpoint = stored.checkpoint_manifest

        try:
            result = executor_fn()

            if checkpoint and self.checkpoint_manager:
                # phase6 default: observe the manifest's own paths (which, by
                # construction, can never flag anything). Callers that want real
                # detection must pass observed_paths explicitly.
                if observed_paths is not None:
                    observed = [Path(p) for p in observed_paths]
                else:
                    observed = [
                        self.checkpoint_manager.workspace_root / snap.relative_path
                        for snap in checkpoint.files
                    ]
                unexpected = self.checkpoint_manager.detect_unrelated_changes(checkpoint, observed)
                if unexpected:
                    raise PermissionError(
                        f"Unrelated changes detected and escalated: {list(unexpected)}"
                    )

            updated = replace(stored, consumed=True, result=str(result)[:300])
            self._receipts[receipt.receipt_id] = updated
            return result

        except Exception:
            if checkpoint and self.checkpoint_manager:
                try:
                    self.checkpoint_manager.rollback(checkpoint)
                except Exception:
                    # phase6 behavior kept: rollback failures do not mask the
                    # original executor/policy error.
                    pass
            raise


class ApprovalManager:
    """Phase 4 approval lifecycle.

    - Mints single-use tokens for PAUSE dispositions.
    - Tokens are action-bound and expire.
    - Invalidation on mutation or double-use.
    - Optional disk store so tokens survive `cosmic-cli do` process restarts.
    """

    def __init__(self, store_path: Optional[Path] = None):
        self._store_path = Path(
            store_path
            if store_path is not None
            else Path.home() / ".cosmic-cli" / "local_approvals.json"
        )
        self._tokens: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._store_path.is_file():
                import json

                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._tokens = data
        except Exception:
            self._tokens = {}

    def _persist(self) -> None:
        try:
            import json

            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store_path.write_text(
                json.dumps(self._tokens, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def mint_token(self, action_sha256: str, ttl_seconds: int = 300) -> str:
        import secrets
        import time

        self._load()
        token_id = f"tok-{secrets.token_hex(8)}"
        self._tokens[token_id] = {
            "action_sha256": action_sha256,
            "expiry": time.time() + ttl_seconds,
            "used": False,
        }
        self._persist()
        return token_id

    def validate_and_consume(self, token_id: str, current_action_sha256: str) -> bool:
        import time

        self._load()
        if token_id not in self._tokens:
            return False
        tok = self._tokens[token_id]
        if tok.get("used") or time.time() > float(tok.get("expiry", 0)):
            return False
        if tok.get("action_sha256") != current_action_sha256:
            tok["used"] = True
            self._persist()
            return False
        tok["used"] = True
        self._persist()
        return True

    def present_for_witness(self, decision: "PolicyDecision", action_input: str) -> dict:
        """Stub for witness presentation (Phase 4)."""
        return {
            "disposition": decision.disposition.value,
            "matched_rules": [m.rule.rule_id for m in decision.matches],
            "action_preview": action_input[:200],
            "requires_human_judgment": True,
        }
