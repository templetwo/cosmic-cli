"""Mandatory Action Gateway — authorize → receipt → execute.

PAUSE invariant (one named thing, three surfaces closed):
  **Exactly one action, exactly once.**

  Tokens are action-bound (sha of canonical binding) and consumed under an
  exclusive flock so double-fire, non-atomic load→set→persist races, and
  dual-process burns cannot mint two executions from one approval.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from cosmic_cli.checkpoint import CheckpointManager, CheckpointManifest
from cosmic_cli.policy import ActionType, Disposition, PolicyDecision

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows
    fcntl = None  # type: ignore


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

    @staticmethod
    def _verify_and_seal_content(verify_path: Path, expected_content: bytes) -> None:
        """Verify written bytes match approval, re-check once, then seal read-only.

        Box 4b / conformance: after verify, a late writer must not stick.
        Seal (chmod u-w) is the portable in-process half of
        reap→snapshot→seal→verify→promote for bridge executors later.
        """
        exp = hashlib.sha256(expected_content).hexdigest()

        def _read_hex() -> str:
            try:
                actual = verify_path.read_bytes() if verify_path.is_file() else b""
            except OSError as e:
                raise PermissionError(
                    f"post-exec content verify failed (unreadable): {e}"
                ) from e
            return hashlib.sha256(actual).hexdigest()

        got = _read_hex()
        if exp != got:
            raise PermissionError(
                "post-exec content mismatch — written bytes differ from "
                "approved content (truncated-hash / swap prevented)"
            )
        # Re-verify (settle) — catches a writer that raced the first check.
        got2 = _read_hex()
        if exp != got2:
            raise PermissionError(
                "post-exec content mismatch on re-verify — non-quiescent write"
            )
        # Seal: promote to read-only so post-return late writes cannot stick.
        try:
            mode = verify_path.stat().st_mode
            verify_path.chmod(mode & ~0o222)  # drop write bits
        except OSError:
            pass  # best-effort; re-verify already ran

    def authorize(
        self,
        action_type: ActionType,
        action_input: str,
        policy_rules: list,
        intended_paths: Optional[List[Path]] = None,
        executor_name: str = "unknown",
        approval_token_id: Optional[str] = None,
        *,
        match_input: Optional[str] = None,
    ) -> AuthorizationReceipt:
        """Authorize an action.

        ``action_input`` is the **commitment** string (canonical binding);
        its sha256 is what PAUSE tokens bind to.

        ``match_input`` (optional) is the corpus policy patterns scan. When
        set, rules match against match_input but the token still binds to
        sha256(action_input) — so content rules fire while EDIT/WRITE
        bindings stay length/hash-safe.
        """
        from dataclasses import replace as dc_replace

        corpus = match_input if match_input is not None else action_input
        decision: PolicyDecision = self.policy_evaluator(
            policy_rules, action_type, corpus
        )
        # Token/commitment always over action_input (binding), not the match corpus.
        if match_input is not None:
            commitment = hashlib.sha256(action_input.encode("utf-8")).hexdigest()
            decision = dc_replace(decision, evaluated_input_sha256=commitment)

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
            if self.approval_manager is None:
                raise PermissionError(
                    "PAUSE requires an ApprovalManager — refuse without token validation"
                )
            if not approval_token_id:
                raise PermissionError("PAUSE requires valid approval token")
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
        """Atomic single-use claim: exactly one action, exactly once."""
        if receipt.disposition != Disposition.PAUSE:
            return
        if self.approval_manager is None:
            raise PermissionError(
                "PAUSE requires an ApprovalManager — refuse without token consume"
            )
        if not receipt.approval_token_id:
            raise PermissionError("PAUSE requires valid approval token")
        # claim_once is transactional (flock + used=0 CAS).
        if not self.approval_manager.claim_once(
            receipt.approval_token_id, receipt.action_sha256
        ):
            raise PermissionError(
                "PAUSE approval already consumed or invalid "
                "(exactly-once invariant)"
            )

    def execute_with_receipt(
        self,
        receipt: AuthorizationReceipt,
        executor_fn: Callable[[], Any],
        observed_paths: Optional[Iterable[Path]] = None,
        *,
        observed_paths_fn: Optional[Callable[[], Iterable[Path]]] = None,
        expected_content: Optional[bytes] = None,
        verify_path: Optional[Path] = None,
    ) -> Any:
        stored = self._receipts.get(receipt.receipt_id)
        if stored is None:
            raise PermissionError("Receipt not minted by this gateway")
        if stored.consumed:
            raise PermissionError("Receipt already consumed")
        if stored.executor != receipt.executor:
            raise PermissionError("Executor identity mismatch")

        self.commit_pause_token(stored)

        checkpoint = stored.checkpoint_manifest

        try:
            result = executor_fn()

            if expected_content is not None and verify_path is not None:
                self._verify_and_seal_content(verify_path, expected_content)

            if checkpoint and self.checkpoint_manager:
                if observed_paths_fn is not None:
                    observed = [Path(p) for p in observed_paths_fn()]
                elif observed_paths is not None:
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
    """Exactly-once, action-bound PAUSE tokens.

    All store mutations (mint / validate / claim_once) hold an exclusive flock
    on ``store_path.lock`` around load→mutate→persist so concurrent processes
    cannot double-spend a token.
    """

    def __init__(self, store_path: Optional[Path] = None):
        self._store_path = Path(
            store_path
            if store_path is not None
            else Path.home() / ".cosmic-cli" / "local_approvals.json"
        )
        self._lock_path = self._store_path.with_suffix(
            self._store_path.suffix + ".lock"
        )
        self._tokens: Dict[str, dict] = {}

    def _atomic_lock_available(self) -> bool:
        """fcntl flock is required — never silently unlock (box 4a / conformance)."""
        return fcntl is not None

    def _lock_file(self):
        if not self._atomic_lock_available():
            raise ApprovalStoreError(
                "fcntl unavailable — refuse non-atomic approval store "
                "(exactly-once requires exclusive lock)"
            )
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._lock_path.parent, 0o700)
        except OSError:
            pass
        fh = open(self._lock_path, "a+", encoding="utf-8")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return fh

    def _unlock_file(self, fh) -> None:
        try:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()

    def _load_unlocked(self) -> None:
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

    def _persist_unlocked(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self._store_path.parent, 0o700)
            except OSError:
                pass
            tmp = self._store_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._tokens, indent=2), encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(self._store_path)
            try:
                os.chmod(self._store_path, 0o600)
            except OSError:
                pass
        except Exception as e:
            raise ApprovalStoreError(
                f"cannot write approval store {self._store_path}: {e}"
            ) from e

    def mint_token(self, action_sha256: str, ttl_seconds: int = 300) -> str:
        fh = self._lock_file()
        try:
            self._load_unlocked()
            token_id = f"tok-{secrets.token_hex(8)}"
            self._tokens[token_id] = {
                "action_sha256": action_sha256,
                "expiry": time.time() + ttl_seconds,
                "used": False,
            }
            self._persist_unlocked()
            return token_id
        finally:
            self._unlock_file(fh)

    def validate(self, token_id: str, current_action_sha256: str) -> bool:
        """Check token is valid for this action without consuming it (under lock)."""
        fh = self._lock_file()
        try:
            self._load_unlocked()
            tok = self._tokens.get(token_id)
            if not tok:
                return False
            if tok.get("used") or time.time() > float(tok.get("expiry", 0)):
                return False
            if tok.get("action_sha256") != current_action_sha256:
                return False
            return True
        finally:
            self._unlock_file(fh)

    def claim_once(self, token_id: str, current_action_sha256: str) -> bool:
        """Transactional consume: CAS used=false→true under exclusive flock.

        This is the single root of the exactly-once invariant — closes
        double-fire, burn races, and non-atomic load→set→persist.

        If fcntl is unavailable, refuses (returns False) rather than racing
        unlocked — box 4a / conformance suite.
        """
        if not self._atomic_lock_available():
            return False
        fh = self._lock_file()
        try:
            self._load_unlocked()
            tok = self._tokens.get(token_id)
            if not tok:
                return False
            if tok.get("used"):
                return False
            if time.time() > float(tok.get("expiry", 0)):
                return False
            if tok.get("action_sha256") != current_action_sha256:
                # Wrong action: burn so the token cannot be retried elsewhere
                tok["used"] = True
                self._persist_unlocked()
                return False
            tok["used"] = True
            self._persist_unlocked()
            return True
        finally:
            self._unlock_file(fh)

    def consume(self, token_id: str) -> bool:
        """Legacy: mark used without re-checking action binding.

        Prefer ``claim_once`` for the exactly-once path. Kept for callers that
        already validated under the same process.
        """
        fh = self._lock_file()
        try:
            self._load_unlocked()
            tok = self._tokens.get(token_id)
            if not tok or tok.get("used"):
                return False
            tok["used"] = True
            self._persist_unlocked()
            return True
        finally:
            self._unlock_file(fh)

    def validate_and_consume(self, token_id: str, current_action_sha256: str) -> bool:
        """One-shot validate+claim (exactly-once)."""
        return self.claim_once(token_id, current_action_sha256)

    def present_for_witness(self, decision: "PolicyDecision", action_input: str) -> dict:
        return {
            "disposition": decision.disposition.value,
            "matched_rules": [m.rule.rule_id for m in decision.matches],
            "action_preview": action_input[:200],
            "requires_human_judgment": True,
        }
