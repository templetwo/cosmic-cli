"""Operator PAUSE approve/decline. Privileged; UI never sees token bytes.

Approve stages a one-retry credential (operator_approval_token) and does
not claim_once. Consume happens on the retry, same as helix accept-pause.
Selection is by action_sha256. last_pause_token.json is not the selector
when more than one action is pending.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from cosmic_cli.gateway import ApprovalManager, ApprovalStoreError
from cosmic_cli.ranking import require_l2_tty

PauseOutcome = Literal[
    "approved",
    "declined",
    "expired",
    "invalid",
    "not_found",
    "ambiguous",
    "ranking_denied",
]
PauseChannel = Literal["local", "gate", "helix"]

STAGE_PATH = Path.home() / ".cosmic-cli" / "operator_approval_token"


def load_staged_token(stage_path: Optional[Path] = None) -> Optional[str]:
    """Read the one-retry stage file. Empty/missing → None. Never logs the value."""
    path = stage_path if stage_path is not None else STAGE_PATH
    try:
        if not path.is_file():
            return None
        tok = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return tok or None


@dataclass(frozen=True)
class PauseHandle:
    """Bus/UI-safe. No token, no tok- prefix, no Helix hex credential."""

    action_sha256: str
    action_summary: str = ""
    channel: PauseChannel = "local"
    session_id: Optional[str] = None
    mission_id: Optional[str] = None
    pending_id: Optional[int] = None
    expires_at: Optional[str] = None
    rule_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class PauseResolution:
    outcome: PauseOutcome
    handle: PauseHandle
    by: Optional[Literal["operator"]] = None
    staged_for_gate: bool = False
    message: str = ""
    # In-process Stargazer retry ONLY. Never copy onto widgets/events/logs.
    approval_token_id: Optional[str] = None


def write_stage_file(token: str, dest: Path = STAGE_PATH) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(token + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(dest)
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass


def _ranking(action: str, *, require_tty: bool) -> Optional[PauseResolution]:
    if not require_tty:
        return None
    blocked = require_l2_tty(action)
    if not blocked:
        return None
    return PauseResolution(
        outcome="ranking_denied",
        handle=PauseHandle(action_sha256=""),
        message=blocked,
    )


def resolve_action_sha256(
    query: str = "",
    *,
    manager: Optional[ApprovalManager] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Pick a pending action binding. Concurrent pauses require an explicit sha.

    Does not read last_pause_token.json. Returns (sha, error).
    """
    mgr = manager or ApprovalManager()
    unused = mgr.unused_action_shas()
    q = (query or "").strip().lower()
    if q:
        matches = [s for s in unused if s.lower() == q or s.lower().startswith(q)]
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, "no unused token for that action_sha256"
        return None, "ambiguous action_sha256 prefix"
    if len(unused) == 1:
        return unused[0], None
    if not unused:
        return None, "nothing to accept"
    return (
        None,
        f"{len(unused)} pending PAUSE actions; pass action_sha256 "
        "(do not use last_pause_token.json as the selector)",
    )


def approve_pause(
    handle: PauseHandle,
    *,
    manager: Optional[ApprovalManager] = None,
    stage_path: Path = STAGE_PATH,
    require_tty: bool = True,
    stage_file: bool = True,
) -> PauseResolution:
    ranked = _ranking("helix accept-pause", require_tty=require_tty)
    if ranked:
        return PauseResolution(
            outcome="ranking_denied",
            handle=handle,
            message=ranked.message,
        )
    sha = (handle.action_sha256 or "").strip()
    if not sha:
        return PauseResolution(
            outcome="invalid",
            handle=handle,
            message="PAUSE handle missing action_sha256",
        )
    mgr = manager or ApprovalManager()
    try:
        token = mgr.peek_unused(sha)
    except ApprovalStoreError as e:
        return PauseResolution(
            outcome="invalid",
            handle=handle,
            message=str(e),
        )
    if not token:
        return PauseResolution(
            outcome="not_found",
            handle=handle,
            message="no unused token for that action",
        )
    if stage_file:
        write_stage_file(token, stage_path)
    return PauseResolution(
        outcome="approved",
        handle=handle,
        by="operator",
        staged_for_gate=stage_file,
        message="staged for one retry — re-run the blocked action",
        approval_token_id=token,
    )


def decline_pause(
    handle: PauseHandle,
    *,
    manager: Optional[ApprovalManager] = None,
    require_tty: bool = True,
) -> PauseResolution:
    ranked = _ranking("helix accept-pause", require_tty=require_tty)
    if ranked:
        return PauseResolution(
            outcome="ranking_denied",
            handle=handle,
            message=ranked.message,
        )
    sha = (handle.action_sha256 or "").strip()
    if not sha:
        return PauseResolution(
            outcome="invalid",
            handle=handle,
            message="PAUSE handle missing action_sha256",
        )
    mgr = manager or ApprovalManager()
    try:
        burned = mgr.burn_unused(sha)
    except ApprovalStoreError as e:
        return PauseResolution(
            outcome="invalid",
            handle=handle,
            message=str(e),
        )
    if not burned:
        return PauseResolution(
            outcome="not_found",
            handle=handle,
            message="no unused token to decline",
        )
    return PauseResolution(
        outcome="declined",
        handle=handle,
        by="operator",
        message="declined — mission stays blocked; no remint",
    )


def approve_helix_pending(
    handle: PauseHandle,
    *,
    token: str,
    confirm=None,
    require_tty: bool = True,
) -> PauseResolution:
    """Helix-origin PAUSE: confirm_pending is the authority, not local store."""
    ranked = _ranking("helix accept-pause", require_tty=require_tty)
    if ranked:
        return PauseResolution(
            outcome="ranking_denied",
            handle=handle,
            message=ranked.message,
        )
    tok = (token or "").strip()
    if not tok:
        return PauseResolution(
            outcome="not_found",
            handle=handle,
            message="no in-process Helix pending token for this action",
        )
    if confirm is None:
        from cosmic_cli import helix_bridge

        confirm = helix_bridge.confirm_pending
    try:
        result = confirm(tok)
    except Exception as e:
        return PauseResolution(
            outcome="invalid",
            handle=handle,
            message=str(e),
        )
    inner = (result or {}).get("result") if isinstance(result, dict) else None
    if not (
        isinstance(result, dict)
        and result.get("ok")
        and isinstance(inner, dict)
        and inner.get("ok")
    ):
        return PauseResolution(
            outcome="invalid",
            handle=handle,
            message="Helix confirm_pending refused",
        )
    return PauseResolution(
        outcome="approved",
        handle=handle,
        by="operator",
        message="Helix pending confirmed — re-run the blocked action",
    )


def accept_pause_cli(
    query: str = "",
    *,
    manager: Optional[ApprovalManager] = None,
    stage_path: Path = STAGE_PATH,
    require_tty: bool = True,
) -> PauseResolution:
    """CLI entry: unique pending sha, or an explicit sha/prefix in query."""
    ranked = _ranking("helix accept-pause", require_tty=require_tty)
    if ranked:
        return ranked
    mgr = manager or ApprovalManager()
    sha, err = resolve_action_sha256(query, manager=mgr)
    if err or not sha:
        return PauseResolution(
            outcome="ambiguous" if err and "pending PAUSE" in (err or "") else "not_found",
            handle=PauseHandle(action_sha256=query or ""),
            message=err or "nothing to accept",
        )
    return approve_pause(
        PauseHandle(action_sha256=sha),
        manager=mgr,
        stage_path=stage_path,
        require_tty=False,
        stage_file=True,
    )
