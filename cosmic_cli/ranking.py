"""Privilege ranking — approval flows down, never sideways or up.

Anthony's law (2026-07-19): approval of an action at level N must be issued
from a level strictly greater than N, through a channel no actor at level ≤ N
can reach. No actor approves its own action or its own level.

Ranking (high → low):
  L3 human owner (Anthony) — push, deposit, stack-policy enactment
  L2 reviewer / operator at a TTY — PAUSE approval, code review
  L1 cosmic-cli / compass — gates L0; cannot self-approve governance
  L0 gated model / agent shell — acts, approves nothing

Role sets the level, not identity: the same seat is L2 reviewing and L0 when
driving a gated shell. The PAUSE approval channel must be L2-only.

This module is the structural hinge for the PAUSE channel — not a patch on
accept-pause alone. See COSMIC.md § Ranking and PRIVILEGE_RANKING.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Substrings that mean "touching the L2 approval surface from L0 shell".
# Matched case-insensitively. Always blocked except --mode full (L2/L3 blast).
APPROVAL_SURFACE_SUBSTR = (
    "accept-pause",
    "show-pause-token",
    "helix confirm",
    "last_pause_token",
    "operator_approval_token",
    "local_approvals.json",
    "cosmic_approval_token=",
    "cosmic_approval_token =",
)

# Filenames the L0 actor must never READ/WRITE via tool doors.
APPROVAL_TOKEN_BASENAMES = frozenset(
    {
        "last_pause_token.json",
        "operator_approval_token",
        "local_approvals.json",
        "local_approvals.json.lock",
    }
)


def is_interactive_l2() -> bool:
    """True when a human/operator TTY is present (L2+ channel).

    Non-interactive agent shells (piped stdin, headless PreToolUse children,
    CI) are L0 and must not reach the approval surface.
    """
    for stream in (sys.stdin, sys.stderr, sys.stdout):
        try:
            if stream is not None and stream.isatty():
                return True
        except Exception:
            continue
    # Explicit L2/L3 break-glass for scripted operator seats that own the risk.
    if os.environ.get("COSMIC_L2_OPERATOR", "").strip() in ("1", "true", "yes"):
        return True
    return False


def require_l2_tty(action: str) -> Optional[str]:
    """Return a deny reason if *action* is not allowed at L0; None if L2 ok."""
    if is_interactive_l2():
        return None
    return (
        f"[BLOCKED] privilege ranking: {action} is L2-only "
        "(interactive TTY / COSMIC_L2_OPERATOR). "
        "L0 shells cannot approve their own level."
    )


def touches_approval_surface(command: str) -> Optional[str]:
    """If shell/code corpus touches the PAUSE approval channel, return reason."""
    if not command:
        return None
    low = command.lower()
    for pat in APPROVAL_SURFACE_SUBSTR:
        if pat in low:
            return (
                f"[BLOCKED] privilege ranking: approval surface {pat!r} is L2-only "
                "(L1 WITNESS). L0 cannot reach accept-pause / token files / "
                "COSMIC_APPROVAL_TOKEN."
            )
    return None


def is_approval_token_path(path: str) -> bool:
    """True for PAUSE token store paths (must not enter L0 READ/WRITE tools)."""
    if not path:
        return False
    p = Path(path)
    name = p.name.lower()
    if name in APPROVAL_TOKEN_BASENAMES:
        return True
    # Any path under ~/.cosmic-cli that looks like an approval artifact
    parts = [x.lower() for x in p.parts]
    if ".cosmic-cli" in parts and (
        "token" in name or "approval" in name or name.endswith(".json.lock")
    ):
        return True
    return False
