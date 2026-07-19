"""Privilege ranking — approval flows down, never sideways or up.

Anthony's law (2026-07-19): approval of an action at level N must be issued
from a level strictly greater than N, through a channel no actor at level ≤ N
can reach. No actor approves its own action or its own level.

Ranking (high → low):
  L3 human owner (Anthony) — push, deposit, stack-policy enactment
  L2 reviewer / operator at a TTY — PAUSE approval, code review
  L1 cosmic-cli / compass — gates L0; cannot self-approve governance
  L0 gated model / agent shell — acts, approves nothing

Hierarchy of enforcement (strongest → weakest):
  1. kernel isolation — sandbox.toml deny-glob on ~/.cosmic-cli (sandbox.py)
  2. privilege ranking — TTY-only approval CLI (this module)
  3. command classification — substring + decode expansion (DiD, known-evadable
     alone; kept so the naive path fails closed even without a kernel floor)

Role sets the level, not identity. See COSMIC.md § Ranking, PRIVILEGE_RANKING.md.
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path
from typing import List, Optional

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
    ".cosmic-cli/",
    ".cosmic-cli'",
    '.cosmic-cli"',
)

# Opaque / decoder wrappers that hide the real command from substring DiD.
# Fail-closed in safe+interactive: L0 must not run uninspectable payloads.
# --mode full skips (L2/L3 blast-radius opt-in).
_OPAQUE_SHELL = re.compile(
    r"("
    r"base64\s+(-[dD]|--decode)|"
    r"\|\s*(sh|bash|zsh|dash)\b|"
    r"\b(sh|bash|zsh)\s+-c\b|"
    r"python3?\s+-c\b|"
    r"perl\s+-e\b|"
    r"ruby\s+-e\b|"
    r"node\s+(-e|--eval)\b|"
    r"\beval\b|"
    r"openssl\s+enc\s+-d|"
    r"xxd\s+-r|"
    r"/dev/stdin|"
    r"process\.popen|"
    r"os\.system\s*\(|"
    r"subprocess\.(run|call|popen)|"
    r"commands\.getoutput"
    r")",
    re.IGNORECASE,
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

_B64_BLOB = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def is_interactive_l2() -> bool:
    """True when a human/operator TTY is present (L2+ channel).

    Non-interactive agent shells (piped stdin, headless PreToolUse children,
    CI) are L0 and must not reach the approval surface.

    There is NO env-var break-glass. A flag L0 can set is an L2 credential
    reachable by L0 — the same ranking bug as self-approve. Scripted L2 seats
    must attach a real TTY (ssh -t, script(1), or a human console).
    """
    for stream in (sys.stdin, sys.stderr, sys.stdout):
        try:
            if stream is not None and stream.isatty():
                return True
        except Exception:
            continue
    return False


def require_l2_tty(action: str) -> Optional[str]:
    """Return a deny reason if *action* is not allowed at L0; None if L2 ok."""
    if is_interactive_l2():
        return None
    return (
        f"[BLOCKED] privilege ranking: {action} is L2-only "
        "(interactive TTY required). "
        "L0 shells cannot approve their own level. "
        "No env break-glass — that would be an L2 credential L0 can set."
    )


def _decoded_fragments(command: str) -> List[str]:
    """Best-effort base64 expansion so hidden payloads still hit DiD scan."""
    frags: List[str] = []
    for m in _B64_BLOB.finditer(command):
        blob = m.group(0)
        # Pad
        pad = (-len(blob)) % 4
        try:
            raw = base64.b64decode(blob + ("=" * pad), validate=False)
        except Exception:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not text or not any(c.isalpha() for c in text):
            continue
        frags.append(text)
    return frags


def classification_corpus(command: str) -> str:
    """Command plus decoded fragments — input to surface / opaque checks."""
    if not command:
        return ""
    parts = [command]
    parts.extend(_decoded_fragments(command))
    return "\n".join(parts)


def is_opaque_shell_wrapper(command: str) -> Optional[str]:
    """Fail-closed on decoder/interpreter wrappers that hide the real action."""
    if not command:
        return None
    if _OPAQUE_SHELL.search(command):
        return (
            "[BLOCKED] privilege ranking: opaque shell wrapper "
            "(base64|sh / interpreter -c / eval). "
            "Classification cannot see through these; L0 must not use them. "
            "Use --mode full only as an L2/L3 blast-radius opt-in."
        )
    return None


def touches_approval_surface(command: str) -> Optional[str]:
    """If shell/code corpus touches the PAUSE approval channel, return reason.

    Scans the raw command and base64-decoded fragments (DiD). Kernel sandbox
    is the load-bearing floor; this is the known-partial classifier layer.
    """
    if not command:
        return None
    opaque = is_opaque_shell_wrapper(command)
    if opaque:
        return opaque
    corpus = classification_corpus(command).lower()
    for pat in APPROVAL_SURFACE_SUBSTR:
        if pat in corpus:
            return (
                f"[BLOCKED] privilege ranking: approval surface {pat!r} is L2-only "
                "(L1 WITNESS / DiD). L0 cannot reach accept-pause / token files / "
                "COSMIC_APPROVAL_TOKEN. Kernel floor: sandbox.toml deny ~/.cosmic-cli."
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
    parts = [x.lower() for x in p.parts]
    if ".cosmic-cli" in parts and (
        "token" in name or "approval" in name or name.endswith(".json.lock")
    ):
        return True
    return False
