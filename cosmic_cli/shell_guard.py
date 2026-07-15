"""Shell safety for Stargazer (audit C1/H1/H2)."""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Substrings matched case-insensitively in safe mode.
DANGEROUS_SUBSTR = (
    "rm -rf",
    "rm -r ",
    "rm -fr",
    "sudo ",
    " mkfs",
    "mkfs.",
    "dd if=",
    "> /dev",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "shutdown",
    "reboot",
    "diskutil erase",
    "git push --force",
    "git push -f",
    "git reset --hard",
    "git clean -fd",
    "git clean -f",
    "find -delete",
    "find . -delete",
    "> ~/.ssh",
    "authorized_keys",
    "drop table",
    "drop database",
)

# Network / exfil verbs — blocked in safe mode (H2 partial).
NETWORK_SUBSTR = (
    "curl ",
    "curl\t",
    "wget ",
    "wget\t",
    " scp ",
    "scp ",
    "rsync ",
    "nc ",
    "ncat ",
    "ssh ",
    "ftp ",
    "sftp ",
    "http://",
    "https://",
)

RM_RECURSIVE = re.compile(
    r"\brm\b.*(-[a-zA-Z]*[rf]|--recursive)",
    re.IGNORECASE,
)


def check_shell(cmd: str, *, exec_mode: str = "safe") -> Optional[str]:
    """Return a block message if command must not run, else None."""
    if exec_mode == "full":
        return None
    cmd_l = cmd.lower()
    for d in DANGEROUS_SUBSTR:
        if d.lower() in cmd_l:
            return f"[BLOCKED] dangerous pattern in {exec_mode} mode: {d!r}"
    if RM_RECURSIVE.search(cmd):
        return f"[BLOCKED] recursive rm blocked in {exec_mode} mode"
    if exec_mode == "safe":
        for n in NETWORK_SUBSTR:
            if n.lower() in cmd_l:
                return (
                    f"[BLOCKED] network/exfil verb blocked in safe mode: {n.strip()!r}. "
                    "Use --mode interactive or full if intentional."
                )
    return None


def gate_shell(cmd: str, *, exec_mode: str = "safe") -> Tuple[bool, str]:
    """(allowed, message)."""
    msg = check_shell(cmd, exec_mode=exec_mode)
    if msg:
        return False, msg
    return True, "ok"
