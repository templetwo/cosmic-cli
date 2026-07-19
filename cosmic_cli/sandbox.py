"""L0 shell sandbox floor — deny-glob on the PAUSE token store.

Privilege ranking hierarchy (strongest → weakest):
  1. kernel isolation (this module)
  2. privilege ranking (TTY-only approval CLI)
  3. command classification (substring / decode DiD)

Classification alone is evadable (base64|sh, python -c). The floor makes
~/.cosmic-cli unreachable by L0 shell regardless of command shape.

Config: repo-root sandbox.toml (deny_globs). Enforced for cosmic-cli agent
shell (agents._run_shell) in safe/interactive modes. --mode full skips.
"""

from __future__ import annotations

import os
import platform
import shlex
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DENY = ("~/.cosmic-cli",)


def token_store_dir() -> Path:
    return Path.home() / ".cosmic-cli"


def load_deny_paths() -> List[Path]:
    """Resolve deny globs to concrete directory paths (files' parents ok)."""
    globs = list(_DEFAULT_DENY)
    cfg = _PKG_ROOT / "sandbox.toml"
    if cfg.is_file():
        try:
            text = cfg.read_text(encoding="utf-8")
            in_globs = False
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("deny_globs"):
                    in_globs = True
                    continue
                if in_globs:
                    if s.startswith("["):
                        break
                    if s.startswith('"') or s.startswith("'"):
                        # "path", or "path"
                        raw = s.strip().rstrip(",").strip().strip('"').strip("'")
                        if raw and not raw.endswith("/**"):
                            globs.append(raw)
                        elif raw.endswith("/**"):
                            globs.append(raw[:-3])
        except OSError:
            pass
    out: List[Path] = []
    seen = set()
    for g in globs:
        p = Path(os.path.expanduser(g.split("/**")[0])).resolve()
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def seatbelt_profile(deny_paths: Optional[Sequence[Path]] = None) -> str:
    """macOS Seatbelt profile: allow default, deny read/write under token store."""
    paths = list(deny_paths) if deny_paths is not None else load_deny_paths()
    lines = [
        "(version 1)",
        "(allow default)",
        "; Cosmic privilege-ranking floor: L0 must not reach PAUSE token store",
    ]
    for p in paths:
        sp = str(p)
        # Escape backslashes for seatbelt string
        sp = sp.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'(deny file-read* (subpath "{sp}"))')
        lines.append(f'(deny file-write* (subpath "{sp}"))')
        lines.append(f'(deny file-read-metadata (subpath "{sp}"))')
    return "\n".join(lines) + "\n"


def write_seatbelt_profile(path: Optional[Path] = None) -> Path:
    dest = path or Path(tempfile.mkstemp(prefix="cosmic-seatbelt-", suffix=".sb")[1])
    dest.write_text(seatbelt_profile(), encoding="utf-8")
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass
    return dest


def sandbox_available() -> bool:
    if platform.system() == "Darwin":
        return Path("/usr/bin/sandbox-exec").is_file() or bool(
            _which("sandbox-exec")
        )
    return False


def _which(name: str) -> Optional[str]:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def wrap_argv_for_l0_shell(cmd: str, *, shell: str = "/bin/zsh") -> List[str]:
    """Return argv to run *cmd* under the L0 floor when available.

    On macOS: sandbox-exec -f profile shell -c cmd
    Elsewhere: bare shell -c cmd (classification remains DiD; floor is macOS-first).
    """
    shell_bin = shell if Path(shell).is_file() else ( _which("zsh") or _which("bash") or "/bin/sh")
    if platform.system() == "Darwin" and sandbox_available():
        profile = write_seatbelt_profile()
        exe = _which("sandbox-exec") or "/usr/bin/sandbox-exec"
        return [exe, "-f", str(profile), shell_bin, "-c", cmd]
    return [shell_bin, "-c", cmd]


def describe_floor() -> str:
    paths = ", ".join(str(p) for p in load_deny_paths())
    backend = "sandbox-exec/Seatbelt" if sandbox_available() else "none (classification+TTY only)"
    return f"L0 sandbox floor backend={backend}; deny=[{paths}]"
