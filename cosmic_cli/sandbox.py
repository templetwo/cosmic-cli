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


# Cached probe: binary present ≠ seatbelt can apply. Nested sandboxes (Grok Build
# under Seatbelt, CI, some macOS policies) return "sandbox_apply: Operation not
# permitted" and every L0 shell would fail closed wrongly. Probe once per process.
_SEATBELT_APPLIES: Optional[bool] = None


def reset_seatbelt_probe() -> None:
    """Test hook: clear the seatbelt usability cache."""
    global _SEATBELT_APPLIES
    _SEATBELT_APPLIES = None


def mark_seatbelt_unusable() -> None:
    """Sticky: seatbelt cannot apply in this process (nested sandbox observed)."""
    global _SEATBELT_APPLIES
    _SEATBELT_APPLIES = False


def seatbelt_applies(*, force_probe: bool = False) -> bool:
    """True only when sandbox-exec can actually apply a profile here.

    Classification + TTY ranking remain when this is False; we just skip a
    kernel floor that cannot bind so missions do not hang on every SHELL.
    """
    global _SEATBELT_APPLIES
    if _SEATBELT_APPLIES is not None and not force_probe:
        return _SEATBELT_APPLIES
    if platform.system() != "Darwin" or not sandbox_available():
        _SEATBELT_APPLIES = False
        return False
    try:
        import subprocess

        profile = write_seatbelt_profile()
        exe = _which("sandbox-exec") or "/usr/bin/sandbox-exec"
        r = subprocess.run(
            [exe, "-f", str(profile), "/bin/echo", "cosmic-seatbelt-ok"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = (r.stdout or "") + (r.stderr or "")
        ok = (
            r.returncode == 0
            and "cosmic-seatbelt-ok" in (r.stdout or "")
            and "Operation not permitted" not in out
            and "sandbox_apply" not in out
        )
        _SEATBELT_APPLIES = bool(ok)
    except Exception:
        _SEATBELT_APPLIES = False
    return _SEATBELT_APPLIES


def is_sandbox_apply_failure(stderr: str = "", stdout: str = "") -> bool:
    """True when a run failed because seatbelt could not apply (nested sandbox)."""
    blob = f"{stderr or ''}\n{stdout or ''}"
    return "sandbox_apply" in blob or (
        "sandbox-exec" in blob and "Operation not permitted" in blob
    )


def wrap_argv_for_l0_shell(
    cmd: str, *, shell: str = "/bin/zsh", force_bare: bool = False
) -> List[str]:
    """Return argv to run *cmd* under the L0 floor when available and usable.

    On macOS when seatbelt applies: sandbox-exec -f profile shell -c cmd
    Elsewhere / nested-sandbox hosts: bare shell -c cmd (classification DiD
    still gates; floor is best-effort).
    """
    shell_bin = shell if Path(shell).is_file() else (_which("zsh") or _which("bash") or "/bin/sh")
    if (
        not force_bare
        and platform.system() == "Darwin"
        and sandbox_available()
        and seatbelt_applies()
    ):
        profile = write_seatbelt_profile()
        exe = _which("sandbox-exec") or "/usr/bin/sandbox-exec"
        return [exe, "-f", str(profile), shell_bin, "-c", cmd]
    return [shell_bin, "-c", cmd]


def describe_floor() -> str:
    paths = ", ".join(str(p) for p in load_deny_paths())
    if sandbox_available() and seatbelt_applies():
        backend = "sandbox-exec/Seatbelt"
    elif sandbox_available():
        backend = "seatbelt-binary-present-but-unusable (nested/denied; bare shell + DiD)"
    else:
        backend = "none (classification+TTY only)"
    return f"L0 sandbox floor backend={backend}; deny=[{paths}]"
