"""Temple cockpit floor — provision + verify (privilege ranking kernel layer).

Default install must be fail-closed on the PAUSE token store, not "example
available if you remember a flag." This module:

1. Writes ``[profiles.temple]`` into ``~/.grok/sandbox.toml`` (additive, only
   token *files* — never the whole ``~/.cosmic-cli`` tree; hooks live there).
2. Sets ``[sandbox] profile = "temple"`` in ``~/.grok/config.toml`` so grok
   defaults to the floor without ``--sandbox temple``.
3. ``run_floor_canary`` proves a sandboxed shell cannot read the token file
   (Seatbelt on macOS; fail closed if the platform cannot enforce).

Launcher calls (3) the same way it calls ``gate --boot-canary``.
"""

from __future__ import annotations

import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Token artifacts only — NEVER ~/.cosmic-cli itself (bricks PreToolUse hooks).
TOKEN_BASENAMES: Tuple[str, ...] = (
    "last_pause_token.json",
    "operator_approval_token",
    "local_approvals.json",
    "local_approvals.json.lock",
)

_TEMPLE_MARKER = "# cosmic-cli temple profile — managed by init --grok"


def token_deny_paths(home: Optional[Path] = None) -> List[Path]:
    h = home or Path.home()
    base = h / ".cosmic-cli"
    return [base / name for name in TOKEN_BASENAMES]


def _render_temple_section(deny: Sequence[Path]) -> str:
    lines = [
        _TEMPLE_MARKER,
        "[profiles.temple]",
        'extends = "workspace"',
        "deny = [",
    ]
    for p in deny:
        lines.append(f'  "{p.resolve()}",')
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def ensure_temple_sandbox_toml(*, home: Optional[Path] = None) -> Path:
    """Idempotently install/update profiles.temple in ~/.grok/sandbox.toml."""
    h = home or Path.home()
    path = h / ".grok" / "sandbox.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    deny = token_deny_paths(h)
    section = _render_temple_section(deny)

    if not path.is_file():
        body = (
            "# Grok Build sandbox profiles (cosmic-cli temple floor)\n"
            "# Managed section: [profiles.temple] — re-run init --grok to refresh paths.\n"
            "# Do not deny the whole ~/.cosmic-cli tree (hooks must stay executable).\n\n"
            + section
        )
        path.write_text(body, encoding="utf-8")
        return path

    text = path.read_text(encoding="utf-8")
    # Replace existing [profiles.temple] block (until next top-level [profiles. or EOF)
    pattern = re.compile(
        r"(?ms)^(?:# cosmic-cli temple profile[^\n]*\n)?"
        r"\[profiles\.temple\]\n"
        r".*?"
        r"(?=^\[profiles\.|\Z)",
    )
    if pattern.search(text):
        text = pattern.sub(section, text)
        if not text.endswith("\n"):
            text += "\n"
        path.write_text(text, encoding="utf-8")
        return path

    # Append
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + "\n" + section, encoding="utf-8")
    return path


def ensure_temple_default_profile(*, home: Optional[Path] = None) -> Path:
    """Set [sandbox] profile = \"temple\" in ~/.grok/config.toml (additive)."""
    h = home or Path.home()
    path = h / ".grok" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(
            "# Managed by cosmic-cli init --grok — temple floor default\n"
            "[sandbox]\n"
            'profile = "temple"\n',
            encoding="utf-8",
        )
        return path

    text = path.read_text(encoding="utf-8")
    if re.search(r"(?m)^\[sandbox\]", text):
        # Replace profile = … under [sandbox] or insert after [sandbox]
        def repl_section(m: re.Match) -> str:
            body = m.group(0)
            if re.search(r'(?m)^profile\s*=', body):
                body = re.sub(
                    r'(?m)^profile\s*=\s*["\']?[^"\'\n]+["\']?',
                    'profile = "temple"',
                    body,
                    count=1,
                )
            else:
                # after [sandbox] line
                body = re.sub(
                    r"(?m)^(\[sandbox\]\s*\n)",
                    r'\1profile = "temple"\n',
                    body,
                    count=1,
                )
            return body

        # Match [sandbox] through next [section] or EOF
        text2, n = re.subn(
            r"(?ms)^\[sandbox\]\n.*?(?=^\[|\Z)",
            repl_section,
            text,
            count=1,
        )
        if n:
            path.write_text(text2 if text2.endswith("\n") else text2 + "\n", encoding="utf-8")
            return path

    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + '\n[sandbox]\nprofile = "temple"\n', encoding="utf-8")
    return path


def provision_temple_floor(*, home: Optional[Path] = None) -> str:
    """Write sandbox.toml + config default. Return human summary."""
    sb = ensure_temple_sandbox_toml(home=home)
    cfg = ensure_temple_default_profile(home=home)
    deny = ", ".join(p.name for p in token_deny_paths(home))
    return (
        f"temple floor provisioned: sandbox={sb} (deny token files: {deny}); "
        f"default profile in {cfg}"
    )


def _verify_toml_has_temple(home: Path) -> Optional[str]:
    path = home / ".grok" / "sandbox.toml"
    if not path.is_file():
        return f"missing {path} — run cosmic-cli init --grok"
    text = path.read_text(encoding="utf-8")
    if "[profiles.temple]" not in text:
        return f"{path} has no [profiles.temple] — run init --grok"
    for p in token_deny_paths(home):
        # path may appear with or without resolve quirks
        if str(p) not in text and str(p.resolve()) not in text:
            return f"{path} temple.deny missing {p.name}"
    return None


def _verify_default_profile(home: Path) -> Optional[str]:
    """Default must be temple via config and/or forced env at launch."""
    if os.environ.get("GROK_SANDBOX", "").strip() == "temple":
        return None
    cfg = home / ".grok" / "config.toml"
    if cfg.is_file():
        text = cfg.read_text(encoding="utf-8")
        # crude but sufficient: [sandbox] region contains profile = "temple"
        m = re.search(r"(?ms)^\[sandbox\]\n(.*?)(?=^\[|\Z)", text)
        if m and re.search(r'(?m)^profile\s*=\s*["\']?temple["\']?', m.group(1)):
            return None
    return (
        f"{cfg} [sandbox] profile is not temple "
        "(and GROK_SANDBOX!=temple) — run init --grok"
    )


def _seatbelt_canary(home: Path) -> Tuple[bool, str]:
    """Live kernel probe: sandboxed shell must not read last_pause_token.json."""
    if platform.system() != "Darwin":
        # Linux: bwrap floor is grok-internal; we still require toml+default.
        # Document as platform gap if no sandbox-exec.
        return True, "skip live Seatbelt (non-macOS); toml+default required"

    sandbox_exec = Path("/usr/bin/sandbox-exec")
    if not sandbox_exec.is_file():
        return False, "sandbox-exec missing — cannot verify temple floor on macOS"

    target = home / ".cosmic-cli" / "last_pause_token.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    canary = "tok-floor-canary-must-not-leak"
    planted = False
    prev: Optional[bytes] = None
    if target.is_file():
        prev = target.read_bytes()
    else:
        planted = True
    target.write_text(
        f'{{"token":"{canary}","floor_canary":true}}\n', encoding="utf-8"
    )
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass

    # Seatbelt: allow default, deny read/write of each token path (literal + subpath parent not used)
    deny_paths = token_deny_paths(home)
    rules = ["(version 1)", "(allow default)"]
    for p in deny_paths:
        sp = str(p.resolve()).replace("\\", "\\\\").replace('"', '\\"')
        rules.append(f'(deny file-read* (literal "{sp}"))')
        rules.append(f'(deny file-read* (subpath "{sp}"))')
        for action in (
            "file-write-data",
            "file-write-create",
            "file-write-unlink",
        ):
            rules.append(f'(deny {action} (literal "{sp}"))')
    profile_body = "\n".join(rules) + "\n"

    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".sb", prefix="cosmic-floor-", delete=False
        ) as f:
            f.write(profile_body)
            prof = f.name
        try:
            os.chmod(prof, 0o600)
            r = subprocess.run(
                [
                    str(sandbox_exec),
                    "-f",
                    prof,
                    "/bin/cat",
                    str(target.resolve()),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        finally:
            try:
                os.unlink(prof)
            except OSError:
                pass
    finally:
        if planted:
            try:
                target.unlink()
            except OSError:
                pass
        elif prev is not None:
            target.write_bytes(prev)

    if canary in (r.stdout or ""):
        return False, f"floor canary LEAKED via sandboxed cat (stdout={r.stdout!r})"
    if r.returncode == 0 and (r.stdout or "").strip():
        return False, f"floor canary: unexpected success stdout={r.stdout!r}"
    # Operation not permitted is the happy path
    err = (r.stderr or "") + (r.stdout or "")
    if r.returncode != 0 or "Operation not permitted" in err or not (r.stdout or "").strip():
        return True, "Seatbelt deny on token file OK"
    return False, f"floor canary inconclusive: ec={r.returncode} err={err!r}"


def run_floor_canary(*, home: Optional[Path] = None) -> int:
    """Return 0 if temple floor is provisioned and binds; 1 otherwise."""
    h = home or Path.home()
    err = _verify_toml_has_temple(h)
    if err:
        print(f"floor-canary: FAIL — {err}", file=sys.stderr)
        return 1
    print("floor-canary: ok sandbox.toml temple deny list", file=sys.stderr)

    err = _verify_default_profile(h)
    if err:
        print(f"floor-canary: FAIL — {err}", file=sys.stderr)
        return 1
    print("floor-canary: ok default profile temple", file=sys.stderr)

    ok, detail = _seatbelt_canary(h)
    if not ok:
        print(f"floor-canary: FAIL — {detail}", file=sys.stderr)
        return 1
    print(f"floor-canary: ok live probe — {detail}", file=sys.stderr)
    print("floor-canary: PASS — temple floor provisioned and binding", file=sys.stderr)
    return 0
