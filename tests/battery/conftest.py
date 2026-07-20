"""Cosmic CLI conformance battery — shared fixtures, environment guards, ASR.

Design rules for this battery:

1. A green run must MEAN something. Any test whose guarantee the host cannot
   enforce (chmod seals under uid 0, flock on a filesystem without it, TTY
   checks with no TTY) is SKIPPED with a printed reason, never silently passed.
   ``test_00_environment`` prints the capability matrix so a run's scope is
   visible in the log.
2. Attack tests are marked ``@pytest.mark.attack(cls=...)``. The plugin below
   computes Attack Success Rate per class: a FAILING attack test means the
   attack SUCCEEDED. ASR is the number the paper reports.
3. Protocol tests are black box. They invoke ``cosmic-cli gate`` as a real
   subprocess with a real environment, because the contract under test is a
   process contract (stdout bytes, env-only nonce), not a Python API.
4. Nothing touches the operator's real ``~/.cosmic-cli``. Subprocess tests get
   an isolated HOME.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "scripts" / "hooks" / "cosmic-gate-wrapper.sh"
LAUNCHER = REPO / "scripts" / "cosmic-launch-grok.sh"

SENTINEL_PREFIX = "COSMIC-ALLOW v1 "

# Prefer the tree under test, not a stale cosmic-cli earlier on PATH
# (this machine has multiple entrypoints; wrong one falsely inflates ASR).
def _gate_argv() -> List[str]:
    venv_cli = REPO / "venv" / "bin" / "cosmic-cli"
    if venv_cli.is_file():
        return [str(venv_cli), "gate", "--hook", "grok"]
    return [sys.executable, "-m", "cosmic_cli.main", "gate", "--hook", "grok"]



# --------------------------------------------------------------------------
# Capability detection
# --------------------------------------------------------------------------

def _has_flock() -> bool:
    try:
        import fcntl  # noqa: F401
    except Exception:
        return False
    return True


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _has_kernel_sandbox() -> Tuple[bool, str]:
    """Is a real kernel floor available on this host?"""
    if sys.platform == "darwin":
        return (shutil.which("sandbox-exec") is not None, "macOS Seatbelt")
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="utf-8") as fh:
                pass
        except Exception:
            return (False, "linux (unknown)")
        # Landlock presence is kernel-version dependent; probe the ABI file.
        return (Path("/sys/kernel/security/landlock").exists(), "Linux Landlock")
    return (False, f"{sys.platform} (no kernel floor)")


CAPS: Dict[str, object] = {
    "platform": platform.platform(),
    "python": sys.version.split()[0],
    "uid": os.getuid() if hasattr(os, "getuid") else -1,
    "is_root": _is_root(),
    "has_flock": _has_flock(),
    "has_tty": sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False,
    "kernel_sandbox": _has_kernel_sandbox(),
    # Gate argv always resolves this tree; PATH entry is informational only.
    "cosmic_on_path": True,
    "gate_argv": " ".join(_gate_argv()),
    "wrapper_present": WRAPPER.is_file(),
    "launcher_present": LAUNCHER.is_file(),
}


# Guard markers other modules import.
requires_nonroot = pytest.mark.skipif(
    CAPS["is_root"],
    reason=(
        "uid 0 bypasses POSIX write bits, so chmod-based seals cannot be "
        "falsified here. This is an ENVIRONMENT limit, not a pass."
    ),
)
requires_flock = pytest.mark.skipif(
    not CAPS["has_flock"], reason="fcntl.flock unavailable on this host"
)
requires_cli = pytest.mark.skipif(
    False,  # gate runs via _gate_argv() against this tree
    reason="unused",
)
requires_wrapper = pytest.mark.skipif(
    not CAPS["wrapper_present"], reason="scripts/hooks/cosmic-gate-wrapper.sh missing"
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def nonce() -> str:
    """A conforming nonce: >=128 bits, lowercase hex (gate regex ^[0-9a-f]{32,}$)."""
    return secrets.token_hex(16)


@pytest.fixture
def iso_home(tmp_path: Path) -> Path:
    """An isolated HOME so no test can read or write the operator's real store."""
    h = tmp_path / "home"
    (h / ".cosmic-cli").mkdir(parents=True)
    return h


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def gate(iso_home: Path, workspace: Path):
    """Black-box gate runner. Returns (stdout, stderr, returncode)."""

    def _run(
        envelope: Optional[dict],
        *,
        nonce: Optional[str] = "auto",
        raw_stdin: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ) -> Tuple[str, str, int]:
        env = dict(os.environ)
        env["HOME"] = str(iso_home)
        env.pop("COSMIC_GATE_NONCE", None)
        env.pop("COSMIC_APPROVAL_TOKEN", None)
        if nonce == "auto":
            nonce = secrets.token_hex(16)
        if nonce is not None:
            env["COSMIC_GATE_NONCE"] = nonce
        if extra_env:
            env.update(extra_env)
        payload = raw_stdin if raw_stdin is not None else json.dumps(envelope)
        # Force PYTHONPATH so -m cosmic_cli.main always hits this tree
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            _gate_argv(),
            input=payload,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(workspace),
            timeout=timeout,
        )
        return proc.stdout, proc.stderr, proc.returncode

    _run.nonce_for = lambda: secrets.token_hex(16)  # type: ignore[attr-defined]
    return _run


@pytest.fixture
def wrapper(iso_home: Path, workspace: Path):
    """Black-box wrapper runner. Returns (decision_dict_or_None, stdout, rc)."""

    def _run(
        envelope: Optional[dict],
        *,
        raw_stdin: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        path_override: Optional[str] = None,
        timeout: int = 60,
    ):
        env = dict(os.environ)
        env["HOME"] = str(iso_home)
        env.pop("COSMIC_GATE_NONCE", None)
        env.pop("COSMIC_APPROVAL_TOKEN", None)
        if path_override is not None:
            env["PATH"] = path_override
        if extra_env:
            env.update(extra_env)
        # Prefer this tree's CLI unless the test intentionally isolates PATH
        # (e.g. path_override="/bin:/usr/bin" to prove missing-cosmic-cli deny).
        if path_override is None:
            env["PATH"] = str(REPO / "venv" / "bin") + os.pathsep + env.get("PATH", "")
            env["COSMIC_CLI_HOME"] = str(REPO)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        payload = raw_stdin if raw_stdin is not None else json.dumps(envelope)
        proc = subprocess.run(
            ["bash", str(WRAPPER)],
            input=payload,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(workspace),
            timeout=timeout,
        )
        try:
            decision = json.loads(proc.stdout.strip())
        except Exception:
            decision = None
        return decision, proc.stdout, proc.returncode

    return _run


def envelope(tool: str, tool_input: dict, cwd: str = "/tmp") -> dict:
    """Build a cockpit PreToolUse envelope in the shape the gate parses."""
    return {
        "hookEventName": "pre_tool_use",
        "toolName": tool,
        "toolInput": tool_input,
        "cwd": cwd,
        "workspaceRoot": cwd,
        "sessionId": "battery-session",
        "toolUseId": "battery-tooluse",
    }


# --------------------------------------------------------------------------
# ASR plugin
# --------------------------------------------------------------------------

_ATTACK_RESULTS: List[Tuple[str, str, str]] = []  # (cls, nodeid, outcome)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "attack(cls): an adversarial probe. A FAILING attack test means the "
        "attack SUCCEEDED and counts toward Attack Success Rate.",
    )
    config.addinivalue_line(
        "markers", "known_gap(ref): documents a gap that is open by design decision."
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    marker = item.get_closest_marker("attack")
    if marker is None:
        return
    cls = marker.kwargs.get("cls", "unclassified")
    _ATTACK_RESULTS.append((cls, item.nodeid, report.outcome))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    tr = terminalreporter
    tr.write_sep("=", "COSMIC BATTERY: environment")
    for k, v in CAPS.items():
        tr.write_line(f"  {k:18} {v}")

    if not _ATTACK_RESULTS:
        return

    by_cls: Dict[str, List[str]] = {}
    for cls, nodeid, res in _ATTACK_RESULTS:
        by_cls.setdefault(cls, []).append(res)

    tr.write_sep("=", "COSMIC BATTERY: Attack Success Rate")
    total = succeeded = 0
    for cls in sorted(by_cls):
        results = by_cls[cls]
        # A skipped probe is not evidence either way; exclude from the rate.
        scored = [r for r in results if r in ("passed", "failed")]
        s = sum(1 for r in scored if r == "failed")
        total += len(scored)
        succeeded += s
        skipped = len(results) - len(scored)
        rate = (s / len(scored) * 100) if scored else float("nan")
        note = f"  ({skipped} skipped, unscored)" if skipped else ""
        tr.write_line(f"  {cls:28} ASR {rate:6.1f}%   {s}/{len(scored)} succeeded{note}")
    overall = (succeeded / total * 100) if total else float("nan")
    tr.write_line("")
    tr.write_line(f"  {'OVERALL':28} ASR {overall:6.1f}%   {succeeded}/{total} succeeded")
    tr.write_line("")
    tr.write_line("  ASR target for a conforming build: 0.0% on every class.")

    # Machine-readable artifact for the deposit.
    out = Path(os.environ.get("COSMIC_BATTERY_JSON", REPO / "battery-results.json"))
    try:
        out.write_text(
            json.dumps(
                {
                    "environment": {k: str(v) for k, v in CAPS.items()},
                    "classes": {
                        cls: {
                            "scored": len([r for r in rs if r in ("passed", "failed")]),
                            "succeeded": sum(1 for r in rs if r == "failed"),
                            "skipped": sum(1 for r in rs if r == "skipped"),
                        }
                        for cls, rs in by_cls.items()
                    },
                    "overall_asr_pct": overall,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tr.write_line(f"  results written: {out}")
    except Exception as e:  # pragma: no cover
        tr.write_line(f"  (could not write results json: {e})")
