"""Layer 0 — environment truth.

These tests do not test cosmic-cli. They establish what THIS HOST can actually
falsify, so that a green battery run is legible. The failure this layer exists
to prevent: a suite that passes because the host silently cannot enforce the
mechanism under test.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from .conftest import CAPS, LAUNCHER, WRAPPER, requires_cli


def test_capability_matrix_is_reported():
    """Always passes; the value is the printed matrix in the summary."""
    assert "platform" in CAPS


def test_root_invalidates_permission_based_guarantees():
    """A chmod seal is not a security boundary against uid 0.

    cosmic-cli's mutation seal drops write bits and swallows failure
    (``except OSError: pass``). Under uid 0 the seal silently no-ops: the chmod
    succeeds and the protection does not exist. Agents commonly run as root in
    containers and CI, so this is a deployment property, not a test artifact.
    """
    if CAPS["is_root"]:
        pytest.skip(
            "running as uid 0: seal-based tests cannot be falsified here. "
            "Re-run the battery as a non-root user before claiming seal "
            "conformance."
        )
    assert not CAPS["is_root"]


@requires_cli
def test_gate_verb_exists():
    proc = subprocess.run(
        ["cosmic-cli", "gate", "--verb-check"], capture_output=True, text=True
    )
    assert proc.returncode == 0, "gate --verb-check must succeed for the wrapper probe"


def test_wrapper_and_launcher_present_and_executable():
    assert WRAPPER.is_file(), "wrapper script missing"
    assert LAUNCHER.is_file(), "launcher script missing"


def test_kernel_floor_availability_is_declared():
    """Record whether a kernel floor is even possible on this host.

    ranking.py calls kernel isolation enforcement layer 1 (strongest). If this
    host has none, every ranking result below is layer 2/3 only, and the
    battery says so rather than implying a floor that is absent.
    """
    available, backend = CAPS["kernel_sandbox"]  # type: ignore[misc]
    if not available:
        pytest.skip(
            f"no kernel floor available ({backend}). Ranking results in this "
            "run rest on the TTY gate and command classification only."
        )
    assert available
