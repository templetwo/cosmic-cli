"""Phase 6 integration test: gateway + checkpoint + rollback + unrelated-change escalation.

Source: phase6_complete tests/test_seven_gates.py, adapted to the hardened
checkpoint/gateway APIs and strengthened:
- mgr._hash_file (phase6-only helper) -> local sha256 helper;
- bare try/except around the failing executor -> pytest.raises (must raise);
- the unrelated-change escalation assertion in phase6 was inside a
  try/except that could never fire (the gateway re-passed the manifest's own
  paths to verify_changes). Here real observed_paths are supplied and the
  escalation MUST fire, and the rollback-on-breach is asserted too.
"""

import hashlib
from pathlib import Path

import pytest

from cosmic_cli.checkpoint import CheckpointManager
from cosmic_cli.gateway import ActionGateway
from cosmic_cli.policy import Disposition, ActionType, PolicyDecision


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simple_evaluator(rules, action_type, action_input):
    return PolicyDecision(
        disposition=Disposition.OPEN,
        matches=(),
        default_used=True,
        policy_sha256="p",
        evaluated_input_sha256="a"
    )


def test_seven_gates_all_triggered_and_filesystem_verified(tmp_path: Path):
    ws = tmp_path
    mgr = CheckpointManager(ws)
    gw = ActionGateway(simple_evaluator, checkpoint_manager=mgr)

    f = ws / "critical.txt"
    f.write_text("original content")
    original_hash = _sha256(f)

    receipt = gw.authorize(ActionType.SHELL, "edit", [], [f])
    assert receipt.checkpoint_id is not None
    assert receipt.checkpoint_manifest is not None

    def failing_executor():
        f.write_text("corrupted during failure")
        raise RuntimeError("simulated failure")

    with pytest.raises(RuntimeError, match="simulated failure"):
        gw.execute_with_receipt(receipt, failing_executor)

    # Rollback-on-failure restored the checkpointed content byte-for-byte.
    assert f.read_text() == "original content"
    assert _sha256(f) == original_hash

    unrelated = ws / "unexpected.log"
    unrelated.write_text("created outside plan")
    receipt2 = gw.authorize(ActionType.SHELL, "edit2", [], [f])

    def safe_executor():
        f.write_text("changed under plan")

    # Hardened adaptation: pass real observations so detection can actually run.
    with pytest.raises(PermissionError, match="Unrelated changes detected and escalated"):
        gw.execute_with_receipt(receipt2, safe_executor, observed_paths=[f, unrelated])

    # Policy breach rolled the intended file back to its checkpointed state.
    assert f.read_text() == "original content"
