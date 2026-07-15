"""Seven adversarial tests for the hardened CheckpointManager.

Source: phase5_checkpoint_v0.1 verbatim.
"""

from pathlib import Path
import json
import pytest

from cosmic_cli.checkpoint import CheckpointError, CheckpointManager


def test_roundtrip_preserves_duplicate_basenames(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = tmp_path / "a" / "config.txt"
    second = tmp_path / "b" / "config.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    manager = CheckpointManager(tmp_path)
    manifest = manager.create_checkpoint([first, second])

    first.write_text("changed-a", encoding="utf-8")
    second.write_text("changed-b", encoding="utf-8")
    manager.rollback(manifest)

    assert first.read_text(encoding="utf-8") == "alpha"
    assert second.read_text(encoding="utf-8") == "beta"


def test_rollback_restores_deleted_file(tmp_path: Path):
    target = tmp_path / "state.txt"
    target.write_text("pre-state", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    manifest = manager.create_checkpoint([target])

    target.unlink()
    manager.rollback(manifest)

    assert target.read_text(encoding="utf-8") == "pre-state"


def test_rollback_removes_file_created_after_checkpoint(tmp_path: Path):
    target = tmp_path / "new.txt"
    manager = CheckpointManager(tmp_path)
    manifest = manager.create_checkpoint([target])

    target.write_text("created later", encoding="utf-8")
    manager.rollback(manifest)

    assert not target.exists()


def test_tampered_backup_blocks_rollback(tmp_path: Path):
    target = tmp_path / "state.txt"
    target.write_text("original", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    manifest = manager.create_checkpoint([target])

    backup = Path(manifest.backup_root) / "payload" / "state.txt"
    backup.write_text("tampered", encoding="utf-8")
    target.write_text("modified", encoding="utf-8")

    with pytest.raises(CheckpointError, match="backup corrupted"):
        manager.rollback(manifest)


def test_tampered_manifest_is_rejected(tmp_path: Path):
    target = tmp_path / "state.txt"
    target.write_text("original", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    manifest = manager.create_checkpoint([target])
    manifest_path = Path(manifest.backup_root) / "manifest.json"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["checkpoint_id"] = "forged"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CheckpointError, match="manifest integrity"):
        manager.rollback(manifest)


def test_path_escape_is_rejected(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    manager = CheckpointManager(tmp_path)

    with pytest.raises(CheckpointError, match="escapes workspace"):
        manager.create_checkpoint([outside])


def test_unrelated_change_detection(tmp_path: Path):
    intended = tmp_path / "intended.txt"
    unrelated = tmp_path / "other.txt"
    intended.write_text("a", encoding="utf-8")
    unrelated.write_text("b", encoding="utf-8")

    manager = CheckpointManager(tmp_path)
    manifest = manager.create_checkpoint([intended])

    assert manager.detect_unrelated_changes(
        manifest, [intended, unrelated]
    ) == ("other.txt",)
