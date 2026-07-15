"""Hardened checkpoint/rollback foundation (Phase 5, from phase5_checkpoint_v0.1).

Content-preserving checkpoints with canonical-JSON manifest integrity hashes,
backup/restore hash verification, atomic per-file restore, and explicit
unrelated-change detection. All failures raise CheckpointError.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import os
import secrets
import shutil
import tempfile
import time


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be created, verified, or restored."""


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: str
    existed: bool
    pre_sha256: str | None
    backup_sha256: str | None
    mode: int | None


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int
    checkpoint_id: str
    created_at_unix_ns: int
    workspace_root: str
    backup_root: str
    files: tuple[FileSnapshot, ...]
    manifest_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_manifest_payload(data: dict) -> bytes:
    unsigned = {k: v for k, v in data.items() if k != "manifest_sha256"}
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class CheckpointManager:
    """Content-preserving checkpoint and exact rollback foundation.

    Scope:
      * regular files only;
      * all paths must resolve inside workspace_root;
      * existing files are restored byte-for-byte;
      * files absent at checkpoint time are removed during rollback;
      * unrelated-change detection is explicit and separate from restoration.
    """

    SCHEMA_VERSION = 1

    def __init__(self, workspace_root: Path, backup_dir: Path | None = None):
        self.workspace_root = workspace_root.resolve(strict=True)
        if not self.workspace_root.is_dir():
            raise CheckpointError("workspace_root must be a directory")

        requested_backup = backup_dir or (self.workspace_root / ".cosmicbak")
        self.backup_dir = requested_backup.resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_target(self, path: Path) -> tuple[Path, Path]:
        candidate = path if path.is_absolute() else self.workspace_root / path
        resolved_parent = candidate.parent.resolve(strict=True)
        resolved = resolved_parent / candidate.name

        try:
            relative = resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise CheckpointError(f"path escapes workspace: {path}") from exc

        if resolved.exists() and not resolved.is_file():
            raise CheckpointError(f"only regular files are supported: {path}")
        if resolved.is_symlink():
            raise CheckpointError(f"symlinks are not supported: {path}")

        return resolved, relative

    def create_checkpoint(self, paths: Iterable[Path]) -> CheckpointManifest:
        normalized: dict[str, tuple[Path, Path]] = {}
        for path in paths:
            absolute, relative = self._normalize_target(Path(path))
            normalized[relative.as_posix()] = (absolute, relative)

        if not normalized:
            raise CheckpointError("at least one target path is required")

        checkpoint_id = f"ckpt-{time.time_ns()}-{secrets.token_hex(8)}"
        final_root = self.backup_dir / checkpoint_id
        if final_root.exists():
            raise CheckpointError("checkpoint id collision")

        temp_root = Path(
            tempfile.mkdtemp(prefix=f".{checkpoint_id}.", dir=self.backup_dir)
        )

        snapshots: list[FileSnapshot] = []
        try:
            payload_root = temp_root / "payload"
            payload_root.mkdir()

            for key in sorted(normalized):
                source, relative = normalized[key]
                if source.exists():
                    pre_sha = _sha256_file(source)
                    mode = source.stat().st_mode & 0o777
                    backup = payload_root / relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, backup)
                    backup_sha = _sha256_file(backup)
                    if backup_sha != pre_sha:
                        raise CheckpointError(
                            f"backup verification failed for {relative.as_posix()}"
                        )
                    snapshots.append(
                        FileSnapshot(
                            relative_path=relative.as_posix(),
                            existed=True,
                            pre_sha256=pre_sha,
                            backup_sha256=backup_sha,
                            mode=mode,
                        )
                    )
                else:
                    snapshots.append(
                        FileSnapshot(
                            relative_path=relative.as_posix(),
                            existed=False,
                            pre_sha256=None,
                            backup_sha256=None,
                            mode=None,
                        )
                    )

            unsigned = {
                "schema_version": self.SCHEMA_VERSION,
                "checkpoint_id": checkpoint_id,
                "created_at_unix_ns": time.time_ns(),
                "workspace_root": str(self.workspace_root),
                "backup_root": str(final_root),
                "files": [asdict(item) for item in snapshots],
            }
            manifest_sha = _sha256_bytes(_canonical_manifest_payload(unsigned))
            manifest_data = dict(unsigned, manifest_sha256=manifest_sha)

            manifest_path = temp_root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest_data, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            os.replace(temp_root, final_root)
            return self.load_manifest(final_root / "manifest.json")
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise

    def load_manifest(self, manifest_path: Path) -> CheckpointManifest:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if data.get("schema_version") != self.SCHEMA_VERSION:
            raise CheckpointError("unsupported checkpoint schema")

        expected = data.get("manifest_sha256")
        actual = _sha256_bytes(_canonical_manifest_payload(data))
        if not expected or not secrets.compare_digest(expected, actual):
            raise CheckpointError("manifest integrity check failed")

        files = tuple(FileSnapshot(**item) for item in data["files"])
        manifest = CheckpointManifest(
            schema_version=data["schema_version"],
            checkpoint_id=data["checkpoint_id"],
            created_at_unix_ns=data["created_at_unix_ns"],
            workspace_root=data["workspace_root"],
            backup_root=data["backup_root"],
            files=files,
            manifest_sha256=data["manifest_sha256"],
        )
        if Path(manifest.workspace_root).resolve() != self.workspace_root:
            raise CheckpointError("manifest belongs to another workspace")
        return manifest

    def detect_unrelated_changes(
        self,
        manifest: CheckpointManifest,
        observed_paths: Iterable[Path],
    ) -> tuple[str, ...]:
        intended = {item.relative_path for item in manifest.files}
        unrelated: list[str] = []
        for path in observed_paths:
            _, relative = self._normalize_target(Path(path))
            key = relative.as_posix()
            if key not in intended:
                unrelated.append(key)
        return tuple(sorted(set(unrelated)))

    def rollback(self, manifest: CheckpointManifest) -> None:
        """Restore all files or refuse entirely (no partial restore).

        Preflight verifies every backup hash before any workspace mutation.
        """
        verified = self.load_manifest(Path(manifest.backup_root) / "manifest.json")
        payload_root = Path(verified.backup_root) / "payload"

        # --- preflight: refuse-all if any backup is missing/corrupt ---
        for snapshot in verified.files:
            if not snapshot.existed:
                continue
            backup = payload_root / snapshot.relative_path
            if not backup.is_file():
                raise CheckpointError(
                    f"backup missing for {snapshot.relative_path} — refuse-all rollback"
                )
            backup_sha = _sha256_file(backup)
            if backup_sha != snapshot.backup_sha256:
                raise CheckpointError(
                    f"backup corrupted for {snapshot.relative_path} — refuse-all rollback"
                )

        # --- apply only after all backups verify ---
        for snapshot in verified.files:
            target = self.workspace_root / snapshot.relative_path
            if snapshot.existed:
                backup = payload_root / snapshot.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                temp_target = target.with_name(
                    f".{target.name}.cosmic-restore-{secrets.token_hex(4)}"
                )
                shutil.copy2(backup, temp_target)
                os.replace(temp_target, target)
                if snapshot.mode is not None:
                    target.chmod(snapshot.mode)

                restored_sha = _sha256_file(target)
                if restored_sha != snapshot.pre_sha256:
                    raise CheckpointError(
                        f"restore verification failed for {snapshot.relative_path}"
                    )
            else:
                if target.exists():
                    if not target.is_file() or target.is_symlink():
                        raise CheckpointError(
                            f"refusing to remove non-regular target: "
                            f"{snapshot.relative_path}"
                        )
                    target.unlink()
