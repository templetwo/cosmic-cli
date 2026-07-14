"""cosmic-cli init — drop project agent notes (differentiator, not config spam)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from cosmic_cli.principles import INIT_DOC

DEFAULT_NAME = "COSMIC.md"


def write_init(target_dir: Path, *, force: bool = False, name: str = DEFAULT_NAME) -> Tuple[str, bool]:
    target_dir = target_dir.resolve()
    if not target_dir.is_dir():
        return f"[Error] not a directory: {target_dir}", False
    path = target_dir / name
    if path.exists() and not force:
        return f"[Error] {path} exists (use --force to overwrite)", False
    try:
        path.write_text(INIT_DOC.strip() + "\n", encoding="utf-8")
    except OSError as e:
        return f"[Error] write failed: {e}", False
    return f"init ok: {path}", True
