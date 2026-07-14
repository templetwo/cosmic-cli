"""Project context: scan + read files for Stargazer."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    ".egg-info",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
    ".coverage",
    "_archive",
}


class ContextManager:
    """Scans and reads project files under a root directory."""

    DEFAULT_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".md",
        ".json",
        ".html",
        ".css",
        ".scss",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".sh",
        ".rs",
        ".go",
        ".rb",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        "Dockerfile",
        "Makefile",
    }

    def __init__(
        self,
        root_dir: str = ".",
        extensions: Optional[Set[str]] = None,
        max_files: int = 2000,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.extensions = extensions or self.DEFAULT_EXTENSIONS
        self.max_files = max_files
        self.file_map = self._scan_files()

    def _scan_files(self) -> List[Path]:
        scanned: List[Path] = []
        if not self.root_dir.is_dir():
            return scanned
        for file_path in self.root_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in file_path.parts):
                continue
            name = file_path.name
            if name in self.extensions or file_path.suffix in self.extensions:
                scanned.append(file_path)
                if len(scanned) >= self.max_files:
                    break
        return scanned

    def read_file(self, file_path: str) -> str:
        """Read a file relative to root (or absolute under root)."""
        raw = file_path.strip().strip("'\"")
        target = Path(raw)
        if not target.is_absolute():
            target = self.root_dir / raw
        try:
            target = target.resolve()
        except OSError as e:
            return f"[Error] Could not resolve {file_path}: {e}"

        # Stay inside work dir
        try:
            target.relative_to(self.root_dir)
        except ValueError:
            return f"[Error] Path escapes work dir: {file_path}"

        if not target.is_file():
            return f"[Error] File not found: {file_path}"
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"[Error] Binary or non-utf8 file: {file_path}"
        except OSError as e:
            return f"[Error] Could not read {file_path}: {e}"

    def build_context_prompt(self, files: List[str]) -> str:
        chunks: List[str] = []
        for file_path in files:
            content = self.read_file(file_path)
            chunks.append(f"--- START OF FILE: {file_path} ---\n{content}\n--- END OF FILE: {file_path} ---\n")
        return "\n".join(chunks)

    def get_file_tree(self) -> str:
        lines: List[str] = []
        for path in sorted(self.file_map):
            try:
                lines.append(str(path.relative_to(self.root_dir)))
            except ValueError:
                lines.append(str(path))
        return "\n".join(lines)
