import os
from pathlib import Path
from typing import List, Set

class ContextManager:
    """
    Manages the project context by scanning and reading relevant files.
    """
    DEFAULT_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.json',
        '.html', '.css', '.scss', '.yaml', '.yml', '.toml',
        'Dockerfile', '.sh',
    }

    def __init__(self, root_dir: str = '.', extensions: Set[str] = None):
        self.root_dir = Path(root_dir).resolve()
        self.extensions = extensions or self.DEFAULT_EXTENSIONS
        self.file_map = self._scan_files()

    def _scan_files(self) -> List[Path]:
        """Scans the root directory for files with allowed extensions."""
        scanned_files = []
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                if file_path.name in self.extensions or file_path.suffix in self.extensions:
                    if not any(part in ['.git', '__pycache__', 'node_modules', 'venv', '.egg-info'] for part in file_path.parts):
                        scanned_files.append(file_path)
        return scanned_files

    def read_file(self, file_path: str) -> str:
        """Reads the content of a single file relative to the root directory."""
        target_file = self.root_dir / file_path
        if not target_file.is_file():
            return f"[Error] File not found: {file_path}"
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"[Error] Could not read file {file_path}: {e}"

    def build_context_prompt(self, files: List[str]) -> str:
        """Builds a single string prompt containing the content of multiple files."""
        context_str = ""
        for file_path in files:
            content = self.read_file(file_path)
            context_str += f"--- START OF FILE: {file_path} ---\n"
            context_str += content
            context_str += f"\n--- END OF FILE: {file_path} ---\n\n"
        return context_str

    def get_file_tree(self) -> str:
        """Returns a string representing the file tree."""
        tree_str = ""
        for path in sorted(self.file_map):
            relative_path = path.relative_to(self.root_dir)
            tree_str += f"{relative_path}\n"
        return tree_str 