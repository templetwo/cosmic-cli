"""Filesystem tools for Stargazer — search, list, surgical edit, write.

Filesystem is ground truth. Tools return observations; they do not "think."
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cosmic_cli.context import SKIP_DIR_NAMES


def normalize_user_path(path: str, root: Path) -> Path:
    """Resolve a user-supplied path under root.

    CRITICAL: never use str.lstrip('./') — that strips ANY leading '/' or '.',
    turning '/Users/foo' into 'Users/foo' (relative to cwd). Classic footgun.
    """
    raw = path.strip().strip("'\"")
    if not raw or raw == ".":
        return root.resolve()
    root_r = root.resolve()
    if raw.startswith("~/"):
        candidate = (Path.home() / raw[2:]).expanduser().resolve()
    elif raw == "~":
        candidate = Path.home().resolve()
    else:
        p = Path(raw)
        if p.is_absolute():
            candidate = p.resolve()
        else:
            while raw.startswith("./"):
                raw = raw[2:]
            candidate = (root_r / raw).resolve()
    try:
        candidate.relative_to(root_r)
    except ValueError as e:
        raise PermissionError(
            f"path escapes work dir ({root_r}): {path}"
        ) from e
    return candidate


def rel_key(path: str, root: Path) -> str:
    """Stable relative path key for caches (posix-style)."""
    target = normalize_user_path(path, root)
    return target.relative_to(root.resolve()).as_posix()


def _under_root(root: Path, path: str) -> Path:
    return normalize_user_path(path, root)


def _skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in path.parts)


def tool_list(root: Path, dir_path: str = ".") -> str:
    try:
        target = _under_root(root, dir_path or ".")
    except PermissionError as e:
        return f"[Error] {e}"
    if not target.is_dir():
        return f"[Error] not a directory: {dir_path}"
    entries: List[str] = []
    try:
        for p in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if p.name.startswith(".") and p.name not in {".env.example", ".gitignore"}:
                continue
            if _skip(p):
                continue
            kind = "dir " if p.is_dir() else "file"
            rel = p.relative_to(root)
            entries.append(f"{kind}  {rel}")
    except OSError as e:
        return f"[Error] list failed: {e}"
    if not entries:
        return "(empty)"
    return "\n".join(entries[:500])


def tool_glob(root: Path, pattern: str, limit: int = 200) -> str:
    pattern = pattern.strip() or "**/*"
    # pathlib glob is relative to root
    try:
        matches = []
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            if _skip(p):
                continue
            matches.append(str(p.relative_to(root)))
            if len(matches) >= limit:
                break
        # also try rglob if pattern has no **
        if not matches and "**" not in pattern and "/" not in pattern:
            for p in root.rglob(pattern):
                if not p.is_file() or _skip(p):
                    continue
                matches.append(str(p.relative_to(root)))
                if len(matches) >= limit:
                    break
    except OSError as e:
        return f"[Error] glob failed: {e}"
    if not matches:
        return f"(no matches for {pattern!r})"
    return "\n".join(sorted(matches))


def tool_grep(
    root: Path,
    pattern: str,
    path: str = ".",
    glob: Optional[str] = None,
    limit: int = 80,
) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"[Error] invalid regex: {e}"

    try:
        base = _under_root(root, path or ".")
    except PermissionError as e:
        return f"[Error] {e}"

    hits: List[str] = []
    files: List[Path] = []
    if base.is_file():
        files = [base]
    else:
        iterator = base.rglob(glob or "*") if glob else base.rglob("*")
        for p in iterator:
            if not p.is_file() or _skip(p):
                continue
            if glob and not p.match(glob):
                # pathlib match is name-based; also check suffix style
                if not p.name.endswith(glob.lstrip("*")):
                    continue
            files.append(p)
            if len(files) > 2000:
                break

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rel = fp.relative_to(root)
                hits.append(f"{rel}:{i}:{line[:240]}")
                if len(hits) >= limit:
                    return "\n".join(hits) + f"\n… (hit limit {limit})"
    return "\n".join(hits) if hits else f"(no matches for /{pattern}/)"


def _ensure_owner_writable(target: Path) -> None:
    """Clear seal from a prior mutation so sequential edits can proceed."""
    try:
        if target.is_file():
            mode = target.stat().st_mode
            if not (mode & 0o200):
                target.chmod(mode | 0o200)
    except OSError:
        pass


def tool_edit(
    root: Path,
    path: str,
    old: str,
    new: str,
    *,
    require_unique: bool = True,
) -> Tuple[str, bool]:
    """Surgical replace. Returns (observation, success)."""
    try:
        target = _under_root(root, path)
    except PermissionError as e:
        return f"[Error] {e}", False
    if not target.is_file():
        return f"[Error] file not found: {path}", False
    _ensure_owner_writable(target)
    try:
        original = target.read_text(encoding="utf-8")
    except OSError as e:
        return f"[Error] read failed: {e}", False

    count = original.count(old)
    if count == 0:
        return (
            "[Error] old_string not found. Re-READ the file and match exact text "
            "(including whitespace).",
            False,
        )
    if require_unique and count > 1:
        return (
            f"[Error] old_string matched {count} times; must be unique. "
            "Widen context in old_string.",
            False,
        )

    # Guard: refuse obviously truncated string literals (odd unescaped quote count on change)
    if path.endswith((".py", ".js", ".ts", ".tsx", ".jsx")):
        # crude: line-level unbalanced double quotes after edit often mean model cut off
        for line in new.splitlines():
            if line.count('"') % 2 == 1 and '\\"' not in line:
                return (
                    "[Error] new_text looks truncated (unbalanced quotes). "
                    "Re-send EDIT with the COMPLETE line including closing quotes.",
                    False,
                )

    updated = original.replace(old, new, 1 if require_unique else count)
    bak = target.with_suffix(target.suffix + ".cosmicbak")
    try:
        shutil.copy2(target, bak)
        target.write_text(updated, encoding="utf-8")
    except OSError as e:
        return f"[Error] write failed: {e}", False

    # Python syntax gate — restore backup on failure
    if target.suffix == ".py":
        try:
            compile(updated, str(target), "exec")
        except SyntaxError as e:
            try:
                shutil.copy2(bak, target)
            except OSError:
                pass
            return (
                f"[Error] EDIT rejected — Python syntax error after change: {e}. "
                f"File restored from {bak.name}. Fix new_text and retry.",
                False,
            )

    rel = target.relative_to(root.resolve()).as_posix()
    return (
        f"EDIT ok: rel={rel} abs={target} "
        f"(1 replacement, backup {bak.name}, {len(original)}→{len(updated)} chars)",
        True,
    )


def tool_mkdir(root: Path, path: str) -> Tuple[str, bool]:
    """Create directory (parents ok)."""
    try:
        target = _under_root(root, path)
    except PermissionError as e:
        return f"[Error] {e}", False
    try:
        existed = target.is_dir()
        target.mkdir(parents=True, exist_ok=True)
        rel = target.relative_to(root.resolve()).as_posix()
        state = "existed" if existed else "created"
        return f"MKDIR ok: rel={rel} abs={target} ({state})", True
    except OSError as e:
        return f"[Error] mkdir failed: {e}", False


def tool_write(root: Path, path: str, content: str, *, overwrite: bool = True) -> Tuple[str, bool]:
    try:
        target = _under_root(root, path)
    except PermissionError as e:
        return f"[Error] {e}", False
    if target.exists() and not overwrite:
        return f"[Error] exists (overwrite=False): {path}", False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _ensure_owner_writable(target)
            bak = target.with_suffix(target.suffix + ".cosmicbak")
            shutil.copy2(target, bak)
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"[Error] write failed: {e}", False
    rel = target.relative_to(root.resolve()).as_posix()
    return (
        f"WRITE ok: rel={rel} abs={target} ({len(content)} chars)",
        True,
    )


def tool_create(root: Path, path: str, content: str) -> Tuple[str, bool]:
    """One-shot: mkdir parents + write file. Preferred for 'folder then file' intents."""
    try:
        target = _under_root(root, path)
    except PermissionError as e:
        return f"[Error] {e}", False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _ensure_owner_writable(target)
            bak = target.with_suffix(target.suffix + ".cosmicbak")
            shutil.copy2(target, bak)
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"[Error] create failed: {e}", False
    rel = target.relative_to(root.resolve()).as_posix()
    return (
        f"CREATE ok: rel={rel} abs={target} "
        f"(parents ensured, {len(content)} chars)",
        True,
    )


def looks_like_path_bug(rel_path: str) -> bool:
    """Heuristic: nested Users/… under home workdir from the old lstrip bug."""
    p = rel_path.replace("\\", "/")
    if p.startswith("Users/") or "/Users/" in p:
        return True
    # accidental double home: home/Users/name/...
    parts = p.split("/")
    if len(parts) >= 2 and parts[0] == "Users":
        return True
    return False


def parse_edit_payload(body: str) -> Optional[Tuple[str, str, str]]:
    """EDIT: path|||old|||new  (triple pipe delimiter)."""
    parts = body.split("|||")
    if len(parts) < 3:
        return None
    path = parts[0].strip()
    # rejoin in case new/old contained something weird — only first split is path
    old = parts[1]
    new = "|||".join(parts[2:])
    # strip one leading newline often added for readability
    if old.startswith("\n"):
        old = old[1:]
    if new.startswith("\n"):
        new = new[1:]
    return path, old, new


def parse_write_payload(body: str) -> Optional[Tuple[str, str]]:
    """WRITE: path|||content"""
    if "|||" not in body:
        return None
    path, content = body.split("|||", 1)
    path = path.strip()
    if content.startswith("\n"):
        content = content[1:]
    return path, content


def parse_grep_payload(body: str) -> Tuple[str, str, Optional[str]]:
    """GREP: pattern  — optional path/glob via ' ||| ' separators (not bare |).

    Examples:
      GREP: def parse_step
      GREP: def parse_step ||| cosmic_cli
      GREP: TODO ||| . ||| glob=*.py

    Bare `|` is regex alternation and must NOT split the payload.
    Also accepts legacy ' | ' (space-pipe-space) separators.
    """
    body = body.strip()
    if "|||" in body:
        parts = [p.strip() for p in body.split("|||")]
    elif " | " in body:
        parts = [p.strip() for p in body.split(" | ")]
    else:
        parts = [body]
    pattern = parts[0] if parts else body
    path = "."
    glob: Optional[str] = None
    for p in parts[1:]:
        if p.startswith("glob="):
            glob = p[5:].strip()
        elif p:
            path = p
    return pattern, path, glob
