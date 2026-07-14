from pathlib import Path

import pytest

from cosmic_cli.tools import (
    normalize_user_path,
    parse_edit_payload,
    parse_grep_payload,
    rel_key,
    tool_edit,
    tool_glob,
    tool_grep,
    tool_list,
    tool_write,
)


def test_normalize_absolute_under_root(tmp_path: Path):
    """Regression: lstrip('./') must NOT strip absolute path slashes."""
    desktop = tmp_path / "Desktop" / "notes"
    desktop.mkdir(parents=True)
    abs_path = str(desktop / "better-cli.md")
    # classic bug: "/Users/...".lstrip("./") -> "Users/..."
    assert abs_path.lstrip("./") != abs_path or abs_path.startswith("/")
    target = normalize_user_path(abs_path, tmp_path)
    assert target == (tmp_path / "Desktop" / "notes" / "better-cli.md").resolve()
    assert rel_key(abs_path, tmp_path) == "Desktop/notes/better-cli.md"
    assert rel_key("Desktop/notes/better-cli.md", tmp_path) == "Desktop/notes/better-cli.md"
    assert rel_key("./Desktop/notes/better-cli.md", tmp_path) == "Desktop/notes/better-cli.md"


def test_normalize_rejects_escape(tmp_path: Path):
    with pytest.raises(PermissionError):
        normalize_user_path("/etc/passwd", tmp_path)


def test_write_absolute_lands_correctly(tmp_path: Path):
    abs_path = str(tmp_path / "Desktop" / "x.md")
    obs, ok = tool_write(tmp_path, abs_path, "# hi\n")
    assert ok
    assert (tmp_path / "Desktop" / "x.md").read_text(encoding="utf-8") == "# hi\n"
    # must NOT create nested Users/... style garbage
    assert not (tmp_path / "Users").exists()


def test_list_and_glob(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")
    listing = tool_list(tmp_path, ".")
    assert "a.py" in listing
    g = tool_glob(tmp_path, "**/*.py")
    assert "a.py" in g
    assert "sub/b.py" in g or "sub" in g


def test_grep(tmp_path: Path):
    (tmp_path / "hit.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    out = tool_grep(tmp_path, r"def hello", ".")
    assert "hit.py" in out
    assert "def hello" in out


def test_edit_requires_unique(tmp_path: Path):
    p = tmp_path / "f.py"
    p.write_text("aa\nbb\naa\n", encoding="utf-8")
    obs, ok = tool_edit(tmp_path, "f.py", "aa", "zz")
    assert not ok
    assert "2 times" in obs
    obs, ok = tool_edit(tmp_path, "f.py", "bb\n", "cc\n")
    assert ok
    assert p.read_text(encoding="utf-8") == "aa\ncc\naa\n"
    assert (tmp_path / "f.py.cosmicbak").exists()


def test_edit_rejects_truncated_python(tmp_path: Path):
    p = tmp_path / "n.py"
    p.write_text('greeting = "hello"\n', encoding="utf-8")
    obs, ok = tool_edit(
        tmp_path,
        "n.py",
        'greeting = "hello"\n',
        'greeting = "hello cosmos\n',  # missing closing quote
    )
    assert not ok
    assert "truncated" in obs.lower() or "syntax" in obs.lower()
    assert p.read_text(encoding="utf-8") == 'greeting = "hello"\n'


def test_write_and_parse():
    assert parse_edit_payload("p.py|||old|||new") == ("p.py", "old", "new")
    assert parse_grep_payload("foo|bar") == ("foo|bar", ".", None)
    assert parse_grep_payload("foo ||| src ||| glob=*.py") == ("foo", "src", "*.py")
    assert parse_grep_payload("foo | src") == ("foo", "src", None)
    parsed = parse_edit_payload("a|||x|||y|||z")
    assert parsed is not None
    assert parsed[0] == "a"
    assert parsed[1] == "x"
    assert parsed[2] == "y|||z"


def test_write_tool(tmp_path: Path):
    obs, ok = tool_write(tmp_path, "new.txt", "hello")
    assert ok
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello"
