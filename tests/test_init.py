from pathlib import Path

from cosmic_cli.init_cmd import write_init


def test_init_writes_cosmic_md(tmp_path: Path):
    msg, ok = write_init(tmp_path)
    assert ok
    p = tmp_path / "COSMIC.md"
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "Stargazer" in text
    assert "CREATE" in text or "READ" in text
    # no force → refuse overwrite
    msg2, ok2 = write_init(tmp_path)
    assert not ok2
    msg3, ok3 = write_init(tmp_path, force=True)
    assert ok3
