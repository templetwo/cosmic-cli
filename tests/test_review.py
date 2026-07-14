import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from cosmic_cli.review import (
    build_bundle,
    collect_diff,
    load_session,
    run_review,
)


def test_collect_diff_from_backup(tmp_path: Path):
    p = tmp_path / "a.py"
    bak = tmp_path / "a.py.cosmicbak"
    bak.write_text("x = 1\n", encoding="utf-8")
    p.write_text("x = 2\n", encoding="utf-8")
    d = collect_diff(tmp_path, "a.py")
    assert d is not None
    assert "x = 1" in d or "+x = 2" in d or "2" in d


def test_build_bundle_and_session(tmp_path: Path, monkeypatch):
    sess_dir = tmp_path / "sessions"
    sess_dir.mkdir()
    monkeypatch.setattr("cosmic_cli.review.SESSION_DIR", sess_dir)
    sid = "testsession"
    lines = [
        {"event": "start", "directive": "change x", "session": sid},
        {"event": "end", "edited": ["a.py"], "status": "complete"},
    ]
    (sess_dir / f"{sid}.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
    )
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "a.py.cosmicbak").write_text("x = 1\n", encoding="utf-8")
    meta = load_session(sid)
    assert meta["edited"] == ["a.py"]
    assert meta["directive"] == "change x"
    bundle = build_bundle(tmp_path, session_id=sid)
    assert "a.py" in bundle.paths
    assert "a.py" in bundle.diffs


def test_run_review_parses_json():
    bundle = build_bundle(Path("."), paths=[])
    # empty → CLEAR without API
    report = run_review(bundle, api_key="x")
    assert report["verdict"] == "CLEAR"

    # with diffs, mock client
    from cosmic_cli.review import ReviewBundle

    b = ReviewBundle(
        paths=["f.py"],
        diffs={"f.py": "--- a\n+++ b\n+x=1\n"},
        directive="add x",
    )
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(
        {
            "verdict": "WARN",
            "findings": [
                {
                    "severity": "WARN",
                    "title": "no test",
                    "detail": "missing unit test",
                    "path": "f.py",
                }
            ],
            "residual_risk": "low",
            "suggested_tests": ["pytest"],
        }
    )
    mock_chat = MagicMock()
    mock_chat.sample.return_value = mock_resp
    mock_client = MagicMock()
    mock_client.chat.create.return_value = mock_chat
    with patch("cosmic_cli.review.Client", return_value=mock_client):
        report = run_review(b, api_key="test")
    assert report["verdict"] == "WARN"
    assert report["findings"][0]["title"] == "no test"
