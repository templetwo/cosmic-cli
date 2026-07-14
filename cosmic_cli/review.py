"""Second-pass review seat — adversarial read of a mission's edits.

Separate from the builder loop: same model family is fine, different prompt.
Reviews diffs only (plus optional test output), never re-implements.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xai_sdk import Client
from xai_sdk.chat import system, user

from cosmic_cli.agents import DEFAULT_MODEL, SESSION_DIR


REVIEW_SYSTEM = """You are an independent code reviewer for a coding agent mission.
You did NOT write this change. You are Cold Eye: find what breaks before reality does.

Rules:
- Only judge from the provided diffs and context. Do not invent files you cannot see.
- Prefer concrete, falsifiable findings over style nits.
- Severity: FATAL (wrong/broken/security), WARN (likely issue), NOTE (nit/suggestion).
- If nothing real is wrong, say CLEAR with residual risk.
- Output strict JSON only, no markdown fences:
{
  "verdict": "CLEAR" | "WARN" | "BLOCK",
  "findings": [{"severity":"FATAL|WARN|NOTE","title":"...","detail":"...","path":"..."}],
  "residual_risk": "...",
  "suggested_tests": ["..."]
}
"""


@dataclass
class ReviewBundle:
    paths: List[str] = field(default_factory=list)
    diffs: Dict[str, str] = field(default_factory=dict)
    directive: str = ""
    session_id: str = ""
    finish_text: str = ""


def collect_diff(root: Path, path: str) -> Optional[str]:
    """Diff path.cosmicbak vs path, or git diff if no backup."""
    target = root / path
    bak = root / f"{path}.cosmicbak"
    if bak.is_file() and target.is_file():
        r = subprocess.run(
            ["diff", "-u", str(bak), str(target)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.stdout or r.stderr or "(no diff output)"
    # fall back to git
    r = subprocess.run(
        ["git", "diff", "--", path],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(root),
    )
    if r.returncode in (0, 1) and (r.stdout or "").strip():
        return r.stdout
    if target.is_file():
        try:
            body = target.read_text(encoding="utf-8")
            return f"--- /dev/null\n+++ {path}\n@@ full file @@\n" + body[:8000]
        except OSError:
            return None
    return None


def load_session(session_id: str) -> Dict[str, Any]:
    path = SESSION_DIR / f"{session_id}.jsonl"
    if not path.is_file():
        return {}
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    edited: List[str] = []
    directive = ""
    finish = ""
    for e in events:
        if e.get("event") == "start":
            directive = e.get("directive", directive)
        if e.get("event") == "end":
            edited = list(e.get("edited") or edited)
        if e.get("event") == "step" and str(e.get("action", "")).upper().startswith(
            "FINISH:"
        ):
            finish = str(e.get("action", ""))
    # also gather edited from any end-like fields
    for e in reversed(events):
        if e.get("edited"):
            edited = list(e["edited"])
            break
    return {
        "events": events,
        "edited": edited,
        "directive": directive,
        "finish": finish,
        "session_id": session_id,
    }


def latest_session_id() -> Optional[str]:
    if not SESSION_DIR.is_dir():
        return None
    files = sorted(SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    return files[-1].stem


def build_bundle(
    root: Path,
    *,
    session_id: Optional[str] = None,
    paths: Optional[List[str]] = None,
    directive: str = "",
) -> ReviewBundle:
    bundle = ReviewBundle(directive=directive)
    if session_id:
        meta = load_session(session_id)
        bundle.session_id = session_id
        bundle.directive = meta.get("directive") or directive
        bundle.finish_text = meta.get("finish") or ""
        paths = paths or list(meta.get("edited") or [])
    bundle.paths = list(paths or [])
    for p in bundle.paths:
        d = collect_diff(root, p)
        if d:
            bundle.diffs[p] = d[:12000]
    return bundle


def run_review(
    bundle: ReviewBundle,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    if not bundle.diffs and not bundle.paths:
        return {
            "verdict": "CLEAR",
            "findings": [],
            "residual_risk": "Nothing to review (no diffs/paths).",
            "suggested_tests": [],
        }

    diff_blob = "\n\n".join(
        f"### {path}\n```diff\n{diff}\n```" for path, diff in bundle.diffs.items()
    )
    prompt = f"""Mission directive:
{bundle.directive or "(unknown)"}

Session: {bundle.session_id or "(none)"}
Finish claim:
{bundle.finish_text or "(none)"}

Diffs under review:
{diff_blob or "(no unified diff — paths only: " + ", ".join(bundle.paths) + ")"}

Review now. JSON only.
"""
    client = Client(api_key=api_key)
    chat = client.chat.create(model=model, temperature=0.0)
    chat.append(system(REVIEW_SYSTEM))
    chat.append(user(prompt))
    response = chat.sample()
    raw = response.content.strip() if hasattr(response, "content") else str(response)
    # tolerate accidental fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "verdict": "WARN",
            "findings": [
                {
                    "severity": "WARN",
                    "title": "reviewer_non_json",
                    "detail": raw[:2000],
                    "path": "",
                }
            ],
            "residual_risk": "Reviewer did not return parseable JSON.",
            "suggested_tests": [],
            "raw": raw[:4000],
        }
    data["_meta"] = {
        "model": model,
        "session": bundle.session_id,
        "paths": bundle.paths,
    }
    return data
