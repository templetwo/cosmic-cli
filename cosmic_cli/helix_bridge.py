"""T2Helix bridge — local memory/task substrate (not Sovereign Stack).

Uses Node RPC into templetwo/t2helix (grok-adapter + chronicle).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RPC_SCRIPT = PACKAGE_ROOT / "scripts" / "helix_rpc.js"


def _default_data_dirs() -> List[Path]:
    home = Path.home()
    return [
        home / ".claude" / "plugins" / "data" / "t2helix-templetwo-t2helix",
        home / ".t2helix-data",
    ]


def resolve_t2helix_root() -> Optional[Path]:
    env = os.environ.get("T2HELIX_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    home = Path.home()
    for c in (home / "t2helix", home / "code" / "t2helix", PACKAGE_ROOT.parent / "t2helix"):
        if (c / "lib" / "grok-adapter.js").is_file():
            return c
    return None


def resolve_data_dir() -> Path:
    env = os.environ.get("T2HELIX_DATA_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    for d in _default_data_dirs():
        if d.is_dir():
            return d
    fallback = Path.home() / ".t2helix-data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def available() -> bool:
    if not shutil.which("node"):
        return False
    if not RPC_SCRIPT.is_file():
        return False
    root = resolve_t2helix_root()
    return root is not None and (root / "lib" / "grok-adapter.js").is_file()


def call(op: str, timeout: float = 30.0, **kwargs: Any) -> Dict[str, Any]:
    """Call Helix RPC. Always returns a dict with ok bool."""
    if not available():
        return {
            "ok": False,
            "error": "t2helix unavailable (need node + T2HELIX_ROOT checkout)",
            "op": op,
        }
    root = resolve_t2helix_root()
    data = resolve_data_dir()
    env = os.environ.copy()
    env["T2HELIX_ROOT"] = str(root)
    env["T2HELIX_DATA_DIR"] = str(data)
    payload = {"op": op, **kwargs}
    try:
        proc = subprocess.run(
            ["node", str(RPC_SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(root),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "helix rpc timeout", "op": op}
    except OSError as e:
        return {"ok": False, "error": str(e), "op": op}

    out = (proc.stdout or "").strip()
    if not out:
        return {
            "ok": False,
            "error": f"empty rpc response (exit {proc.returncode}): {proc.stderr[:300]}",
            "op": op,
        }
    try:
        data_out = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": f"bad json from rpc: {out[:400]}",
            "stderr": (proc.stderr or "")[:400],
            "op": op,
        }
    return data_out


def health() -> Dict[str, Any]:
    return call("health")


def boot(query: str, *, top_k: int = 6) -> Dict[str, Any]:
    return call("boot", query=query, topK=top_k)


def recall(query: str, *, top_k: int = 6) -> Dict[str, Any]:
    return call("recall", query=query, topK=top_k)


def record(
    content: str,
    *,
    session_id: Optional[str] = None,
    domain: str = "cosmic-cli",
    layer: str = "hypothesis",
    tags: Optional[List[str]] = None,
    intensity: float = 0.6,
) -> Dict[str, Any]:
    return call(
        "record",
        content=content,
        session_id=session_id,
        domain=domain,
        layer=layer,
        tags=tags or ["source:cosmic-cli"],
        intensity=intensity,
    )


def witness(action: str, *, session_id: Optional[str] = None) -> Dict[str, Any]:
    return call("witness", action=action, session_id=session_id)


def set_goal(goal: str, *, session_id: Optional[str] = None, why: str = "cosmic-cli") -> Dict[str, Any]:
    return call("set_goal", goal=goal, session_id=session_id, why=why)


def open_thread(question: str, *, domain: str = "cosmic-cli", context: str = "") -> Dict[str, Any]:
    return call("open_thread", question=question, domain=domain, context=context)


def get_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    return call("get_state", session_id=session_id)


def format_boot_context(boot_result: Dict[str, Any], limit: int = 4) -> str:
    """Compact text for injection into Stargazer prompt."""
    if not boot_result.get("ok"):
        return f"[helix offline: {boot_result.get('error', 'unknown')}]"
    inner = boot_result.get("result") or {}
    lines = [
        f"T2Helix boot · session={inner.get('sessionId')} · "
        f"memory={inner.get('memoryCount', 0)} · data={inner.get('dataDir')}"
    ]
    mem = inner.get("memory") or []
    for i, m in enumerate(mem[:limit]):
        if isinstance(m, dict):
            content = (m.get("content") or m.get("text") or str(m))[:240]
            layer = m.get("layer", "?")
            lines.append(f"  [{layer}] {content}")
        else:
            lines.append(f"  {str(m)[:240]}")
    return "\n".join(lines)


def format_state_context(state_result: Dict[str, Any], limit: int = 5) -> str:
    """Open threads + goal from get_state for agent context."""
    if not state_result.get("ok"):
        return ""
    inner = state_result.get("result") or {}
    lines: List[str] = []
    goal = inner.get("goal") or inner.get("session_goal")
    if isinstance(goal, dict):
        gtext = goal.get("goal") or goal.get("text") or str(goal)
        lines.append(f"Active goal: {str(gtext)[:200]}")
    elif goal:
        lines.append(f"Active goal: {str(goal)[:200]}")
    threads = inner.get("open_threads") or inner.get("threads") or []
    if threads:
        lines.append("Open threads:")
        for t in threads[:limit]:
            if isinstance(t, dict):
                tid = t.get("id", "?")
                q = t.get("question") or t.get("content") or str(t)
                lines.append(f"  #{tid}: {str(q)[:160]}")
            else:
                lines.append(f"  {str(t)[:160]}")
    return "\n".join(lines)


def load_project_notes(cwd: Optional[Path] = None) -> str:
    """Load COSMIC.md / AGENTS.md from cwd if present."""
    root = cwd or Path.cwd()
    for name in ("COSMIC.md", "AGENTS.md", "CLAUDE.md"):
        p = root / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                return f"Project notes ({name}):\n{text[:3000]}"
            except OSError:
                continue
    return ""
