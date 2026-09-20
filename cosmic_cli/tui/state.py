"""Pure Pilot Board reducer. No I/O, no Textual widgets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional

from cosmic_cli.events import (
    FINISHED_STATUSES,
    NEEDS_REVIEW_BASES,
    NON_FINISH_TERMINAL,
    NONTERMINAL_STATUSES,
    is_compat_alias,
    normalize_legacy,
)

# Matches cosmic_cli.agents.MAX_STEPS_DEFAULT without importing the agent.
_DEFAULT_MAX_STEPS = 30

MISSION_STATUSES = (
    "ready",
    "running",
    "verified",
    "needs_review",
    "blocked",
    "error",
    "max_steps",
    "passed",
)
# Read-only paint if a tape still carries it. The reducer never originates it.
LEGACY_PAINT_STATUS = "complete"


@dataclass(frozen=True)
class StepEvent:
    # Event timestamp. UI arrival HH:MM:SS is paint, not reducer state.
    ts: str
    kind: str
    summary: str
    path: Optional[str] = None
    compass: Optional[str] = None
    receipt_id: Optional[str] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class Mission:
    key: str
    directive: str = ""
    status: str = "ready"
    model: str = ""
    steps_taken: int = 0
    max_steps: int = _DEFAULT_MAX_STEPS
    finish_basis: Optional[str] = None
    review_mode: bool = False
    verify_cmd: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    steps: List[StepEvent] = field(default_factory=list)
    last_mutation_path: Optional[str] = None
    last_checkpoint: Optional[str] = None
    last_diff: Optional[str] = None
    session: Optional[str] = None
    exec_mode: Optional[str] = None


@dataclass(frozen=True)
class PendingPause:
    """Allowlisted fields only. Credential keys never become attributes."""

    action_summary: str
    mission_key: str
    pending_id: Optional[Any] = None
    action_sha256: Optional[str] = None
    rule: Optional[str] = None
    expires_at: Optional[str] = None
    channel: str = "local"


@dataclass(frozen=True)
class BoardState:
    missions: Dict[str, Mission] = field(default_factory=dict)
    selected_key: Optional[str] = None
    filter: str = "all"
    pending_pauses: List[PendingPause] = field(default_factory=list)
    compass_today: Dict[str, int] = field(default_factory=dict)
    compass_total: Dict[str, int] = field(default_factory=dict)
    goal: Optional[str] = None
    helix_on: bool = True
    # None = unknown. Do not treat unknown as healthy.
    floor_ok: Optional[bool] = None
    review_report: Optional[Any] = None
    verify_cmd_default: Optional[str] = None
    review_default: bool = False


def apply_event(state: BoardState, event: dict) -> BoardState:
    """Return a new BoardState. Compat aliases are dropped (no double-count)."""
    if not isinstance(event, Mapping):
        return state
    if is_compat_alias(event):
        return state
    rec = normalize_legacy(event)
    name = rec.get("event")
    if name == "mission.start":
        return _apply_start(state, rec)
    if name == "mission.status":
        return _apply_status(state, rec)
    if name == "step.proposed":
        return _apply_step_proposed(state, rec)
    if name == "compass.verdict":
        return _apply_compass(state, rec)
    if name == "gate.pause_minted":
        return _apply_pause_minted(state, rec)
    if name == "gate.pause_resolved":
        return _apply_pause_resolved(state, rec)
    if name == "fs.mutate":
        return _apply_fs_mutate(state, rec)
    if name in ("finish.declared", "mission.end"):
        return _apply_finish(state, rec)
    return state


def _opt_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def _mission_key(rec: Mapping[str, Any]) -> Optional[str]:
    key = rec.get("mission")
    if isinstance(key, str) and key:
        return key
    return None


def _copy_mission(mission: Mission) -> Mission:
    return replace(mission, logs=list(mission.logs), steps=list(mission.steps))


def _put(
    state: BoardState,
    key: str,
    mission: Mission,
    **board: Any,
) -> BoardState:
    missions = dict(state.missions)
    missions[key] = mission
    return replace(state, missions=missions, **board)


def _touch(state: BoardState, rec: Mapping[str, Any]) -> Optional[tuple[str, Mission]]:
    key = _mission_key(rec)
    if key is None:
        return None
    mission = state.missions.get(key)
    if mission is None:
        return None
    return key, mission


def _int_n(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _apply_start(state: BoardState, rec: Mapping[str, Any]) -> BoardState:
    key = _mission_key(rec)
    if key is None:
        return state
    max_steps = _int_n(rec.get("max_steps"))
    if max_steps is None:
        max_steps = _DEFAULT_MAX_STEPS
    verify_cmd = rec.get("verify_cmd")
    if isinstance(verify_cmd, str):
        verify_cmd = verify_cmd.strip() or None
    else:
        verify_cmd = None
    mission = Mission(
        key=key,
        directive=str(rec.get("directive") or ""),
        status="ready",
        model=str(rec.get("model") or ""),
        max_steps=max_steps,
        review_mode=bool(rec.get("review", False)),
        verify_cmd=verify_cmd,
        session=_opt_str(rec.get("session")),
        exec_mode=_opt_str(rec.get("exec_mode")),
    )
    selected = state.selected_key if state.selected_key else key
    return _put(state, key, mission, selected_key=selected)


def _apply_status(state: BoardState, rec: Mapping[str, Any]) -> BoardState:
    found = _touch(state, rec)
    if found is None:
        return state
    key, mission = found
    status = rec.get("status")
    if not isinstance(status, str) or not status:
        return state
    if status in NONTERMINAL_STATUSES:
        return _put(state, key, replace(_copy_mission(mission), status=status))
    return _put(state, key, _apply_terminal(mission, rec))


def _apply_step_proposed(state: BoardState, rec: Mapping[str, Any]) -> BoardState:
    found = _touch(state, rec)
    if found is None:
        return state
    key, mission = found
    n = _int_n(rec.get("n"))
    taken = mission.steps_taken if n is None else max(mission.steps_taken, n)
    step = StepEvent(
        ts=str(rec.get("ts") or ""),
        kind=str(rec.get("action") or "step.proposed"),
        summary=str(rec.get("head") or rec.get("raw") or rec.get("action") or ""),
        path=_opt_str(rec.get("path")),
        receipt_id=_opt_str(rec.get("receipt_id")),
        detail=_opt_str(rec.get("raw")),
    )
    updated = replace(_copy_mission(mission), steps_taken=taken)
    updated = replace(updated, steps=updated.steps + [step])
    return _put(state, key, updated)


def _apply_compass(state: BoardState, rec: Mapping[str, Any]) -> BoardState:
    classification = rec.get("classification")
    today = dict(state.compass_today)
    if isinstance(classification, str) and classification:
        today[classification] = today.get(classification, 0) + 1
    found = _touch(state, rec)
    if found is None:
        if today == state.compass_today:
            return state
        return replace(state, compass_today=today)
    key, mission = found
    step = StepEvent(
        ts=str(rec.get("ts") or ""),
        kind=str(rec.get("tool_name") or "compass.verdict"),
        summary=str(
            rec.get("action_summary") or rec.get("reason") or classification or ""
        ),
        compass=str(classification) if classification else None,
        receipt_id=_opt_str(rec.get("receipt_id")),
        detail=_opt_str(rec.get("reason")),
    )
    updated = _copy_mission(mission)
    updated = replace(updated, steps=updated.steps + [step])
    return _put(state, key, updated, compass_today=today)


def _apply_pause_minted(state: BoardState, rec: Mapping[str, Any]) -> BoardState:
    channel = rec.get("channel")
    if channel not in ("local", "helix", "gate"):
        channel = "local"
    pause = PendingPause(
        pending_id=rec.get("pending_id"),
        action_sha256=_opt_str(rec.get("action_sha256")),
        action_summary=str(rec.get("action_summary") or ""),
        rule=_opt_str(rec.get("rule") or rec.get("rule_matched")),
        mission_key=str(_mission_key(rec) or ""),
        expires_at=_opt_str(rec.get("expires_at")),
        channel=channel,
    )
    pending = list(state.pending_pauses)
    pending.append(pause)
    return replace(state, pending_pauses=pending)


def _pause_match(pause: PendingPause, rec: Mapping[str, Any]) -> bool:
    pending_id = rec.get("pending_id")
    sha = rec.get("action_sha256")
    if pending_id is not None and pause.pending_id is not None:
        if pending_id == pause.pending_id:
            return True
    if sha and pause.action_sha256 and sha == pause.action_sha256:
        return True
    return False


def _apply_pause_resolved(state: BoardState, rec: Mapping[str, Any]) -> BoardState:
    if rec.get("pending_id") is None and not rec.get("action_sha256"):
        return state
    pending = [p for p in state.pending_pauses if not _pause_match(p, rec)]
    if len(pending) == len(state.pending_pauses):
        return state
    return replace(state, pending_pauses=pending)


def _apply_fs_mutate(state: BoardState, rec: Mapping[str, Any]) -> BoardState:
    found = _touch(state, rec)
    if found is None:
        return state
    key, mission = found
    updates: Dict[str, Any] = {}
    path = rec.get("path") or rec.get("rel")
    if path:
        updates["last_mutation_path"] = str(path)
    if "checkpoint_id" in rec:
        updates["last_checkpoint"] = _opt_str(rec.get("checkpoint_id"))
    if "diff" in rec:
        diff = rec.get("diff")
        updates["last_diff"] = None if diff is None else str(diff)
    if not updates:
        return state
    return _put(state, key, replace(_copy_mission(mission), **updates))


def _apply_finish(state: BoardState, rec: Mapping[str, Any]) -> BoardState:
    found = _touch(state, rec)
    if found is None:
        return state
    key, mission = found
    if not isinstance(rec.get("status"), str) or not rec.get("status"):
        return state
    return _put(state, key, _apply_terminal(mission, rec))


def _apply_terminal(mission: Mission, rec: Mapping[str, Any]) -> Mission:
    status = str(rec.get("status"))
    raw_basis = rec.get("finish_basis") if "finish_basis" in rec else None
    basis = raw_basis if raw_basis not in (None, "") else None
    copied = _copy_mission(mission)

    if status in NON_FINISH_TERMINAL:
        # blocked / max_steps / passed / error never carry a basis.
        return replace(copied, status=status, finish_basis=None)
    if status in NONTERMINAL_STATUSES:
        return replace(copied, status=status)
    if status == LEGACY_PAINT_STATUS:
        return replace(copied, status=LEGACY_PAINT_STATUS)

    if basis == "verifier":
        return replace(copied, status="verified", finish_basis="verifier")
    if status == "verified":
        if basis in NEEDS_REVIEW_BASES:
            return replace(copied, status="needs_review", finish_basis=basis)
        return replace(copied, status="needs_review", finish_basis=None)
    if status == "needs_review":
        finish_basis = basis if basis in NEEDS_REVIEW_BASES else None
        return replace(copied, status="needs_review", finish_basis=finish_basis)
    if status in FINISHED_STATUSES:
        return replace(copied, status=status, finish_basis=None)
    return replace(copied, status=status)
