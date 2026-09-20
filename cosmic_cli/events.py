"""MissionBus v1 schema: statuses, event names, validation, legacy normalize.

This module is the shared contract. Agent emission, the TUI reducer, and
dashboard readers import it; they do not invent parallel enums.

Migration: writers emit canonical namespaced events. Compatibility aliases
(`start`/`step`/`end`) may be dual-written with compat=True and alias_of set
to the canonical seq. Aliases do not consume a new seq. Readers that still
key on the legacy names keep working; iter_canonical() drops the aliases so
a tape is not counted twice.

Normalization never invents finish_basis.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from cosmic_cli.secrets import redact

SCHEMA_VERSION = 1

# Reserved envelope keys. Payload must not override these.
ENVELOPE_KEYS = ("v", "event", "ts", "session", "mission", "seq")

# A mission that reaches the FINISH path ends in exactly one of these.
# "verified" proves only the check the operator's verifier ran.
FINISHED_STATUSES = ("verified", "needs_review")
FINISH_BASES = (
    "verifier",
    "model_declared",
    "synthesized",
    "verifier_blocked",
)
NON_FINISH_TERMINAL = ("blocked", "passed", "max_steps", "error")
NONTERMINAL_STATUSES = ("ready", "running")
# Read-only paint for pre-split echo records. New writers must not emit it.
LEGACY_STATUSES = ("complete",)

ALL_STATUSES = (
    FINISHED_STATUSES + NON_FINISH_TERMINAL + NONTERMINAL_STATUSES + LEGACY_STATUSES
)

NEEDS_REVIEW_BASES = ("model_declared", "synthesized", "verifier_blocked")

# Canonical catalog used in Phase-1. Unknown types are ignored by tolerant readers.
EVENT_NAMES = (
    "mission.start",
    "mission.status",
    "mission.end",
    "mission.cancel",
    "step.proposed",
    "step.started",
    "step.finished",
    "fs.read",
    "fs.mutate",
    "fs.rollback",
    "shell.exec",
    "verify.result",
    "compass.verdict",
    "gate.pause_minted",
    "gate.pause_resolved",
    "gate.receipt",
    "finish.declared",
    "pass.declared",
    "review.completed",
    "steer",
    "log",
)

LEGACY_TO_CANONICAL = {
    "start": "mission.start",
    "step": "step.proposed",
    "end": "mission.end",
}
CANONICAL_TO_LEGACY = {v: k for k, v in LEGACY_TO_CANONICAL.items()}

# Existing parser vocabulary (agents.STEP_PREFIXES) plus the spec catalog.
ACTION_VERBS = (
    "GLOB",
    "GREP",
    "LIST",
    "READ",
    "DIFF",
    "MKDIR",
    "CREATE",
    "WRITE",
    "EDIT",
    "SHELL",
    "CODE",
    "TEST",
    "TODO",
    "INFO",
    "MEMORY",
    "FINISH",
    "PASS",
)

COMPASS_CLASSES = ("OPEN", "PAUSE", "WITNESS")
PAUSE_DECISIONS = ("approved", "declined", "expired", "invalid")
VERIFY_ROLES = ("verify_cmd", "auto_verify")
SHELL_KINDS = ("SHELL", "CODE", "TEST", "VERIFY_CMD", "AUTO_VERIFY")
MUTATE_OPS = ("EDIT", "WRITE", "CREATE", "MKDIR")

# Credential-bearing keys. Presence with a non-empty value is a schema error.
FORBIDDEN_TOKEN_KEYS = frozenset(
    {
        "token",
        "token_id",
        "approval_token",
        "approval_token_id",
        "pending_token",
        "raw_token",
        "COSMIC_APPROVAL_TOKEN",
    }
)

TEXT_CAPS = {
    "raw": 2000,
    "outcome": 2000,
    "text": 2000,
    "directive": 2000,
    "block_message": 2000,
    "reason": 2000,
    "message": 2000,
    "cmd": 2000,
    "head": 120,
    "result_head": 200,
    "output_head": 500,
    "action_summary": 500,
}
DIFF_CAP_BYTES = 8192
TOKEN_ID_PREFIX_MAX = 8


class EventValidationError(ValueError):
    """Strict schema rejection. Distinct from tolerant normalize_legacy."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def action_verb(raw: str) -> str:
    """First colon-delimited token, uppercased. Empty input → empty string."""
    text = (raw or "").strip()
    if not text:
        return ""
    head = text.splitlines()[0].strip()
    if ":" in head:
        return head.split(":", 1)[0].strip().upper()
    return head.strip().upper()


def action_head(raw: str, cap: int = TEXT_CAPS["head"]) -> str:
    line = (raw or "").splitlines()[0] if raw else ""
    line = line.strip()
    return line if len(line) <= cap else line[:cap]


def finish_basis_for_status(status: str, basis: Optional[str]) -> Optional[str]:
    """Return basis if the pair is legal; raise EventValidationError otherwise.

    Absent basis is represented as None, never a null placeholder in records.
    """
    if status == "verified":
        if basis != "verifier":
            raise EventValidationError(
                "verified requires finish_basis='verifier', "
                f"got {basis!r}"
            )
        return basis
    if status == "needs_review":
        if basis not in NEEDS_REVIEW_BASES:
            raise EventValidationError(
                "needs_review requires finish_basis in "
                f"{NEEDS_REVIEW_BASES}, got {basis!r}"
            )
        return basis
    if basis is not None:
        raise EventValidationError(
            f"status {status!r} must omit finish_basis, got {basis!r}"
        )
    return None


def _looks_like_token_body(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value.startswith("tok-") and len(value) >= 12:
        return True
    # Helix pending tokens are 16 hex with no prefix.
    if len(value) == 16:
        try:
            int(value, 16)
            return True
        except ValueError:
            return False
    return False


def _walk_forbidden(obj: Any, path: str = "") -> Optional[str]:
    if isinstance(obj, Mapping):
        for key, val in obj.items():
            here = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_TOKEN_KEYS and val not in (None, "", []):
                return here
            if key == "token_id_prefix":
                if val is None or val == "":
                    continue
                if not isinstance(val, str) or len(val) > TOKEN_ID_PREFIX_MAX:
                    return here
                if _looks_like_token_body(val):
                    return here
            nested = _walk_forbidden(val, here)
            if nested:
                return nested
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            nested = _walk_forbidden(item, f"{path}[{i}]")
            if nested:
                return nested
    elif _looks_like_token_body(obj) and path:
        # Nested free-text fields may still hold a minted token body.
        leaf = path.rsplit(".", 1)[-1]
        if leaf in FORBIDDEN_TOKEN_KEYS or leaf in {
            "token",
            "pending_token",
            "approval_token_id",
        }:
            return path
    return None


def cap_and_redact_text(value: Any, cap: int) -> str:
    text = redact(str(value if value is not None else ""))
    if cap >= 0 and len(text) > cap:
        return text[:cap]
    return text


def apply_redaction(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Redact then truncate known free-text fields. Redaction runs first."""
    out = dict(payload)
    for key, cap in TEXT_CAPS.items():
        if key in out and out[key] is not None:
            out[key] = cap_and_redact_text(out[key], cap)
    if "diff" in out and out["diff"] is not None:
        diff = redact(str(out["diff"]))
        encoded = diff.encode("utf-8")
        if len(encoded) > DIFF_CAP_BYTES:
            cut = encoded[:DIFF_CAP_BYTES].decode("utf-8", errors="ignore")
            out["diff"] = cut
            out["diff_truncated"] = True
        else:
            out["diff"] = diff
    if "warnings" in out and isinstance(out["warnings"], list):
        out["warnings"] = [
            cap_and_redact_text(w, TEXT_CAPS["outcome"]) for w in out["warnings"]
        ]
    return out


def make_event(
    event: str,
    *,
    session: str,
    mission: str,
    seq: int,
    ts: Optional[str] = None,
    **payload: Any,
) -> Dict[str, Any]:
    """Build a canonical v1 record. Payload cannot override envelope keys."""
    overlap = set(payload) & set(ENVELOPE_KEYS)
    if overlap:
        raise EventValidationError(
            f"payload cannot override envelope keys {sorted(overlap)}"
        )
    rec: Dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "event": event,
        "ts": ts or utc_now_iso(),
        "session": session,
        "mission": mission,
        "seq": seq,
    }
    rec.update(apply_redaction(payload))
    validate_event(rec)
    return rec


def make_compat_alias(canonical: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Legacy dual-write line sharing the canonical seq. Does not consume seq."""
    name = canonical.get("event")
    legacy = CANONICAL_TO_LEGACY.get(str(name)) if name else None
    if not legacy:
        return None
    alias = copy.deepcopy(dict(canonical))
    alias["event"] = legacy
    alias["compat"] = True
    alias["alias_of"] = canonical.get("seq")
    if legacy == "step":
        # Pre-bus readers (review.load_session) look for action starting FINISH:.
        raw = alias.get("raw") or alias.get("action") or ""
        if alias.get("action") in ACTION_VERBS or (
            isinstance(alias.get("action"), str)
            and ":" not in str(alias.get("action", ""))
        ):
            alias["action"] = raw or alias.get("action")
    return alias


def validate_event(record: Mapping[str, Any], *, strict_name: bool = False) -> None:
    """Reject missing envelope, illegal finish pairs, and credential payloads.

    Unknown event types are allowed unless strict_name=True, matching the
    tolerant-reader / optional-strict-validator split in the spec.
    """
    if not isinstance(record, Mapping):
        raise EventValidationError("event must be an object")
    for key in ENVELOPE_KEYS:
        if key not in record:
            raise EventValidationError(f"missing envelope field {key!r}")
    if record["v"] != SCHEMA_VERSION:
        raise EventValidationError(f"unsupported schema v={record['v']!r}")
    event = record["event"]
    if not isinstance(event, str) or not event:
        raise EventValidationError("event name must be a non-empty string")
    if strict_name and event not in EVENT_NAMES and event not in LEGACY_TO_CANONICAL:
        raise EventValidationError(f"unknown event type {event!r}")
    if not isinstance(record["ts"], str) or not record["ts"]:
        raise EventValidationError("ts must be a non-empty string")
    if not isinstance(record["session"], str) or not record["session"]:
        raise EventValidationError("session must be a non-empty string")
    if not isinstance(record["mission"], str) or not record["mission"]:
        raise EventValidationError("mission must be a non-empty string")
    seq = record["seq"]
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise EventValidationError(f"seq must be an int >= 0, got {seq!r}")

    forbidden = _walk_forbidden(record)
    if forbidden:
        raise EventValidationError(f"forbidden credential field {forbidden}")

    status = record.get("status")
    basis = record.get("finish_basis") if "finish_basis" in record else None
    if event in ("mission.end", "end", "finish.declared"):
        if status is None:
            raise EventValidationError(f"{event} requires status")
        if status in LEGACY_STATUSES:
            raise EventValidationError("new writers must not emit status='complete'")
        finish_basis_for_status(str(status), basis)
    elif "finish_basis" in record:
        if status is None:
            raise EventValidationError("finish_basis without status")
        finish_basis_for_status(str(status), basis)

    prefix = record.get("token_id_prefix")
    if prefix not in (None, ""):
        if not isinstance(prefix, str) or len(prefix) > TOKEN_ID_PREFIX_MAX:
            raise EventValidationError("token_id_prefix must be a string of length <= 8")
        if _looks_like_token_body(prefix):
            raise EventValidationError("token_id_prefix must not be a credential body")


def normalize_legacy(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Map start/step/end onto canonical names. Never invent finish_basis.

    Missing v/mission/seq stay absent on old records; callers that need a
    strict envelope must not run those rows through validate_event.
    """
    out = dict(record)
    name = out.get("event")
    if name in LEGACY_TO_CANONICAL:
        out["event"] = LEGACY_TO_CANONICAL[name]
        if name == "step":
            raw = out.get("raw")
            action = out.get("action")
            if raw is None and action is not None:
                raw = str(action)
                out["raw"] = raw
            if action is not None:
                verb = action_verb(str(action))
                if verb:
                    out["action"] = verb
            if out.get("head") is None and raw is not None:
                out["head"] = action_head(str(raw))
    # Old records without v remain readable. Do not fill finish_basis.
    return out


def is_compat_alias(record: Mapping[str, Any]) -> bool:
    return bool(record.get("compat")) or (
        "alias_of" in record and record.get("event") in LEGACY_TO_CANONICAL
    )


def iter_canonical(records: Iterable[Mapping[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Skip dual-write aliases so a pane cannot count one step twice."""
    for rec in records:
        if is_compat_alias(rec):
            continue
        yield normalize_legacy(rec)


def tape_identity(record: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    """Dedupe key: mission + seq + canonical event family."""
    name = record.get("event")
    canonical = LEGACY_TO_CANONICAL.get(str(name), name)
    seq = record.get("alias_of", record.get("seq"))
    return (record.get("mission"), seq, canonical)


def unique_mission_stem(
    session_id: str,
    *,
    when: Optional[datetime] = None,
    nonce: Optional[str] = None,
) -> str:
    """Collision-resistant mission file stem. Does not change session_id.

    Approvals stay keyed by session_id. The stem is the JSONL identity.
    Microseconds replace the previous second-resolution stamp so two
    concurrent directives in the same UTC second do not share a file.
    """
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    if nonce:
        return f"{session_id}__{stamp}_{nonce}"
    return f"{session_id}__{stamp}"
