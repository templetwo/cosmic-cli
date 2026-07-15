"""Typed, validated, deterministic policy kernel for Cosmic CLI.

Corrective Phase 3+4 slice (2026-07-15). Assembled from policy_kernel_v0.4.

Implements:
- Phase 0 truthful kernel (validation, typed rules, real hashes)
- Phase 1 monotonic composition (compose_policy_layers with escalation-only)
- Phase 3 explicit MatchType dispatch (LITERAL + REGEX real; all other
  MatchTypes currently fall back to literal substring — they are NOT yet
  rejected at load)
- Phase 4 approval lifecycle (ApprovalManager prototype in gateway.py; the
  gateway enforces the PAUSE token requirement and consumes receipts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Tuple
import hashlib
import json

class Disposition(str, Enum):
    WITNESS = "WITNESS"
    PAUSE = "PAUSE"
    OPEN = "OPEN"

class ActionType(str, Enum):
    SHELL = "SHELL"
    CODE = "CODE"
    READ = "READ"
    WRITE = "WRITE"
    EDIT = "EDIT"
    NETWORK = "NETWORK"
    DELETE = "DELETE"
    GIT = "GIT"
    PUBLISH = "PUBLISH"

class MatchType(str, Enum):
    LITERAL = "literal"
    TOKEN = "token"
    GLOB = "glob"
    REGEX = "regex"
    PATH = "path"
    COMMAND_AST = "command_ast"
    NETWORK_DEST = "network_dest"
    SECRET_SHAPE = "secret_shape"

VALID_DISPOSITIONS: FrozenSet[Disposition] = frozenset(Disposition)
VALID_ACTION_TYPES: FrozenSet[ActionType] = frozenset(ActionType)

class PolicyValidationError(ValueError):
    """Raised when policy is malformed or contains unknown values."""
    pass

@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    disposition: Disposition
    scopes: FrozenSet[ActionType]
    match_type: MatchType
    pattern: str
    source: str = ""
    priority: int = 0
    enabled: bool = True

    def __post_init__(self):
        if not self.rule_id or not self.pattern:
            raise PolicyValidationError("rule_id and pattern are required")
        if self.disposition not in VALID_DISPOSITIONS:
            raise PolicyValidationError(f"Unknown disposition: {self.disposition}")
        if not self.scopes:
            raise PolicyValidationError("scopes cannot be empty")
        for s in self.scopes:
            if s not in VALID_ACTION_TYPES:
                raise PolicyValidationError(f"Unknown action type in scopes: {s}")

SEVERITY = {
    Disposition.OPEN: 0,
    Disposition.PAUSE: 1,
    Disposition.WITNESS: 2,
}

@dataclass(frozen=True)
class RuleMatch:
    rule: PolicyRule
    matched_value: str

@dataclass(frozen=True)
class PolicyDecision:
    disposition: Disposition
    matches: Tuple[RuleMatch, ...]
    default_used: bool
    policy_sha256: str
    evaluated_input_sha256: str
    evaluator_version: str = "0.2-phase0"

def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def validate_and_load_rules(raw_rules: List[dict]) -> List[PolicyRule]:
    """Validate every rule. Raise on unknown disposition or bad data.

    Missing disposition/type is an error (not silent OPEN).
    """
    validated: List[PolicyRule] = []
    for r in raw_rules:
        raw_disp = r.get("disposition") if r.get("disposition") is not None else r.get("type")
        if raw_disp is None or str(raw_disp).strip() == "":
            raise PolicyValidationError(
                f"Missing disposition on rule {r.get('id') or r.get('rule_id') or '?'}"
            )
        try:
            disp = Disposition(str(raw_disp).strip().upper())
        except ValueError:
            raise PolicyValidationError(f"Unknown disposition: {raw_disp}")

        scopes_raw = r.get("scopes") if r.get("scopes") is not None else r.get("scope", "SHELL")
        if isinstance(scopes_raw, str):
            scopes = frozenset(
                ActionType(s.strip()) for s in scopes_raw.split(",") if s.strip()
            )
        else:
            scopes = frozenset(ActionType(s) for s in scopes_raw)

        match_t = MatchType(r.get("match_type", "literal"))

        rule = PolicyRule(
            rule_id=str(r.get("id") or r.get("rule_id", "")),
            disposition=disp,
            scopes=scopes,
            match_type=match_t,
            pattern=str(r.get("pattern", "")),
            source=str(r.get("source", r.get("raw", ""))),
            priority=int(r.get("priority", 0)),
            enabled=bool(r.get("enabled", True)),
        )
        validated.append(rule)
    return validated


def policy_fingerprint(rules: List[PolicyRule]) -> str:
    """Stable hash of a rule set for receipts (not a placeholder string)."""
    payload = [
        {
            "id": r.rule_id,
            "disposition": r.disposition.value,
            "scopes": sorted(s.value for s in r.scopes),
            "match_type": r.match_type.value,
            "pattern": r.pattern,
            "priority": r.priority,
            "enabled": r.enabled,
        }
        for r in rules
    ]
    return _compute_sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

def evaluate_rules(
    rules: List[PolicyRule],
    action_type: ActionType,
    input_str: str,
    policy_sha256: str = ""
) -> PolicyDecision:
    """
    Pure evaluation.
    - Only rules whose scopes include the action_type are considered.
    - All matching rules are returned (no silent override).
    - Strongest disposition wins via explicit lattice.
    """
    if action_type not in VALID_ACTION_TYPES:
        raise PolicyValidationError(f"Unknown action type: {action_type}")

    matches: List[RuleMatch] = []
    normalized_input = input_str.lower()

    for rule in rules:
        if not rule.enabled:
            continue
        if action_type not in rule.scopes:
            continue

        # Phase 3 explicit match_type dispatch (literal default, regex supported, others fall back)
        matched = False
        if rule.match_type == MatchType.LITERAL:
            matched = rule.pattern.lower() in normalized_input
        elif rule.match_type == MatchType.REGEX:
            import re
            matched = bool(re.search(rule.pattern, input_str, re.IGNORECASE))
        else:
            # TOKEN, GLOB, PATH, COMMAND_AST, etc. — literal fallback for Phase 3 prototype
            matched = rule.pattern.lower() in normalized_input

        if matched:
            matches.append(RuleMatch(rule=rule, matched_value=rule.pattern))

    pol_hash = policy_sha256 or (
        policy_fingerprint(rules) if rules else _compute_sha256(b"empty-policy")
    )

    if not matches:
        # Explicit default: OPEN. NOTE: this is fail-open — a true fail-closed
        # default remains future work (v0.4 comment claimed fail-closed; it was not).
        return PolicyDecision(
            disposition=Disposition.OPEN,
            matches=(),
            default_used=True,
            policy_sha256=pol_hash,
            evaluated_input_sha256=_compute_sha256(input_str.encode("utf-8")),
        )

    strongest = max(matches, key=lambda m: SEVERITY[m.rule.disposition])
    return PolicyDecision(
        disposition=strongest.rule.disposition,
        matches=tuple(matches),
        default_used=False,
        policy_sha256=pol_hash,
        evaluated_input_sha256=_compute_sha256(input_str.encode("utf-8")),
    )


@dataclass(frozen=True)
class ComposedPolicyRule:
    """Effective rule after monotonic composition across layers."""
    effective_rule: PolicyRule
    source_rules: Tuple[PolicyRule, ...]
    source_layers: Tuple[str, ...]


def _rule_equivalence_key(rule: PolicyRule) -> tuple:
    """Canonical key for equivalence: scopes + match_type + normalized pattern."""
    return (
        frozenset(rule.scopes),
        rule.match_type,
        rule.pattern.lower().strip()
    )


def compose_policy_layers(
    *layers: List[PolicyRule],
    layer_names: Tuple[str, ...] = ()
) -> List[ComposedPolicyRule]:
    """
    Monotonic policy composition with equivalence key and provenance (Phase 1).

    Equivalent rules (same scopes + match_type + normalized pattern) are collapsed
    to the single highest-severity rule. All contributing source rules and layer names
    are preserved for auditability.

    A later layer may only escalate severity. Downgrade attempts are ignored
    (the stronger rule wins and provenance is retained).
    """
    # Build map from equivalence key to best (highest severity) rule + sources
    best_by_key: Dict[tuple, dict] = {}

    for layer_idx, layer_rules in enumerate(layers):
        layer_name = layer_names[layer_idx] if layer_idx < len(layer_names) else f"layer_{layer_idx}"

        for rule in layer_rules:
            if not rule.enabled:
                continue
            key = _rule_equivalence_key(rule)
            sev = SEVERITY[rule.disposition]

            if key not in best_by_key:
                best_by_key[key] = {
                    "best_rule": rule,
                    "best_sev": sev,
                    "sources": [rule],
                    "layers": [layer_name]
                }
            else:
                entry = best_by_key[key]
                if sev > entry["best_sev"]:
                    # Escalate: replace effective rule, keep all sources
                    entry["best_rule"] = rule
                    entry["best_sev"] = sev
                    entry["sources"].append(rule)
                    entry["layers"].append(layer_name)
                else:
                    # Same or weaker: still record provenance (audit trail)
                    entry["sources"].append(rule)
                    entry["layers"].append(layer_name)

    # Convert to ComposedPolicyRule list
    result: List[ComposedPolicyRule] = []
    for entry in best_by_key.values():
        result.append(ComposedPolicyRule(
            effective_rule=entry["best_rule"],
            source_rules=tuple(entry["sources"]),
            source_layers=tuple(entry["layers"])
        ))
    return result
