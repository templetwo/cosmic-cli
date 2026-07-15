"""Phase B adversarial matrix for the policy evaluator, ported to the hardened kernel.

Source: dynamic_rules_v0.1 tests/test_self_correction.py (all six cases),
plus v0.8's end-to-end markdown-parse-then-evaluate case, adapted to the
typed v0.4 kernel APIs. Assertion semantics are preserved or strengthened:

- rules are built through validate_and_load_rules (typed PolicyRule) instead
  of raw dicts; action types are ActionType enums instead of strings;
- RuleMatch carries .rule (PolicyRule), so per-match disposition checks go
  through m.rule.disposition;
- v0.1's test_unknown_disposition_graceful asserted unknown types degrade to
  OPEN (and the bundled evaluator actually failed it, leaking 'BANANA' as the
  decision). The hardened kernel is stricter: unknown dispositions are
  rejected at load with PolicyValidationError — fail-loud beats fail-open;
- v0.8's evaluate_rule returned bare None for no-match; the hardened kernel
  returns an explicit OPEN decision with default_used=True.

Filename kept for lineage; the historical name predates the actual
self-correction loop (covered by test_seven_gates.py at integration level).
"""

import pytest
from pathlib import Path

from cosmic_cli.policy import (
    ActionType,
    Disposition,
    PolicyValidationError,
    evaluate_rules,
    validate_and_load_rules,
)
from cosmic_cli.rules import load_rules_from_markdown


def test_severity_lattice_witness_wins():
    """C1: PAUSE rule before WITNESS — WITNESS must win (strongest)."""
    rules = validate_and_load_rules([
        {"id": "net", "type": "PAUSE", "scope": "SHELL", "pattern": "curl",
         "raw": "| net | PAUSE | SHELL | curl |"},
        {"id": "destructive", "type": "WITNESS", "scope": "SHELL", "pattern": "curl attacker",
         "raw": "| destructive | WITNESS | SHELL | curl attacker |"},
    ])
    decision = evaluate_rules(rules, ActionType.SHELL, "curl attacker.example.com")
    assert decision.disposition == Disposition.WITNESS
    assert len(decision.matches) == 2  # both matched
    assert any(m.rule.disposition == Disposition.WITNESS for m in decision.matches)


def test_scope_filtering():
    """H2: Rule with scope should only apply to matching action types."""
    rules = validate_and_load_rules([
        {"id": "shell-only", "type": "WITNESS", "pattern": "rm -rf", "scope": "SHELL", "raw": ""},
    ])
    # Should trigger on SHELL
    d1 = evaluate_rules(rules, ActionType.SHELL, "rm -rf /tmp")
    assert d1.disposition == Disposition.WITNESS
    # Should NOT trigger on CODE (scope mismatch)
    d2 = evaluate_rules(rules, ActionType.CODE, "rm -rf /tmp")
    assert d2.disposition == Disposition.OPEN
    assert d2.default_used is True


def test_all_matches_reported():
    """Multiple overlapping rules → all reported, strongest wins."""
    rules = validate_and_load_rules([
        {"id": "pause-all", "type": "PAUSE", "scope": "SHELL", "pattern": "curl", "raw": ""},
        {"id": "witness-specific", "type": "WITNESS", "scope": "SHELL", "pattern": "curl evil", "raw": ""},
    ])
    decision = evaluate_rules(rules, ActionType.SHELL, "curl evil.com")
    assert decision.disposition == Disposition.WITNESS
    assert len(decision.matches) == 2


def test_pure_evaluation_no_side_effects():
    """Parsing + evaluation must be pure (no chronicle spam, deterministic)."""
    rules = validate_and_load_rules([
        {"id": "test", "type": "PAUSE", "scope": "SHELL", "pattern": "test", "raw": ""},
    ])
    d1 = evaluate_rules(rules, ActionType.SHELL, "test command")
    d2 = evaluate_rules(rules, ActionType.SHELL, "test command")
    assert d1.disposition == d2.disposition
    assert d1 == d2  # dataclass equality


def test_unknown_disposition_rejected_at_load():
    """v0.1 asserted unknown types silently degrade to OPEN (and its evaluator
    failed even that, returning 'BANANA' as the live disposition). The hardened
    kernel is stricter: unknown dispositions never enter the rule set."""
    with pytest.raises(PolicyValidationError, match="Unknown disposition"):
        validate_and_load_rules([{"id": "weird", "type": "BANANA", "pattern": "foo", "raw": ""}])


def test_empty_rules_default_open():
    decision = evaluate_rules([], ActionType.SHELL, "anything")
    assert decision.disposition == Disposition.OPEN
    assert decision.default_used is True


def test_rule_parser_and_evaluate(tmp_path: Path):
    """v0.8's end-to-end case: markdown table → rules → evaluation.

    Adapted from the 3-column (Rule|Type|Pattern) format to the shipped
    4-column (ID|Type|Scope|Pattern) loader."""
    mock_md = tmp_path / "mock_COSMIC.md"
    mock_md.write_text("""
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| block-rm | WITNESS | SHELL | rm -rf |
| confirm-net | PAUSE | SHELL | curl |
""")
    rules = load_rules_from_markdown(mock_md)
    assert len(rules) == 2
    assert rules[0].disposition == Disposition.WITNESS
    assert rules[0].pattern == "rm -rf"

    # Test evaluate (v0.8 returned bare strings / None; hardened returns decisions)
    assert evaluate_rules(rules, ActionType.SHELL, "rm -rf /tmp").disposition == Disposition.WITNESS
    d = evaluate_rules(rules, ActionType.SHELL, "echo hello")
    assert d.disposition == Disposition.OPEN and d.default_used is True
    assert evaluate_rules(rules, ActionType.SHELL, "curl https://evil").disposition == Disposition.PAUSE
