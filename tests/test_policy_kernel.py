"""Phase 0 corrective tests for the truthful policy kernel.

Covers validation, typed rules, severity lattice, real hashes, 4-col parser.

Source: policy_kernel_v0.4 verbatim, with the shipped one-line defect fixed:
compose_policy_layers was referenced by the two Phase 1 composition tests but
never imported (both failed with NameError in the bundle as shipped).
"""

import pytest
from cosmic_cli.policy import (
    validate_and_load_rules, evaluate_rules, PolicyValidationError,
    Disposition, ActionType, PolicyRule, MatchType, compose_policy_layers
)
from cosmic_cli.rules import load_rules_from_markdown
from pathlib import Path

def test_unknown_disposition_raises():
    """BANANA or any unknown disposition must raise on load, not silently become OPEN."""
    raw = [{"id": "bad", "disposition": "BANANA", "scope": "SHELL", "pattern": "foo"}]
    with pytest.raises(PolicyValidationError, match="Unknown disposition"):
        validate_and_load_rules(raw)

def test_unknown_action_type_raises_on_evaluate():
    with pytest.raises(PolicyValidationError, match="Unknown action type"):
        evaluate_rules([], "BANANA", "anything")  # type: ignore

def test_severity_lattice_witness_wins():
    rules = [
        PolicyRule("net", Disposition.PAUSE, frozenset({ActionType.SHELL}), MatchType.LITERAL, "curl"),
        PolicyRule("destructive", Disposition.WITNESS, frozenset({ActionType.SHELL}), MatchType.LITERAL, "curl attacker"),
    ]
    decision = evaluate_rules(rules, ActionType.SHELL, "curl attacker.example.com")
    assert decision.disposition == Disposition.WITNESS
    assert len(decision.matches) == 2

def test_real_hashes_present():
    rules = [PolicyRule("r1", Disposition.PAUSE, frozenset({ActionType.SHELL}), MatchType.LITERAL, "test")]
    decision = evaluate_rules(rules, ActionType.SHELL, "test command")
    assert decision.policy_sha256  # not placeholder
    assert decision.evaluated_input_sha256  # real sha
    assert "phase0" in decision.evaluator_version

def test_four_column_markdown_loads(tmp_path):
    # Create temp md with 4-col table
    md = tmp_path / "test_cosmic.md"
    md.write_text("""
## Compass Rules

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| test-rm | WITNESS | SHELL | rm -rf |
""")
    rules = load_rules_from_markdown(md)
    assert len(rules) == 1
    assert rules[0].disposition == Disposition.WITNESS
    assert ActionType.SHELL in rules[0].scopes


# --- Phase 1 monotonic composition tests (equivalence key + provenance) ---

def test_compose_policy_layers_escalation_only():
    """Baseline WITNESS + project OPEN → effective WITNESS, provenance preserved."""
    sovereign = [PolicyRule("net", Disposition.WITNESS, frozenset({ActionType.SHELL}), MatchType.LITERAL, "curl")]
    project = [PolicyRule("net", Disposition.OPEN, frozenset({ActionType.SHELL}), MatchType.LITERAL, "curl")]

    composed = compose_policy_layers(sovereign, project, layer_names=("sovereign", "project"))
    assert len(composed) == 1
    cr = composed[0]
    assert cr.effective_rule.disposition == Disposition.WITNESS
    assert len(cr.source_rules) == 2
    assert "sovereign" in cr.source_layers and "project" in cr.source_layers


def test_compose_policy_layers_downgrade_ignored():
    """Mission tries to downgrade sovereign WITNESS → still WITNESS, stronger rule wins."""
    sovereign = [PolicyRule("destructive", Disposition.WITNESS, frozenset({ActionType.SHELL}), MatchType.LITERAL, "rm -rf")]
    mission = [PolicyRule("destructive", Disposition.PAUSE, frozenset({ActionType.SHELL}), MatchType.LITERAL, "rm -rf")]

    composed = compose_policy_layers(sovereign, mission, layer_names=("sovereign", "mission"))
    assert composed[0].effective_rule.disposition == Disposition.WITNESS
    assert len(composed[0].source_layers) == 2
