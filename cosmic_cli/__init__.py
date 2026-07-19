#!/usr/bin/env python3
"""Cosmic CLI — Temple runtime avionics (Grok Stargazer + T2Helix)."""

from .agents import StargazerAgent
from .checkpoint import CheckpointError, CheckpointManager, CheckpointManifest
from .context import ContextManager
from .gateway import ActionGateway, ApprovalManager, AuthorizationReceipt
from .policy import (
    ActionType,
    Disposition,
    MatchType,
    PolicyDecision,
    PolicyRule,
    PolicyValidationError,
    compose_policy_layers,
    evaluate_rules,
    validate_and_load_rules,
)
from .rules import load_rules_from_markdown
from .self_correction import BoundedSelfCorrection, CorrectionContext, CorrectionState

__version__ = "0.8.5"
__author__ = "Anthony Vasquez Sr. / Temple of Two"
__description__ = (
    "Temple runtime avionics: Grok Stargazer, T2Helix memory, "
    "policy kernel + action gateway"
)

__all__ = [
    "ActionGateway",
    "ActionType",
    "ApprovalManager",
    "AuthorizationReceipt",
    "BoundedSelfCorrection",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointManifest",
    "ContextManager",
    "CorrectionContext",
    "CorrectionState",
    "Disposition",
    "MatchType",
    "PolicyDecision",
    "PolicyRule",
    "PolicyValidationError",
    "StargazerAgent",
    "compose_policy_layers",
    "evaluate_rules",
    "load_rules_from_markdown",
    "validate_and_load_rules",
    "__version__",
    "__author__",
    "__description__",
]
