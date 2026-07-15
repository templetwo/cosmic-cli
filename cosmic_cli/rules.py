"""Markdown loader for human-readable policy view (prototype v0.2).

Note: Canonical policy should be JSON/YAML outside the editable root.
This loader is a convenience projection only. It now supports the documented
4-column format: ID | Type | Scope | Pattern
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from cosmic_cli.policy import (
    PolicyRule, validate_and_load_rules, PolicyValidationError,
    Disposition, ActionType, MatchType
)

# Updated schema for 4 columns
RULE_SCHEMA_4COL = r"^\|\s*([^\|]+?)\s*\|\s*(WITNESS|PAUSE|OPEN)\s*\|\s*([^\|]+?)\s*\|\s*([^\|]+?)\s*\|$"

def load_rules_from_markdown(cosmic_md: Path) -> List[PolicyRule]:
    """Parse 4-column Markdown table into validated PolicyRule list."""
    if not cosmic_md.exists():
        return []
    text = cosmic_md.read_text(encoding="utf-8")
    raw_rules = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Compass Rules"):
            in_section = True
            continue
        if in_section and stripped.startswith("##"):
            break
        if in_section and "|" in stripped and not stripped.startswith("| ID"):
            m = re.match(RULE_SCHEMA_4COL, stripped)
            if m:
                raw_rules.append({
                    "id": m.group(1).strip(),
                    "disposition": m.group(2).strip().upper(),
                    "scope": m.group(3).strip(),
                    "pattern": m.group(4).strip(),
                    "source": stripped,
                    "match_type": "literal"  # default for prototype
                })
    return validate_and_load_rules(raw_rules)
