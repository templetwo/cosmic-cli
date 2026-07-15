"""Markdown loader for human-readable policy view.

Parses ## Compass Rules 4-column tables. Malformed disposition cells fail loud
(never silently dropped to allow).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from cosmic_cli.policy import PolicyRule, PolicyValidationError, validate_and_load_rules

# Valid disposition column
RULE_SCHEMA_4COL = (
    r"^\|\s*([^\|]+?)\s*\|\s*(WITNESS|PAUSE|OPEN)\s*\|\s*([^\|]+?)\s*\|\s*([^\|]+?)\s*\|$"
)
# Row that looks like a rule but has a bad disposition / shape
RULE_ROW_LOOSE = r"^\|\s*([^\|]+?)\s*\|\s*([^\|]+?)\s*\|\s*([^\|]+?)\s*\|\s*([^\|]+?)\s*\|$"


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
        if not in_section or "|" not in stripped:
            continue
        if stripped.startswith("| ID") or re.match(r"^\|\s*[-:]+", stripped):
            continue
        m = re.match(RULE_SCHEMA_4COL, stripped)
        if m:
            raw_rules.append(
                {
                    "id": m.group(1).strip(),
                    "disposition": m.group(2).strip().upper(),
                    "scope": m.group(3).strip(),
                    "pattern": m.group(4).strip(),
                    "source": stripped,
                    "match_type": "literal",
                }
            )
            continue
        loose = re.match(RULE_ROW_LOOSE, stripped)
        if loose:
            bad = loose.group(2).strip()
            raise PolicyValidationError(
                f"Invalid disposition {bad!r} in Compass Rules row "
                f"(want WITNESS|PAUSE|OPEN): {stripped}"
            )
    return validate_and_load_rules(raw_rules)
