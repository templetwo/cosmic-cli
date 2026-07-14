"""Secret hygiene: path denylist + redaction before logs/LLM (audit H3)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Pattern, Tuple

# Basename / suffix patterns that must not be READ into model context.
SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".netrc",
    "credentials",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "authorized_keys",
    "known_hosts",
    ".npmrc",
    ".pypirc",
}

SENSITIVE_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
)

SENSITIVE_PATH_PARTS = (
    ".ssh",
    ".gnupg",
    ".aws",
    "private_keys",
)

# Patterns to redact before writing echoes/sessions or packing into prompts.
_REDACT_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|authorization)\s*[=:]\s*\S+"), r"\1=***REDACTED***"),
    (re.compile(r"\bxai-[A-Za-z0-9]{20,}\b"), "xai-***REDACTED***"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "sk-***REDACTED***"),
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
     "-----BEGIN PRIVATE KEY-----***REDACTED***-----END PRIVATE KEY-----"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b"), "Bearer ***REDACTED***"),
]


def is_sensitive_path(path: str) -> bool:
    p = Path(path)
    name = p.name.lower()
    if name in SENSITIVE_NAMES or name.startswith(".env"):
        return True
    if any(name.endswith(sfx) for sfx in SENSITIVE_SUFFIXES):
        return True
    parts = {part.lower() for part in p.parts}
    if parts & {p.lower() for p in SENSITIVE_PATH_PARTS}:
        return True
    if name.startswith("id_") and "pub" not in name:
        return True
    if "credential" in name:
        return True
    return False


def deny_read_message(path: str) -> str:
    return (
        f"[BLOCKED] refusing to READ sensitive path: {path}. "
        "Secrets must not enter the model context or mission logs."
    )


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for rx, repl in _REDACT_PATTERNS:
        out = rx.sub(repl, out)
    return out


def redact_lines(lines: Iterable[str]) -> List[str]:
    return [redact(ln) for ln in lines]
