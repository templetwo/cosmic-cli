"""Secret hygiene: path denylist + redaction before logs/LLM (audit H3).

Content redaction pattern table is intentionally aligned with
t2helix/lib/secrets.js (single vocabulary for compass detect + scrub).
A divergent Python subset was the Claude 2026-07-14 READ-probe finding:
path lock solid, redact() leaked AWS/GitHub/Stripe/Google/Slack/npm shapes
that the compass already caught.
"""

from __future__ import annotations

import hashlib
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

# (kind, group, pattern_src) — group 0 = whole match is the secret;
# group > 0 = only that capture is masked (label preserved).
# Ordered most-specific first (same order as t2helix/lib/secrets.js).
_PATTERN_SPECS: List[Tuple[str, int, str]] = [
    (
        "private-key",
        0,
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        r"[\s\S]*?(?:-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----|$)",
    ),
    (
        "url-auth",
        1,
        r"[a-z][a-z0-9+.-]*://[^/\s:@]+:([^/\s@]{3,})@",
    ),
    (
        "basic-auth",
        1,
        r"authorization\s*:\s*basic\s+([A-Za-z0-9+/=]{8,})",
    ),
    (
        "bearer",
        1,
        r"(?:authorization\s*:\s*)?bearer\s+([A-Za-z0-9._\-+/=]{8,})",
    ),
    # Stripe underscore keys before generic sk- hyphen keys.
    (
        "stripe-key",
        0,
        r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{10,}",
    ),
    (
        "sk-key",
        0,
        r"\bsk-[A-Za-z0-9_-]{16,}",
    ),
    (
        "xai-key",
        0,
        r"\bxai-[A-Za-z0-9]{20,}",
    ),
    (
        "github-token",
        0,
        r"\b(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b",
    ),
    (
        "google-key",
        0,
        r"\bAIza[0-9A-Za-z_-]{30,}",
    ),
    (
        "slack-token",
        0,
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
    ),
    (
        "npm-token",
        0,
        r"\bnpm_[A-Za-z0-9]{36}\b",
    ),
    (
        "sendgrid-key",
        0,
        r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}",
    ),
    (
        "aws-akid",
        0,
        r"\bAKIA[0-9A-Z]{16}\b",
    ),
    (
        "secret-assign",
        1,
        r"(?<![A-Za-z0-9])(?:password|passwd|secret|api[_-]?key|access[_-]?key|"
        r"auth[_-]?token|token)(?![A-Za-z])[A-Za-z0-9_]*[\"']?\s*[:=]\s*[\"']?"
        r"((?!\[REDACTED)[^\s\"',}]{8,})",
    ),
]

_REDACTORS: List[Tuple[str, int, Pattern[str]]] = [
    (kind, group, re.compile(src, re.IGNORECASE))
    for kind, group, src in _PATTERN_SPECS
]


def _fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8", errors="replace")).hexdigest()[:8]


def _mask_token(kind: str, secret: str) -> str:
    # Slash (not colon) after kind so secret-assign cannot re-match
    # `token:<fp>` / `secret-…:` inside an already-masked span
    # (t2helix secrets.js still uses colon and nests — cosmic does not).
    return f"[REDACTED:{kind}/{_fingerprint(secret)}]"


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
    """Mask known credential shapes. Aligned with t2helix secrets.js vocabulary."""
    if not text:
        return text
    out = text
    for kind, group, rx in _REDACTORS:

        def _repl(
            m: re.Match[str], *, _kind: str = kind, _group: int = group
        ) -> str:
            secret = m.group(0) if _group == 0 else m.group(_group)
            if secret is None or secret == "":
                return m.group(0)
            token = _mask_token(_kind, secret)
            if _group == 0:
                return token
            full = m.group(0)
            idx = full.rfind(secret)
            if idx < 0:
                return token
            return full[:idx] + token + full[idx + len(secret) :]

        out = rx.sub(_repl, out)
    return out


def redact_lines(lines: Iterable[str]) -> List[str]:
    return [redact(ln) for ln in lines]
