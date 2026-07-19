#!/usr/bin/env bash
# cosmic-gate-wrapper.sh — Grok Build PreToolUse positive-protocol wrapper (box 2)
#
# RFC v1.1:
#   - Deny is the ground state
#   - Allow ONLY if entire stdout equals: COSMIC-ALLOW v1 <nonce>
#   - Nonce from env only (fresh per invocation, ≥128 bits)
#   - Gate stderr discarded from the decision (2>/dev/null)
#   - pipefail; missing/broken gate → deny
#   - stdin is opaque envelope bytes; never interpolated into shell code
#
# Install: cosmic-cli init --grok

set -euo pipefail

_deny() {
  local reason="${1:-cosmic gate: denied}"
  python3 -c 'import json,sys; print(json.dumps({"decision":"deny","reason":sys.argv[1]}))' "$reason"
  exit 2
}

_allow() {
  printf '%s\n' '{"decision":"allow"}'
  exit 0
}

if ! command -v cosmic-cli >/dev/null 2>&1; then
  if [[ -x "${HOME}/bin/cosmic-cli" ]]; then
    PATH="${HOME}/bin:${PATH}"
  elif [[ -n "${COSMIC_CLI:-}" && -x "${COSMIC_CLI}" ]]; then
    PATH="$(dirname "${COSMIC_CLI}"):${PATH}"
  fi
fi

if ! command -v cosmic-cli >/dev/null 2>&1; then
  _deny "cosmic-cli not found on PATH"
fi

if ! cosmic-cli gate --verb-check >/dev/null 2>&1; then
  _deny "cosmic-cli gate --verb-check failed (missing or broken)"
fi

if command -v openssl >/dev/null 2>&1; then
  NONCE="$(openssl rand -hex 16)"
else
  NONCE="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
fi
export COSMIC_GATE_NONCE="${NONCE}"

# Operator approval: env or one-shot file (for hook envs that drop exports)
if [[ -z "${COSMIC_APPROVAL_TOKEN:-}" && -f "${HOME}/.cosmic-cli/operator_approval_token" ]]; then
  export COSMIC_APPROVAL_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cosmic-cli/operator_approval_token")"
fi

# Opaque envelope bytes — never expand as shell
INPUT="$(cat)"

set +e
OUT="$(printf '%s' "${INPUT}" | COSMIC_GATE_NONCE="${NONCE}" env COSMIC_APPROVAL_TOKEN="${COSMIC_APPROVAL_TOKEN:-}" cosmic-cli gate --hook grok --mode safe 2>/dev/null)"
set -e

EXPECTED="COSMIC-ALLOW v1 ${NONCE}"
if [[ "${OUT}" == "${EXPECTED}" || "${OUT}" == "${EXPECTED}"$'\n' ]]; then
  _allow
fi

_deny "cosmic gate: no valid COSMIC-ALLOW sentinel"
