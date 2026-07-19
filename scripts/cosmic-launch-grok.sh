#!/usr/bin/env bash
# cosmic-launch-grok.sh — pre-flight boot canary, then exec grok (box 3 + launch)
#
# If any deny canary emits a sentinel, the allow canary fails, or the gate is
# missing, the cockpit refuses to start.

set -euo pipefail

if ! command -v cosmic-cli >/dev/null 2>&1; then
  if [[ -x "${HOME}/bin/cosmic-cli" ]]; then
    PATH="${HOME}/bin:${PATH}"
  fi
fi

if ! command -v cosmic-cli >/dev/null 2>&1; then
  echo "cosmic-launch-grok: cosmic-cli not on PATH — refuse launch" >&2
  exit 1
fi

echo "cosmic-launch-grok: running boot canary…" >&2
if ! cosmic-cli gate --boot-canary; then
  echo "cosmic-launch-grok: BOOT CANARY FAILED — refuse cockpit launch" >&2
  exit 1
fi

if ! command -v grok >/dev/null 2>&1; then
  if [[ -x "${HOME}/.grok/bin/grok" ]]; then
    PATH="${HOME}/.grok/bin:${PATH}"
  fi
fi

if ! command -v grok >/dev/null 2>&1; then
  echo "cosmic-launch-grok: grok not on PATH after canary — refuse launch" >&2
  exit 1
fi

echo "cosmic-launch-grok: canary PASS — exec grok" >&2
exec grok "$@"
