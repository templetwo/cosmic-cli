#!/usr/bin/env bash
# cosmic-launch-grok.sh — boot canary + temple floor canary, then exec grok
#
# Default install is fail-closed: GROK_SANDBOX=temple always for this launcher.
# Floor canary refuses launch if the temple deny-list is missing or unbound.

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

echo "cosmic-launch-grok: running temple floor canary…" >&2
if ! cosmic-cli gate --floor-canary; then
  echo "cosmic-launch-grok: FLOOR CANARY FAILED — refuse cockpit launch" >&2
  echo "cosmic-launch-grok: re-run: cosmic-cli init --grok --force" >&2
  exit 1
fi

if ! command -v grok >/dev/null 2>&1; then
  if [[ -x "${HOME}/.grok/bin/grok" ]]; then
    PATH="${HOME}/.grok/bin:${PATH}"
  fi
fi

if ! command -v grok >/dev/null 2>&1; then
  echo "cosmic-launch-grok: grok not on PATH after canaries — refuse launch" >&2
  exit 1
fi

export GROK_SANDBOX=temple
echo "cosmic-launch-grok: canaries PASS — exec grok (GROK_SANDBOX=temple)" >&2
exec grok "$@"
