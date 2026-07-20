#!/usr/bin/env bash
# verify-linux-floor.sh
#
# Linux kernel-floor verification for the cosmic-cli / grok-build temple floor.
# The reference cosmic-cli floor canary runs a live Seatbelt probe on macOS but
# only a config check on Linux (temple_floor.py skips the live probe off-Darwin).
# This script closes that gap: it runtime-verifies the Linux floor on the box it
# runs on. grok-build's Linux floor is Landlock + a bubblewrap bind-over, so the
# authoritative test is grok-build's own deny-paths e2e (Landlock holds even
# against root); a self-contained bubblewrap check is the fallback.
#
# Usage:  bash scripts/verify-linux-floor.sh
#         GROK_BUILD_SRC=/path/to/grok-build bash scripts/verify-linux-floor.sh
# Then paste the entire output back.

set -uo pipefail

echo "===== ENVIRONMENT ====="
uname -a
echo "kernel: $(uname -r)  uid: $(id -u) ($([ "$(id -u)" = 0 ] && echo root || echo non-root))"
echo "Landlock: $([ -d /sys/kernel/security/landlock ] && echo present || echo 'ABSENT (kernel <5.13 or LSM off)')"
[ -r /sys/kernel/security/landlock/attr/abi ] && echo "Landlock ABI: $(cat /sys/kernel/security/landlock/attr/abi)"
echo "bwrap: $(command -v bwrap || echo MISSING)   cargo: $(command -v cargo || echo MISSING)"

echo
echo "===== TEST A: grok-build deny_paths_e2e (authoritative; Landlock, root-proof) ====="
GB="${GROK_BUILD_SRC:-}"
for c in "$GB" "$HOME/grok-build" "$HOME/code/grok-build" /tmp/grok-build-src /opt/grok-build; do
  [ -n "$c" ] && { [ -f "$c/Cargo.toml" ] || [ -d "$c/crates" ]; } && GB="$c" && break
done
if [ -n "$GB" ] && command -v cargo >/dev/null; then
  echo "grok-build source: $GB"
  ( cd "$GB" && cargo test -p xai-grok-sandbox deny_paths 2>&1 | tail -30 )
  echo "== look for 'test result: ok' on the deny_paths tests =="
else
  echo "SKIP: grok-build source or cargo not found."
  echo "      Set GROK_BUILD_SRC=/path/to/grok-build (with a Rust toolchain) and re-run"
  echo "      for the authoritative Landlock test."
fi

echo
echo "===== TEST B: self-contained bubblewrap bind-over ====="
if command -v bwrap >/dev/null; then
  D=$(mktemp -d); mkdir -p "$D/.cosmic-cli"; M="FLOOR-PROBE-LEAK-$RANDOM$RANDOM"
  printf '{"token":"%s"}\n' "$M" > "$D/.cosmic-cli/last_pause_token.json"
  : > "$D/placeholder"; chmod 000 "$D/placeholder"
  OUT=$(bwrap --ro-bind / / --proc /proc --dev /dev \
        --ro-bind "$D/placeholder" "$D/.cosmic-cli/last_pause_token.json" \
        -- cat "$D/.cosmic-cli/last_pause_token.json" 2>&1) || true
  if printf '%s' "$OUT" | grep -q "$M"; then
    echo "TEST B: FAIL - token LEAKED under bind-over: $OUT"
  else
    echo "TEST B: PASS - bind-over denies the token read (out='${OUT:0:70}')"
    [ "$(id -u)" = 0 ] && echo "  NOTE: as root, mode-000 is DAC-bypassable - Test A (Landlock) is the root-proof layer."
  fi
  rm -rf "$D"
else
  echo "SKIP: bubblewrap not installed (apt-get install -y bubblewrap) for the self-contained check."
fi

echo
echo "===== DONE - paste this entire output back ====="
