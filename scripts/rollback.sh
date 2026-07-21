#!/usr/bin/env bash
# rollback.sh — return the install to a known-good release tag
#
# A botched upgrade already fails CLOSED (gate deny is the ABSENCE of the
# sentinel; the boot and floor canaries refuse cockpit launch), so this is a
# convenience procedure for getting working again, not an incident procedure.
# `git checkout` is destructive to uncommitted work, so a dirty tree is refused
# rather than clobbered, and the tag is validated while the tree is still intact.
# Verification is version-tolerant on purpose: this script only ever moves
# BACKWARD, so it cannot require anything the older code does not have.
#
# TWO PATHS, and it always says which one it took:
#
#   RETAINED WHEEL (preferred) — scripts/snapshot-release.sh already put a wheel
#     for this tag in $ROOT/.cosmic-releases/wheels/<tag>/. Install it with
#     --no-index: no network, no build, and the artifact is the one that was
#     verified good rather than one reconstructed now and assumed to match.
#     Filed per TAG, because wheel filenames carry the version and two tags can
#     ship one version — see snapshot-release.sh. This is grok's
#     model — its rollback repoints a symlink at a binary already sitting in
#     ~/.grok/downloads — and the reason is that reconstruction is not rollback.
#     The checkout is NOT moved on this path.
#
#   GIT CHECKOUT (fallback) — no wheel retained for this tag, so rebuild from
#     source. Needs the network for dependencies. This is the old behaviour and
#     it stays, because retention that started yesterday cannot cover a tag from
#     last year.
#
# The two paths leave the machine in DIFFERENT states. See the README: after a
# wheel install the checkout is no longer the code that runs, and doctor's
# `package` row is what tells you so.
#
# Usage:  scripts/rollback.sh <tag>
#         COSMIC_CLI_HOME=/path/to/checkout scripts/rollback.sh <tag>

set -euo pipefail

ROOT="${COSMIC_CLI_HOME:-$HOME/cosmic-cli}"
RETAIN="${ROOT}/.cosmic-releases"
TAG="${1:-}"

step() {
  local label="$1"; shift
  echo "rollback: ${label}…" >&2
  if ! "$@"; then
    echo "rollback: ${label} FAILED — install may be inconsistent at ${TAG}" >&2
    exit 1
  fi
}

if [[ -z "${TAG}" ]]; then
  echo "rollback: usage: scripts/rollback.sh <tag>" >&2
  echo "rollback: tags: $(git -C "${ROOT}" tag --list 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi

if [[ ! -x "${ROOT}/venv/bin/pip" ]]; then
  echo "rollback: no venv at ${ROOT}/venv — refuse (set COSMIC_CLI_HOME)" >&2
  exit 1
fi

if ! git -C "${ROOT}" rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "rollback: NO SUCH TAG ${TAG} in ${ROOT} — refuse" >&2
  exit 1
fi

# Self-heal the local ignore rule before the dirty check reads it. A previous
# rollback may have checked out a tag whose versioned .gitignore predates the
# retention feature, which leaves .cosmic-releases/ showing as untracked and
# would make the guard below refuse over an artifact this script itself relies
# on. .git/info/exclude is not versioned, so it is the one place the rule
# survives a checkout. Idempotent, touches no tracked file, and mutates nothing
# about the install. Written by scripts/snapshot-release.sh too.
GITDIR="$(git -C "${ROOT}" rev-parse --absolute-git-dir 2>/dev/null || true)"
if [[ -d "${RETAIN}" && -n "${GITDIR}" ]]; then
  EXCLUDE="${GITDIR}/info/exclude"
  if ! grep -qxF '.cosmic-releases/' "${EXCLUDE}" 2>/dev/null; then
    mkdir -p "$(dirname "${EXCLUDE}")"
    printf '%s\n' '.cosmic-releases/' >> "${EXCLUDE}"
  fi
fi

# Kept for BOTH paths, not just the checkout one. The wheel path does not touch
# the tree, but the tree is still where the identity report reads provenance
# from, so a dirty tree after a wheel rollback leaves two different states with
# nothing to tell them apart.
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
  echo "rollback: WORKTREE DIRTY at ${ROOT} — refuse (checkout would clobber it)" >&2
  echo "rollback: commit or stash first: git -C ${ROOT} status" >&2
  exit 1
fi

# Resolve the retained artifact BEFORE anything mutates, like every other check
# here. A pointer whose wheel has been deleted is a stale pointer, not a reason
# to fail: say so and take the source path, which still works.
POINTER="${RETAIN}/tags/${TAG}"
# Artifacts are filed under the TAG, not under the wheel filename: wheels are
# named by PEP 440 version and two tags can ship the same version, so a flat
# directory would hand back whichever tag was snapshotted last while every
# downstream check (metadata vs __version__) agreed with itself and passed.
WHEELDIR="${RETAIN}/wheels/${TAG}"
WHEEL=""
if [[ -f "${POINTER}" ]]; then
  WHEEL="$(head -n 1 "${POINTER}" | tr -d '[:space:]')"
  # The pointer's contents become a path that gets INSTALLED, so it has to be a
  # flat wheel filename and nothing else — no directory parts, no traversal.
  if [[ ! "${WHEEL}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*\.whl$ ]]; then
    echo "rollback: retention pointer ${POINTER} is not a wheel filename ('${WHEEL}') — ignoring" >&2
    WHEEL=""
  elif [[ ! -f "${WHEELDIR}/${WHEEL}" ]]; then
    echo "rollback: retention pointer ${POINTER} names '${WHEEL}' but ${WHEELDIR}/${WHEEL} is missing — ignoring" >&2
    WHEEL=""
  fi
fi

if [[ -n "${WHEEL}" ]]; then
  echo "rollback: ${ROOT} -> ${TAG} via RETAINED WHEEL (offline; checkout NOT moved)" >&2
  echo "rollback: artifact ${WHEELDIR}/${WHEEL}" >&2
  # --no-index is the whole point: this must not reach the network, and it must
  # not quietly resolve against a PyPI that may not be reachable. --find-links is
  # this TAG's directory only, so nothing from another tag can answer.
  #
  # --force-reinstall is not belt-and-braces. pip treats "the same version is
  # already installed" as nothing to do — it says so and exits 0 — and two tags
  # CAN carry the same version, so rolling between them would be a silent no-op
  # that still printed PASS. Forcing makes the install unconditional: the bytes
  # on disk come from the named artifact, which is the entire promise of
  # retention. --no-deps keeps the force from also trying to rebuild the
  # dependency closure, which --no-index could not satisfy; the boot canary
  # below is what proves that closure is still intact, because it imports the
  # whole app rather than asking pip's opinion of it.
  step "offline install ${WHEEL}" "${ROOT}/venv/bin/pip" install \
    --no-index --find-links "${WHEELDIR}" --force-reinstall --no-deps \
    "${WHEELDIR}/${WHEEL}" --quiet
else
  echo "rollback: no retained wheel for ${TAG} at ${POINTER}" >&2
  echo "rollback: ${ROOT} -> ${TAG} via GIT CHECKOUT + rebuild (needs the network)" >&2
  echo "rollback: retain the next one first: scripts/snapshot-release.sh ${TAG}" >&2
  step "checkout ${TAG}" git -C "${ROOT}" checkout "${TAG}"
  step "reinstall" "${ROOT}/venv/bin/pip" install -e "${ROOT}" --quiet
fi

# The canary runs FIRST and is the load-bearing proof: it exists at every tag,
# and it is the step that shows the gate is live and discriminating. Ordering it
# behind a check that older code cannot answer meant the one piece of positive
# evidence never ran.
step "boot canary" "${ROOT}/venv/bin/cosmic-cli" gate --boot-canary

# Verification must not be pinned to a flag only NEWER code has: rollback only
# ever moves backward, so every tag predating a verification verb would report a
# rollback that mechanically succeeded as a failure. Probe, then degrade.
# Captured, not piped to grep: `grep -q` closes the pipe on first match, the
# writer takes EPIPE and exits non-zero, and pipefail then reads a SUPPORTED
# flag as unsupported. Racy on output size, so it fails silently and sometimes.
HELP="$("${ROOT}/venv/bin/cosmic-cli" doctor --help 2>/dev/null || true)"
if [[ "${HELP}" == *"--strict"* ]]; then
  step "doctor --strict" "${ROOT}/venv/bin/cosmic-cli" doctor --strict
else
  echo "rollback: ${TAG} predates 'doctor --strict' — running plain doctor" >&2
  step "doctor" "${ROOT}/venv/bin/cosmic-cli" doctor
fi
if [[ -n "${WHEEL}" ]]; then
  echo "rollback: PASS — ${TAG} installed from the retained wheel and the gate is live" >&2
  echo "rollback: the checkout at ${ROOT} was not moved; the installed code is the wheel" >&2
else
  echo "rollback: PASS — ${ROOT} is ${TAG} and the gate is live" >&2
fi
