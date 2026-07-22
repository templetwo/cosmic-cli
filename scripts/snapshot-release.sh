#!/usr/bin/env bash
# snapshot-release.sh — retain the current known-good build as a local wheel
#
# The principle is borrowed from grok, which keeps every downloaded binary in
# ~/.grok/downloads and makes ~/.grok/bin/grok a symlink into it: rolling back
# is an atomic repoint to an artifact that is ALREADY THERE. Reconstruction is
# not rollback. `git checkout <tag> && pip install -e .` rebuilds the install
# from source and reaches the network for dependencies, which is exactly what
# you do not have on the day you need it.
#
# So: while the install is healthy, put a wheel on the disk. rollback.sh prefers
# it and installs offline.
#
# A SEPARATE script rather than a flag on rollback.sh, on purpose. Snapshot is a
# creation verb: it needs the network once, it needs a build backend, and it is
# run at release time by someone who is not rolling anything back. Rollback is a
# restoration verb whose entire value is working when the network is gone.
# Folding the build into the restore path would make the offline path depend on
# a build, and would put a step that WRITES an artifact inside a script whose
# contract is "check everything, then mutate the install".
#
# Layout under $ROOT/.cosmic-releases (gitignored — see below):
#
#   wheels/v0.9.4/cosmic_cli-0.9.4-py3-none-any.whl   the artifact
#   tags/v0.9.4                                       a pointer: its filename
#
# ONE DIRECTORY PER TAG, because the wheel filename is not unique per tag. Wheels
# are named by PEP 440 VERSION, and two tags can ship the same version — this
# repo already contains a pair (v0.9.2-ranking-floor and v0.9.3-battery both say
# 0.9.1). Filed flat, the second snapshot overwrites the first with no warning,
# and rollback then installs the wrong tag's code and reports PASS: the operator
# asked for one release and is running another, with nothing downstream able to
# notice, because the version metadata it would check came from the wrong wheel
# too. The tag is validated as a flat name below, so it is safe as a directory.
#
# The pointer file stays for the other half of the problem: tags are named by
# whatever the tagger typed, so `v0.9.4` becomes `0.9.4` but `v0.9.4-rc1` becomes
# `0.9.4rc1`, and stripping a leading `v` gets that wrong. Recording the mapping
# at snapshot time, when both names are in hand, means rollback never has to
# reconstruct it. It is also written LAST, so it doubles as the "this tag is
# retained" marker.
#
# Usage:  scripts/snapshot-release.sh            # snapshot the tag HEAD is on
#         scripts/snapshot-release.sh <tag>      # same, stated explicitly
#         COSMIC_CLI_HOME=/path/to/checkout scripts/snapshot-release.sh <tag>

set -euo pipefail

ROOT="${COSMIC_CLI_HOME:-$HOME/cosmic-cli}"
RETAIN="${ROOT}/.cosmic-releases"
TAG="${1:-}"

step() {
  local label="$1"; shift
  echo "snapshot: ${label}…" >&2
  if ! "$@"; then
    echo "snapshot: ${label} FAILED — nothing was retained" >&2
    exit 1
  fi
}

if [[ ! -x "${ROOT}/venv/bin/pip" ]]; then
  echo "snapshot: no venv at ${ROOT}/venv — refuse (set COSMIC_CLI_HOME)" >&2
  exit 1
fi

GITDIR="$(git -C "${ROOT}" rev-parse --absolute-git-dir 2>/dev/null || true)"
if [[ -z "${GITDIR}" ]]; then
  echo "snapshot: ${ROOT} is not a git checkout — refuse (nothing names the tag)" >&2
  exit 1
fi

# The retention dir has to be invisible to git in EVERY checked-out state, not
# just this one. .gitignore is VERSIONED: roll back to a tag cut before this
# feature existed and the entry is simply not there, `.cosmic-releases/` shows
# up as untracked, and rollback.sh's dirty-tree guard then refuses the very next
# rollback — the retention would have disarmed the mechanism it exists to serve.
# .git/info/exclude is local to the clone and survives any checkout, which is
# exactly the lifetime these wheels have. The committed .gitignore entry stays
# too, for the reader and for a fresh clone that has never snapshotted.
EXCLUDE="${GITDIR}/info/exclude"
if ! grep -qxF '.cosmic-releases/' "${EXCLUDE}" 2>/dev/null; then
  mkdir -p "$(dirname "${EXCLUDE}")"
  printf '%s\n' '.cosmic-releases/' >> "${EXCLUDE}"
  echo "snapshot: ignored .cosmic-releases/ in ${EXCLUDE}" >&2
fi

# A wheel built from a dirty tree is not the tag it would be filed under, and a
# retention dir you cannot trust is worse than an empty one: it is the artifact
# a later rollback will install without asking again.
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
  echo "snapshot: WORKTREE DIRTY at ${ROOT} — refuse (the wheel would not be ${TAG:-HEAD})" >&2
  echo "snapshot: commit or stash first: git -C ${ROOT} status" >&2
  exit 1
fi

if [[ -z "${TAG}" ]]; then
  TAG="$(git -C "${ROOT}" describe --tags --exact-match HEAD 2>/dev/null || true)"
  if [[ -z "${TAG}" ]]; then
    echo "snapshot: HEAD is not at a tag and none was given — refuse" >&2
    echo "snapshot: tags: $(git -C "${ROOT}" tag --list 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
  fi
  echo "snapshot: HEAD is at ${TAG}" >&2
fi

# The pointer is a flat filename, so the tag has to be one. `release/1.0` would
# otherwise silently write into a subdirectory rollback never looks in.
if [[ ! "${TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
  echo "snapshot: tag '${TAG}' is not a flat name — refuse (pointer files are flat)" >&2
  exit 1
fi

if ! git -C "${ROOT}" rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "snapshot: NO SUCH TAG ${TAG} in ${ROOT} — refuse" >&2
  exit 1
fi

# Snapshot NEVER moves the tree. Checking out to build would make this a
# mutation of the install, which is the other script's job and the other
# script's guards. If HEAD is elsewhere, say so and stop.
HEAD_SHA="$(git -C "${ROOT}" rev-parse HEAD)"
TAG_SHA="$(git -C "${ROOT}" rev-parse "refs/tags/${TAG}^{commit}")"
if [[ "${HEAD_SHA}" != "${TAG_SHA}" ]]; then
  echo "snapshot: HEAD is ${HEAD_SHA:0:7} but ${TAG} is ${TAG_SHA:0:7} — refuse" >&2
  echo "snapshot: check the tag out first; snapshot never moves your tree" >&2
  exit 1
fi

# Build from a PRISTINE EXPORT of the tag, never from the working tree. An
# in-tree build ships more than the tag in two ways the guards above cannot see:
#
#   - setuptools' build_py copies sources into $ROOT/build/lib and never prunes
#     files that vanished, so the second snapshot taken in a checkout is the
#     UNION of every tree ever built there. Observed: a wheel filed under a tag
#     that predates a module contained that module, plus a file existing at no
#     tag at all. build/ is gitignored, so the dirty-tree check stays green over
#     it, and the archive-integrity check verifies the zip is intact, not right.
#   - anything gitignored inside the package dir gets packaged too.
#
# `git archive` hands over exactly the tag's tracked content, and the build then
# happens somewhere disposable — so this never writes inside $ROOT at all, which
# is the same promise as "snapshot never moves your tree".
#
# --no-deps: the retained artifact is a promise about THIS code, not about the
# dependency closure. Retaining the whole closure would be a much larger promise
# (and a much larger directory) than one user rolling backward needs — a backward
# rollback lands in a venv that already satisfies the older pins. See the README
# for the honest limit on that.
TMP="$(mktemp -d "${TMPDIR:-/tmp}/cosmic-snapshot.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT
SRC="${TMP}/src"
OUT="${TMP}/wheel"
mkdir -p "${SRC}" "${OUT}"

step "export ${TAG}" \
  git -C "${ROOT}" archive --format=tar -o "${TMP}/src.tar" "refs/tags/${TAG}"
step "unpack ${TAG}" tar -xf "${TMP}/src.tar" -C "${SRC}"

# Build into a temp dir so "which file did we just build" needs no stdout parsing.
step "build wheel for ${TAG}" \
  "${ROOT}/venv/bin/pip" wheel --no-deps --wheel-dir "${OUT}" "${SRC}" --quiet

shopt -s nullglob
built=( "${OUT}"/*.whl )
shopt -u nullglob
if [[ ${#built[@]} -ne 1 ]]; then
  echo "snapshot: expected exactly 1 wheel, got ${#built[@]} — refuse" >&2
  exit 1
fi
WHEEL="$(basename "${built[0]}")"

# The artifact is only worth retaining if it is intact. mktemp may land on a
# different filesystem, so the move below can be a copy, and a copy is a place a
# truncated file can come from. Cheap to check now; expensive to discover on the
# day the wheel is the only copy left.
step "verify ${WHEEL}" "${ROOT}/venv/bin/python" -c \
  'import sys, zipfile; sys.exit(1 if zipfile.ZipFile(sys.argv[1]).testzip() else 0)' \
  "${built[0]}"

# The tag's own directory, rebuilt from empty. Re-snapshotting a tag REPLACES
# its artifact rather than leaving a sibling wheel behind, so the directory
# handed to `pip --find-links` is never a directory of two candidate answers.
# Replacement is announced: an artifact disappearing silently is how a retention
# dir stops being the thing that was verified good.
DEST="${RETAIN}/wheels/${TAG}"
if [[ -e "${DEST}" ]]; then
  echo "snapshot: replacing the artifact already retained for ${TAG}" >&2
  rm -rf "${DEST}"
fi
mkdir -p "${DEST}" "${RETAIN}/tags"
mv -f "${built[0]}" "${DEST}/${WHEEL}"
# Pointer written LAST: until it exists, rollback does not believe the wheel is
# there, so a snapshot interrupted mid-build never advertises a partial artifact.
printf '%s\n' "${WHEEL}" > "${RETAIN}/tags/${TAG}"

echo "snapshot: retained ${TAG} -> ${WHEEL}" >&2
echo "snapshot: ${DEST}/${WHEEL}" >&2
echo "snapshot: rollback.sh ${TAG} will now install offline" >&2
