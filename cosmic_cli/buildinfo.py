"""Which cosmic-cli is actually answering — gathered, never guessed.

`tests/test_build_identity.py` proves the version claims agree at test time.
This module answers the operator's question at RUN time: the binary on PATH,
the interpreter behind it, the metadata directories claiming to be this project,
and the commit the code came from. The failure it exists to catch is drift — a
stale rogue install shadowing the real one on PATH, or distribution metadata
frozen at a version the code no longer is.

Two public surfaces. version_line() is the one-line answer behind `cosmic-cli
--version` and `cosmic-cli version`; collect_identity() is the full report
behind `doctor`.

Everything here is diagnostic. collect_identity() never raises: a doctor that
crashes while reporting identity has destroyed the only evidence available.
Findings are recorded as data and rendered by the caller, so the decision to
warn or to exit non-zero lives at the CLI surface, not here.

Two lists, and the split is the whole verdict policy:

  `signals` — drift in the install that ANSWERED. Metadata that disagrees with
    the code, a foreign tree replying to $COSMIC_CLI_HOME, code with no
    provenance, or something else winning the PATH race. --strict fails here.
  `notes`   — other cosmic-cli launchers sitting elsewhere on the operator's
    PATH. Real, worth printing, and NOT a property of this install: a stale
    3.13 install in /Library says nothing about whether $COSMIC_CLI_HOME is at
    the tag it claims. Making those fatal turned rollback.sh red on a box whose
    rollback had in fact succeeded, which is how a verification step teaches
    the operator to ignore it.

NOT the gate. The gate's hot path may not import this module — it scans PATH,
reads files and shells out to git. See gate.run_gate for the zero-cost line.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cosmic_cli

# PEP 503: the same project ships as "cosmic-cli" in a dist-info and
# "cosmic_cli" in an egg-info. Both are this package.
DIST_NAME = "cosmic-cli"

LAUNCHER_NAME = "cosmic-cli"

# A metadata dir that claims to be us but whose version could not be read. Kept
# in the listing rather than dropped: see _distributions.
UNREADABLE = "unreadable"

# Env var the ~/bin shim uses to locate the install it execs.
HOME_VAR = "COSMIC_CLI_HOME"

_GIT_TIMEOUT = 5  # doctor is interactive; a hung git must not hang the report


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# --- this process -------------------------------------------------------------


def _package_dir() -> Path:
    return Path(cosmic_cli.__file__).resolve().parent


def _repo_root() -> Path:
    """Directory containing the package — the repo root under an editable install."""
    return _package_dir().parent


def _process_identity() -> Dict[str, str]:
    return {
        "version": getattr(cosmic_cli, "__version__", "unknown"),
        "package_dir": str(_package_dir()),
        "executable": sys.executable or "",
        "argv0": sys.argv[0] if sys.argv else "",
    }


# --- distributions ------------------------------------------------------------


def _source_path(dist) -> str:
    """Absolute path of a distribution's metadata directory.

    importlib.metadata exposes no public accessor, but a report that cannot name
    WHICH install to fix is not a report. _path is the metadata dir;
    locate_file("") is the public fallback naming its parent.
    """
    path = getattr(dist, "_path", None) or dist.locate_file("")
    return str(Path(path).resolve())


def _location(dist) -> str:
    """Best-effort path for a distribution, even a broken one."""
    try:
        return _source_path(dist)
    except Exception:
        return "<unknown location>"


def _dir_claims_us(path: str) -> bool:
    """Does this metadata DIRECTORY name us, without reading its contents?

    The last line of defence when METADATA itself is unreadable. pip never
    renames these dirs, so `cosmic_cli-0.2.0.dist-info` and `cosmic_cli.egg-info`
    are attributable on the filename alone.
    """
    stem = Path(path).name
    for suffix in (".dist-info", ".egg-info", ".egg-link"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return _normalized(stem.split("-")[0]) == DIST_NAME


def _installed_distributions():
    """Seam over importlib.metadata.distributions() so tests can inject a broken
    one. Imported lazily: the scan is doctor-only and must not cost import time."""
    from importlib.metadata import distributions

    return distributions()


def _distributions(errors: List[str]) -> List[Dict[str, str]]:
    """Every metadata source visible here that claims to be cosmic-cli.

    Enumerated rather than asked via importlib.metadata.version(), which returns
    the FIRST match on sys.path and stops looking — that is precisely how a
    stale install hides. Two agreeing sources is the NORMAL editable-install
    shape (in-repo egg-info + venv dist-info), so multiplicity alone is never a
    signal; disagreement is.

    A dist whose metadata will not read is RECORDED, never skipped. Silently
    dropping it printed a clean report over a machine carrying an unreadable
    cosmic_cli-*.dist-info — which is precisely the shape of a stale install
    (chmod'd, corrupt, half-removed), i.e. the one this scan exists to find.
    """
    found: Dict[str, str] = {}
    for dist in _installed_distributions():
        where = _location(dist)
        try:
            meta = dist.metadata
            name = (meta["Name"] if meta else "") or ""
            if _normalized(name) == DIST_NAME:
                found[where] = dist.version
                continue
            if name:
                continue  # somebody else's package, legitimately not our problem
            raise ValueError("metadata has no Name field")
        except Exception as e:
            errors.append(f"distribution metadata at {where}: {type(e).__name__}: {e}")
            if _dir_claims_us(where):
                # Unattributable by content, attributable by filename. Listing it
                # as UNREADABLE keeps it in front of the operator and lets
                # _signals treat "cannot be confirmed to agree" as drift.
                found.setdefault(where, UNREADABLE)
    return [{"path": p, "version": v} for p, v in sorted(found.items())]


# --- PATH ---------------------------------------------------------------------


def _first_line(path: Path) -> str:
    """First line of a file, for the shebang. Bytes in, never executed."""
    try:
        with open(path, "rb") as fh:
            return fh.readline(512).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _interpreter_of(shebang: str) -> tuple[str, str]:
    """(kind, program) for a shebang line.

    `#!/usr/bin/env bash` means the file is a LAUNCHER that execs something else
    (the ~/bin shim is exactly this) — not an unidentifiable interpreter. A
    python shebang is the real fingerprint: it names the environment whose
    site-packages answered.
    """
    if not shebang.startswith("#!"):
        return ("unknown", "")
    parts = shebang[2:].strip().split()
    if not parts:
        return ("unknown", "")
    program = parts[0]
    if Path(program).name == "env" and len(parts) > 1:
        program = parts[1]
    base = Path(program).name
    if "python" in base:
        return ("python", program)
    if base in {"bash", "sh", "zsh", "dash"}:
        return ("launcher", program)
    return ("unknown", program)


def _expected_launchers() -> List[str]:
    """Launcher paths that are legitimately allowed to answer.

    The ~/bin shim is what every doc tells the operator to call, and it execs
    $COSMIC_CLI_HOME/venv/bin/cosmic-cli, so both are the same install. The
    running interpreter's own script dir is accepted too: an install from the
    Zenodo zip into some other venv is a valid deployment, and flagging the
    binary that is literally executing right now would be noise, not signal.
    """
    candidates = [
        Path.home() / "bin" / LAUNCHER_NAME,
        _repo_root() / "venv" / "bin" / LAUNCHER_NAME,
    ]
    if sys.executable:
        candidates.append(Path(sys.executable).resolve().parent / LAUNCHER_NAME)
    out: List[str] = []
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except OSError:
            resolved = str(candidate)
        if resolved not in out:
            out.append(resolved)
    return out


def _shim_path() -> str:
    return str(Path.home() / "bin" / LAUNCHER_NAME)


def _path_hits() -> List[Dict[str, Any]]:
    """Every executable `cosmic-cli` on PATH, in PATH order.

    No subprocess: `which -a` would fork a shell to answer a question os.access
    answers directly, and doctor runs on a box whose PATH is the thing under
    suspicion. Each hit's shebang identifies the interpreter behind it without
    running it.
    """
    raw = os.environ.get("PATH", "")
    hits: List[Dict[str, Any]] = []
    seen: set[str] = set()
    expected = set(_expected_launchers())
    shim = _shim_path()
    for entry in raw.split(os.pathsep):
        if not entry:
            continue  # POSIX empty entry means cwd; not a deliberate install
        candidate = Path(entry) / LAUNCHER_NAME
        try:
            if not os.access(candidate, os.X_OK) or not candidate.is_file():
                continue
        except OSError:
            continue
        key = str(candidate)
        if key in seen:
            continue  # a dir listed twice on PATH is not two installs
        seen.add(key)
        try:
            resolved = str(candidate.resolve())
        except OSError:
            resolved = key
        shebang = _first_line(candidate)
        kind, program = _interpreter_of(shebang)
        hits.append({
            "path": key,
            "resolved": resolved,
            "shebang": shebang,
            "kind": kind,
            "interpreter": program,
            "is_shim": key == shim or resolved == shim,
            "expected": key in expected or resolved in expected,
        })
    return hits


# --- git ----------------------------------------------------------------------

_ARCHIVAL_UNEXPANDED = "$Format:"


def _git_describe() -> Dict[str, str]:
    """Version + short commit from the checkout, else the archive substitution.

    Subprocess is acceptable HERE and nowhere else: doctor is a human-invoked
    diagnostic, never the gate. `--always` degrades to a bare short sha in a
    tagless clone; `--dirty` is the part that matters, because a dirty tree
    means the running code is not any tagged version at all.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(_package_dir()), "describe", "--tags", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
        describe = proc.stdout.strip()
        if proc.returncode == 0 and describe:
            match = re.search(r"-g([0-9a-f]{7,})", describe)
            commit = match.group(1) if match else describe.split("-")[0]
            return {
                "source": "git", "describe": describe, "commit": commit,
                "detail": "", "unexpanded": False,
            }
    except Exception:
        pass  # no git, no checkout, or a hung git — the archival fallback follows

    archival = _repo_root() / ".git_archival.txt"
    try:
        text = archival.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    if text and _ARCHIVAL_UNEXPANDED not in text:
        fields = {}
        for line in text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        describe = fields.get("describe-name", "")
        node = fields.get("node", "")
        return {
            "source": "archive",
            "describe": describe or node[:12],
            "commit": node[:12],
            "detail": "from .git_archival.txt (git archive tarball)",
            "unexpanded": False,
        }

    detail = "not a git checkout"
    unexpanded = bool(text)
    if unexpanded:
        # export-subst only expands inside `git archive` output. In a working
        # tree the placeholders are literal, and reporting `$Format:%H$` as a
        # commit id would be worse than reporting nothing. The file being here
        # AND literal means one thing: this tree was COPIED out of a checkout
        # rather than cloned or released. _signals treats that as drift.
        detail = "not a git checkout (.git_archival.txt placeholders unexpanded)"
    return {
        "source": "none", "describe": "", "commit": "", "detail": detail,
        "unexpanded": unexpanded,
    }


# --- provenance of the RUNNING code -------------------------------------------

# Directory names that mean "an installer COPIED this package here". A wheel
# install lands in one of them; an editable install never does, because its
# package dir is still the checkout — which is why `pip install -e .` keeps
# yielding real commits.
_INSTALLED_INTO = frozenset({"site-packages", "dist-packages"})


def _is_installed_copy() -> bool:
    """Is the running package an installed COPY rather than a live checkout?

    Load-bearing because git walks UP from the directory it is given. A venv
    commonly lives inside the checkout, so a wheel installed into it puts the
    package at $ROOT/venv/lib/pythonX.Y/site-packages/cosmic_cli — still inside
    $ROOT's work tree. `git -C <that dir> describe` therefore answers happily,
    with the CHECKOUT's commit, which has no relationship to the bytes an
    installer copied in.

    That is exactly the state an offline wheel rollback produces: the wheel path
    deliberately does not move the checkout, so the checkout is still sitting on
    the commit the operator just rolled away from.
    """
    try:
        return bool(set(_package_dir().parts) & _INSTALLED_INTO)
    except Exception:
        return False  # identity degrades, never raises


def _provenance_commit() -> str:
    """The commit the RUNNING code came from, or "" when nothing can prove one.

    Anything that PRINTS a commit goes through here rather than reading
    _git_describe() directly: git will answer for an enclosing checkout it has
    no right to speak for, and a hex sha satisfies every syntactic check while
    naming the wrong build.

    doctor keeps showing the raw describe, because it prints the `package` row
    beside it and a path under site-packages is the tell. The one-line surface
    has no room for a tell, so it drops the claim instead of qualifying it —
    same rule as the bare tag name in format_version_line.
    """
    if _is_installed_copy():
        return ""
    return _git_describe().get("commit", "")


# --- version line -------------------------------------------------------------

# A git object id and nothing else. Anything that fails this is dropped from the
# line rather than shown: see format_version_line.
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)

# Display width for the commit. The two resolvers hand back different lengths (7
# from `git describe`, 12 from the archival node), so the line is normalised
# here and reads the same whichever one answered.
_SHORT_COMMIT = 7


def format_version_line(version: str, commit: str = "") -> str:
    """`cosmic-cli 0.9.4 (f1d7ac9)`, or `cosmic-cli 0.9.4` when there is no commit.

    Pure — no git, no environment, no I/O — so the rendered shape can be pinned
    by a case matrix instead of by whatever the machine running the tests
    happens to be. tests/test_buildinfo.py holds that matrix.

    A commit that is not a hex object id is DROPPED, not printed. The degraded
    line is still true; `cosmic-cli 0.9.4 (v0.9.4)` would be a false claim about
    provenance. Nothing is ever rendered as "unknown" either: an omitted field
    says "not available" without teaching the operator to skim past noise.
    """
    line = f"{DIST_NAME} {version}"
    commit = (commit or "").strip()
    if _COMMIT_RE.match(commit):
        line += f" ({commit[:_SHORT_COMMIT].lower()})"
    return line


def version_line() -> str:
    """The identity line for THIS install, resolved at run time.

    grok bakes version+commit into the binary and has build.rs re-run whenever
    .git/HEAD moves, so the embed cannot go stale. pip has no equivalent
    trigger: a commit baked into a Python package at install time silently keeps
    reporting the commit it was installed from. So this resolves instead of
    embedding, and pays one `git describe` for it — acceptable because nothing
    but a human ever asks.

    A clean checkout sitting exactly ON a tag yields a bare `v0.9.4` from
    describe with no `-g<sha>` suffix, so no commit is shown there. That is the
    one case where the version alone already names the commit; between tags,
    where the version is ambiguous, describe supplies it.

    An installed (non-editable) copy shows no commit either — see
    _provenance_commit. Printing one there would name whatever repository the
    site-packages directory happens to sit inside.
    """
    try:
        commit = _provenance_commit()
    except Exception:  # identity must degrade, never raise — same rule as below
        commit = ""
    return format_version_line(
        getattr(cosmic_cli, "__version__", "unknown"), commit
    )


# --- signals ------------------------------------------------------------------


def _home_mismatch(process: Dict[str, str]) -> Optional[str]:
    """$COSMIC_CLI_HOME naming one install while a different tree replies.

    The shim execs `$COSMIC_CLI_HOME/venv/bin/cosmic-cli`, so the operator has
    declared where the install lives. If the code that actually loaded is not
    under that root, a foreign checkout is answering in its name — the one case
    where the running launcher genuinely is NOT the install it claims to be.
    Containment, not equality: a non-editable install puts the package under
    $COSMIC_CLI_HOME/venv/lib/..., which is still the declared install.
    """
    declared = os.environ.get(HOME_VAR, "")
    package_dir = process.get("package_dir", "")
    if not declared or not package_dir:
        return None
    try:
        root = Path(declared).expanduser().resolve()
        package = Path(package_dir).resolve()
    except OSError:
        return None
    if root == package or root in package.parents:
        return None
    return (
        f"{HOME_VAR} is {root} but the code answering loaded from {package} "
        "— a different checkout is running under this install's name"
    )


def _signals(process: Dict[str, str], dists: List[Dict[str, str]],
             hits: List[Dict[str, Any]], git: Dict[str, str]) -> List[str]:
    """Drift in the install that ANSWERED. --strict fails on exactly these.

    The scope rule: everything here must be a property of this install, not of
    the operator's box. Silence is the normal state of a healthy one.
    """
    out: List[str] = []
    version = process["version"]

    versions = {d["version"] for d in dists if d["version"] != UNREADABLE}
    if len(versions) > 1:
        out.append(
            "distribution metadata disagrees with itself: "
            + ", ".join(f"{d['version']} <- {d['path']}" for d in dists)
        )
    for dist in dists:
        if dist["version"] == UNREADABLE:
            out.append(
                f"cosmic-cli distribution at {dist['path']} has unreadable "
                "metadata — a version that cannot be read cannot be confirmed "
                "to agree, and a half-removed install is exactly this shape"
            )
        elif dist["version"] != version:
            out.append(
                f"distribution {dist['path']} reports {dist['version']} but "
                f"cosmic_cli.__version__ is {version} (reinstall: pip install -e .)"
            )

    mismatch = _home_mismatch(process)
    if mismatch:
        out.append(mismatch)

    if git.get("unexpanded"):
        out.append(
            "the running code has no provenance: .git_archival.txt sits beside "
            "it with placeholders unexpanded and there is no git checkout — a "
            "COPIED working tree, not a clone and not a release artifact"
        )

    # PATH ORDER, and only the winner. Whatever answers FIRST is what a bare
    # `cosmic-cli` runs; that it is not this install is drift. Note the test is
    # `expected`, not `is_shim`: a perfectly good install with no ~/bin shim
    # (the Zenodo disaster-floor shape) answers from its own venv/bin and is
    # not drifted, and failing --strict there made rollback unable to verify
    # the very deployment the docs point at.
    if hits and not hits[0]["expected"]:
        out.append(
            f"first cosmic-cli on PATH is {hits[0]['path']}, which is not this "
            f"install ({_package_dir()}) — PATH order decides which one answers"
        )
    return out


def _path_notes(hits: List[Dict[str, Any]]) -> List[str]:
    """Other cosmic-cli launchers on the box. Printed, never fatal.

    A stale install further DOWN the operator's PATH is a property of the
    machine: it cannot change which code just answered, and it says nothing
    about whether the checkout is at the tag it claims. It stays visible here
    because the operator should still clean it up, but it does not fail
    --strict, and therefore does not fail a rollback that worked.
    """
    return [
        f"another cosmic-cli on PATH: {hit['path']} "
        f"(shebang: {hit['shebang'] or '<none>'})"
        for i, hit in enumerate(hits)
        if not hit["expected"] and i > 0  # i == 0 is already a signal
    ]


# --- public -------------------------------------------------------------------


def collect_identity() -> Dict[str, Any]:
    """Everything known about which build is running. Never raises.

    Keys: process, distributions, path_hits, git, signals, notes, errors.
    `signals` is empty on a healthy install and is what --strict fails on;
    `notes` holds environment observations that are worth printing but are not
    this install's fault. Each entry of either is a human-readable sentence
    naming the exact path to fix. `errors` records anything that failed to
    gather, so a partial report says so instead of reading as a clean one.
    Rendering and exit-code policy belong to the caller.
    """
    errors: List[str] = []

    def _safe(label: str, fn, fallback):
        try:
            return fn()
        except Exception as e:  # a partial report beats a crashed one
            errors.append(f"{label}: {type(e).__name__}: {e}")
            return fallback

    process = _safe("process", _process_identity, {
        "version": getattr(cosmic_cli, "__version__", "unknown"),
        "package_dir": "", "executable": "", "argv0": "",
    })
    dists = _safe("distributions", lambda: _distributions(errors), [])
    hits = _safe("path", _path_hits, [])
    git = _safe("git", _git_describe, {
        "source": "none", "describe": "", "commit": "", "detail": "unavailable",
        "unexpanded": False,
    })
    signals = _safe("signals", lambda: _signals(process, dists, hits, git), [])
    notes = _safe("notes", lambda: _path_notes(hits), [])

    return {
        "process": process,
        "distributions": dists,
        "path_hits": hits,
        "git": git,
        "signals": signals,
        "notes": notes,
        "errors": errors,
    }
