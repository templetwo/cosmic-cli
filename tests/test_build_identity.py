"""Build identity: one version, asserted across every place that claims one.

Releases are archived to Zenodo with a permanent DOI, so a version that
disagrees with itself is a citation defect, not a cosmetic one. cosmic_cli
__version__ is the single source of truth; pyproject.toml reads it, and the
citation metadata, the README's version claims, and every distribution
metadata directory visible on this machine must agree with it.
"""

import json
import re
import warnings
from importlib.metadata import distributions
from pathlib import Path

import pytest

from cosmic_cli import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]

# PEP 503 normalization: the same project ships as "cosmic-cli" in a dist-info
# and "cosmic_cli" in an egg-info, and both are this package.
DIST_NAME = "cosmic-cli"


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# Top-level `version:` in CITATION.cff, anchored to column 0 so the indented
# "Version DOI (v...)" description below it cannot match. Each quoting style
# gets its own branch because one shared `"?...."?` pattern returns the quotes
# as part of the value for single-quoted YAML, and does not match at all when a
# trailing comment follows — a present key reported as a missing key is the most
# misleading failure a guard can produce.
_CFF_VERSION_RE = re.compile(
    r"""^version:[ \t]*
        (?:
            "(?P<dq>[^"\n]*)"          # version: "0.9.4"
          | '(?P<sq>[^'\n]*)'          # version: '0.9.4'
          | (?P<bare>[^\s\#][^\n]*?)   # version: 0.9.4
        )
        (?:[ \t]+\#[^\n]*)?            # a YAML comment needs leading whitespace
        [ \t\r]*$
    """,
    re.MULTILINE | re.VERBOSE,
)

# README version claims, each anchored to the sentence carrying it. Both MUST
# match: if the surrounding prose is reworded, the test goes red rather than
# silently asserting nothing.
_README_CLAIMS = (
    (
        "header badge line",
        # **Default model:** `grok-4.5` · **v0.9.4** · substrate: ...
        re.compile(r"^\*\*Default model:\*\*.*?\*\*v([0-9][^*\s]*)\*\*", re.MULTILINE),
    ),
    (
        "human-readable citation block",
        # > (v0.9.4). The Temple of Two. Zenodo. https://doi.org/...
        re.compile(r"^>\s*\(v([0-9][^)\s]*)\)\.\s*The Temple of Two", re.MULTILINE),
    ),
)


def _source_path(dist) -> str:
    """Absolute path of a distribution's metadata directory.

    importlib.metadata exposes no public accessor for it, but a failure message
    without it is useless — the point is to name WHICH install to fix.
    PathDistribution._path is the metadata dir; locate_file("") is the public
    fallback that at least names the directory containing it.
    """
    path = getattr(dist, "_path", None) or dist.locate_file("")
    return str(Path(path).resolve())


def _pkg_info_version(pkg_info: Path) -> str:
    """First `Version:` header out of a PKG-INFO (RFC 822 headers)."""
    for line in pkg_info.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            break  # headers end at the first blank line; the description follows
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return "<no Version: header>"


def _version_sources() -> list[tuple[str, str]]:
    """Every metadata source visible here that claims to be cosmic-cli.

    WHY enumerate instead of importlib.metadata.version("cosmic-cli"): version()
    returns the FIRST match on sys.path and stops looking. pytest runs from the
    repo root, so the in-tree cosmic_cli.egg-info shadows the venv's dist-info —
    and any in-tree build regenerates that egg-info from the current
    __version__, so a single-source check degrades to `__version__ ==
    __version__` and disarms itself. Reading a plausible-but-wrong metadata
    source is the exact bug this guard exists to prevent. Comparing every source
    also catches the original failure: a stale 0.9.1 dist-info sitting beside a
    fresh 0.9.4 one.
    """
    sources: dict[str, str] = {}  # resolved metadata dir -> version it reports
    for dist in distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if not name or _normalized(name) != DIST_NAME:
            continue
        sources[_source_path(dist)] = dist.version

    # Also read the in-tree egg-info straight off disk: if the repo root is not
    # on sys.path, distributions() never sees it, and a stale build artifact
    # that nothing checks is the same wrong-metadata-source failure mode.
    pkg_info = REPO_ROOT / "cosmic_cli.egg-info" / "PKG-INFO"
    if pkg_info.is_file():
        sources.setdefault(str(pkg_info.parent.resolve()), _pkg_info_version(pkg_info))
    return sorted(sources.items())


def _citation_cff_version() -> str:
    """Read the top-level `version:` key out of CITATION.cff.

    Parsed by regex rather than PyYAML: the citation file is the only YAML the
    test suite touches, and it is not worth a test dependency.
    """
    text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = _CFF_VERSION_RE.search(text)
    assert match, "CITATION.cff has no top-level version key"
    value = match.group("dq")
    if value is None:
        value = match.group("sq")
    if value is None:
        value = match.group("bare")
    return value.strip()


def _zenodo_version() -> str:
    text = (REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8")
    return json.loads(text)["version"]


def test_citation_cff_matches_package_version():
    cff = _citation_cff_version()
    assert cff == __version__, (
        f"CITATION.cff says {cff}, cosmic_cli.__version__ says {__version__}"
    )


def test_zenodo_json_matches_package_version():
    zenodo = _zenodo_version()
    assert zenodo == __version__, (
        f".zenodo.json says {zenodo}, cosmic_cli.__version__ says {__version__}"
    )


def test_readme_version_claims_match_package_version():
    """The README states the version twice, one of them inside the citation
    block readers copy from — a stale number there is a citation defect with no
    compiler to catch it.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for label, pattern in _README_CLAIMS:
        match = pattern.search(text)
        assert match, (
            f"README.md {label} no longer states a version in the expected "
            f"shape; the guard cannot assert what it cannot find "
            f"(pattern: {pattern.pattern!r})"
        )
        assert match.group(1) == __version__, (
            f"README.md {label} says v{match.group(1)}, "
            f"cosmic_cli.__version__ says {__version__}"
        )


def test_readme_states_no_other_package_version():
    """Sweep for version claims added later that the anchored patterns miss.

    Deliberately narrow to v-prefixed three-component tokens: `COSMIC-ALLOW RFC
    v1.1` is a protocol version and `grok-4.5` a model name, and neither is a
    claim about this package's version.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    found = re.findall(r"\bv(\d+\.\d+\.\d+[^\s)*`]*)", text)
    assert found, "README.md no longer states a package version anywhere"
    stale = sorted({v for v in found if v != __version__})
    assert not stale, (
        f"README.md carries version claims {stale} but cosmic_cli.__version__ "
        f"is {__version__}"
    )


def test_all_distribution_metadata_matches_package_version():
    """The trap-closer: distribution metadata freezes at build/install time.

    Under an editable install, bumping __version__ does not refresh the
    dist-info, so the binary on PATH keeps reporting the old version. This goes
    red in exactly that state; the fix is `pip install -e .`, not a code change.
    Every visible source is checked rather than the first one importlib finds,
    so an in-tree build cannot regenerate the egg-info and then pass the guard
    on its own say-so.
    """
    sources = _version_sources()
    if not sources:
        # The only legitimate skip: a source-only clone that has never been
        # built or installed has no metadata that could disagree. If ANY source
        # exists we assert, because skipping then is a silent false green — exit
        # 0 with the guard never armed. Warn so the skip is visible in the run
        # output instead of hiding behind -q.
        warnings.warn(
            "BUILD IDENTITY GUARD NOT ASSERTED: no cosmic-cli distribution "
            "metadata found anywhere (no dist-info, no egg-info). Run "
            "`pip install -e .` to arm this guard before trusting a green run.",
            UserWarning,
            stacklevel=2,
        )
        pytest.skip(
            "no cosmic-cli distribution metadata anywhere; version guard UNARMED"
        )

    listing = "\n".join(f"  {version}  <-  {path}" for path, version in sources)
    stale = [(path, version) for path, version in sources if version != __version__]
    assert not stale, (
        f"distribution metadata disagrees with cosmic_cli.__version__ "
        f"({__version__}).\nDisagreeing sources:\n"
        + "\n".join(f"  {version}  <-  {path}" for path, version in stale)
        + f"\nAll visible sources:\n{listing}\n"
        "Reinstall the stale one (`pip install -e .`) or delete the stale build "
        "artifact. Do not change __version__ to make this pass."
    )
