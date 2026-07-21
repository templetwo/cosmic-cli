"""Runtime build identity: which cosmic-cli answered, and what disagrees.

test_build_identity.py proves the version claims agree at TEST time. This covers
the RUN-time report. Two properties matter and both are asserted here:

  1. collect_identity() never raises. A doctor that crashes while reporting
     identity has destroyed the only evidence available about a broken install.
  2. A signal means genuine drift IN THIS INSTALL. Two metadata sources
     agreeing is the normal editable-install shape, so multiplicity alone must
     stay silent — a diagnostic that cries wolf on every healthy machine trains
     the operator to ignore it, which is worse than not having it. Stale
     cosmic-cli launchers elsewhere on the operator's PATH are `notes`, not
     signals, for the same reason: they cannot make a rollback wrong.

The signals are exercised against synthetic environments (a tmp PATH, a tmp
HOME), never by mutating the real install.
"""

import os
import stat
from pathlib import Path

import pytest

from cosmic_cli import __version__, buildinfo

EXPECTED_KEYS = {
    "process", "distributions", "path_hits", "git", "signals", "notes", "errors",
}

# A clean git result: most signal tests are not about provenance.
_GIT = {"source": "git", "describe": "v9.9.9", "commit": "abc1234",
        "detail": "", "unexpanded": False}


def _fake_launcher(directory: Path, shebang: str = "#!/usr/bin/env bash") -> Path:
    """An executable named cosmic-cli. Never run — only stat'd and read."""
    directory.mkdir(parents=True, exist_ok=True)
    launcher = directory / buildinfo.LAUNCHER_NAME
    launcher.write_text(f"{shebang}\nexit 0\n", encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return launcher


# ---- collect_identity contract ----------------------------------------------


def test_collect_identity_returns_the_documented_keys():
    ident = buildinfo.collect_identity()
    assert set(ident) == EXPECTED_KEYS
    assert isinstance(ident["signals"], list)
    assert isinstance(ident["distributions"], list)
    assert isinstance(ident["path_hits"], list)


def test_collect_identity_reports_this_process():
    process = buildinfo.collect_identity()["process"]
    assert process["version"] == __version__
    # The package dir must be the one this test imported, or the report is about
    # some other install than the one running.
    assert Path(process["package_dir"]) == Path(buildinfo.__file__).resolve().parent


def test_collect_identity_never_raises_when_every_source_is_hostile(monkeypatch, tmp_path):
    """Each section is gathered defensively and failures are reported, not raised."""
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("a file where PATH claims a directory\n")
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(not_a_dir), "/nonexistent", ""])
    )
    monkeypatch.setenv("HOME", str(tmp_path / "no-such-home"))

    def _boom(*_args):
        raise RuntimeError("metadata unreadable")

    monkeypatch.setattr(buildinfo, "_distributions", _boom)
    monkeypatch.setattr(buildinfo, "_git_describe", _boom)

    ident = buildinfo.collect_identity()
    assert set(ident) == EXPECTED_KEYS
    # A section that failed must SAY so; a short report reading as a clean one is
    # the exact failure this block exists to prevent.
    assert len(ident["errors"]) == 2
    assert all("RuntimeError" in e for e in ident["errors"])


# ---- distribution signals ----------------------------------------------------


def _process(version: str = __version__) -> dict:
    return {"version": version, "package_dir": "", "executable": "", "argv0": ""}


def test_two_agreeing_sources_are_not_a_signal():
    """The editable-install shape: in-repo egg-info + venv dist-info, agreeing."""
    dists = [
        {"path": "/repo/cosmic_cli.egg-info", "version": __version__},
        {"path": "/repo/venv/.../cosmic_cli-x.dist-info", "version": __version__},
    ]
    assert buildinfo._signals(_process(), dists, [], _GIT) == []


def test_disagreeing_sources_signal_and_name_both_paths():
    dists = [
        {"path": "/repo/cosmic_cli.egg-info", "version": __version__},
        {"path": "/usr/local/stale.dist-info", "version": "0.1.0"},
    ]
    signals = buildinfo._signals(_process(), dists, [], _GIT)
    assert signals, "a version disagreement must not be silent"
    joined = " ".join(signals)
    assert "/usr/local/stale.dist-info" in joined  # name the install to fix
    assert "0.1.0" in joined


def test_metadata_lagging_the_code_version_signals():
    """The editable-install trap: __version__ bumped, dist-info never refreshed."""
    dists = [{"path": "/repo/venv/cosmic_cli-old.dist-info", "version": "0.0.1"}]
    signals = buildinfo._signals(_process(), dists, [], _GIT)
    assert any("pip install -e" in s for s in signals)


def test_no_distributions_visible_is_not_a_signal():
    """A source-only clone has no metadata that could disagree with anything."""
    assert buildinfo._signals(_process(), [], [], _GIT) == []


# ---- PATH walk ---------------------------------------------------------------


def test_path_walk_finds_launchers_in_path_order_and_reads_the_shebang(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_launcher(tmp_path / "bin")  # the ~/bin shim
    _fake_launcher(tmp_path / "rogue", "#!/usr/bin/python3.13")
    monkeypatch.setenv("PATH", os.pathsep.join([
        str(tmp_path / "bin"), str(tmp_path / "empty"), str(tmp_path / "rogue"),
    ]))

    hits = buildinfo._path_hits()
    assert [Path(h["path"]).parent.name for h in hits] == ["bin", "rogue"]
    assert hits[0]["is_shim"] and hits[0]["expected"]
    assert hits[0]["kind"] == "launcher"  # env bash = a shim, not an unknown
    assert hits[1]["kind"] == "python"
    assert hits[1]["interpreter"] == "/usr/bin/python3.13"


def test_a_non_executable_file_named_cosmic_cli_is_not_a_hit(monkeypatch, tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / buildinfo.LAUNCHER_NAME).write_text("not executable\n")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert buildinfo._path_hits() == []


def test_a_stale_install_behind_the_shim_is_a_note_not_a_signal(monkeypatch, tmp_path):
    """It is still REPORTED — it just cannot fail --strict.

    A stale cosmic-cli further down PATH cannot change which code answered and
    says nothing about whether the checkout is at the tag it claims. Making it
    fatal turned `rollback.sh` red on a box whose rollback had succeeded.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_launcher(tmp_path / "bin")
    _fake_launcher(tmp_path / "rogue", "#!/usr/bin/python3.13")
    monkeypatch.setenv("PATH", os.pathsep.join([
        str(tmp_path / "bin"), str(tmp_path / "rogue")
    ]))
    hits = buildinfo._path_hits()
    assert buildinfo._signals(_process(), [], hits, _GIT) == []
    notes = buildinfo._path_notes(hits)
    assert len(notes) == 1
    assert "rogue" in notes[0], "an unrecognized install must still be named"


def test_a_rogue_install_ahead_of_the_shim_signals(monkeypatch, tmp_path):
    """PATH ORDER decides which install answers, so the WINNER is a signal."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_launcher(tmp_path / "bin")
    _fake_launcher(tmp_path / "rogue", "#!/usr/bin/python3.13")
    monkeypatch.setenv("PATH", os.pathsep.join([
        str(tmp_path / "rogue"), str(tmp_path / "bin")
    ]))
    hits = buildinfo._path_hits()
    signals = buildinfo._signals(_process(), [], hits, _GIT)
    assert len(signals) == 1
    assert "first cosmic-cli on PATH" in signals[0]
    assert "rogue" in signals[0]
    # Not double-reported: the winner is already fatal.
    assert buildinfo._path_notes(hits) == []


def test_only_the_shim_on_path_is_silent(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_launcher(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert buildinfo._signals(_process(), [], buildinfo._path_hits(), _GIT) == []


def test_this_installs_own_launcher_with_no_shim_is_silent(monkeypatch, tmp_path):
    """The Zenodo disaster-floor shape: a venv install and no ~/bin shim.

    `pip install <zenodo zip>` produces venv/bin/cosmic-cli and nothing in
    ~/bin. Nothing is wrong with that install, and it is the deployment the
    README points at when the remote is gone — so it must verify clean, or
    rollback.sh (whose verification step IS `doctor --strict`) can never
    complete on the box that needs it most.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))  # no bin/ at all
    monkeypatch.setattr(buildinfo, "_repo_root", lambda: tmp_path / "install")
    own_bin = _fake_launcher(tmp_path / "install" / "venv" / "bin").parent
    monkeypatch.setenv("PATH", os.pathsep.join([str(own_bin), "/usr/bin", "/bin"]))
    hits = buildinfo._path_hits()
    assert hits and hits[0]["expected"] and not hits[0]["is_shim"]
    assert buildinfo._signals(_process(), [], hits, _GIT) == []
    assert buildinfo._path_notes(hits) == []


# ---- the install that answered -----------------------------------------------


def test_a_foreign_tree_answering_for_cosmic_cli_home_signals(monkeypatch, tmp_path):
    """The COSMIC_CLI_HOME workflow's blind spot.

    The shim execs `$COSMIC_CLI_HOME/venv/bin/cosmic-cli`, so the operator has
    declared where the install lives. Code loading from anywhere else is a
    different checkout answering in its name, and the machine verdict must see
    it — the human-readable rows already did.
    """
    monkeypatch.setenv(buildinfo.HOME_VAR, str(tmp_path / "declared"))
    process = dict(_process(), package_dir=str(tmp_path / "elsewhere" / "cosmic_cli"))
    signals = buildinfo._signals(process, [], [], _GIT)
    assert len(signals) == 1
    assert buildinfo.HOME_VAR in signals[0]
    assert "elsewhere" in signals[0]


def test_a_non_editable_install_under_cosmic_cli_home_is_silent(monkeypatch, tmp_path):
    """Containment, not equality: site-packages under the declared root is fine."""
    monkeypatch.setenv(buildinfo.HOME_VAR, str(tmp_path))
    process = dict(
        _process(),
        package_dir=str(tmp_path / "venv" / "lib" / "python3.10" / "site-packages"
                        / "cosmic_cli"),
    )
    assert buildinfo._signals(process, [], [], _GIT) == []


def test_a_copied_working_tree_has_no_provenance_and_signals():
    """Unexpanded export-subst placeholders + no git = a COPY of a checkout.

    A clone reports git; a release artifact reports an expanded archival file.
    Only a copied tree carries the literal `$Format:` placeholders, and that
    code cannot be tied to any commit at all.
    """
    git = {"source": "none", "describe": "", "commit": "",
           "detail": "not a git checkout (.git_archival.txt placeholders unexpanded)",
           "unexpanded": True}
    signals = buildinfo._signals(_process(), [], [], git)
    assert len(signals) == 1
    assert "provenance" in signals[0]


def test_a_plain_non_git_install_is_not_a_provenance_signal():
    """A wheel/sdist install simply has no archival file. Silence, not noise."""
    git = {"source": "none", "describe": "", "commit": "",
           "detail": "not a git checkout", "unexpanded": False}
    assert buildinfo._signals(_process(), [], [], git) == []


# ---- unreadable distribution metadata ----------------------------------------


def test_an_unreadable_cosmic_cli_dist_is_listed_recorded_and_signalled(
    monkeypatch, tmp_path
):
    """A dist that cannot be read must not VANISH from the report.

    chmod 000 on a stale `cosmic_cli-*.dist-info` used to print `no identity
    drift` over a machine that had one — the exact shape (half-removed,
    corrupt, permission-broken) this scan exists to find.
    """
    broken = tmp_path / "cosmic_cli-0.2.0.dist-info"
    broken.mkdir()

    class _Unreadable:
        _path = broken

        @property
        def metadata(self):
            raise PermissionError(f"[Errno 13] Permission denied: {broken / 'METADATA'}")

    monkeypatch.setattr(
        buildinfo, "_installed_distributions", lambda: [_Unreadable()]
    )
    errors: list = []
    dists = buildinfo._distributions(errors)

    assert [d["path"] for d in dists] == [str(broken)]
    assert dists[0]["version"] == buildinfo.UNREADABLE
    assert errors and "PermissionError" in errors[0]
    signals = buildinfo._signals(_process(), dists, [], _GIT)
    assert len(signals) == 1
    assert "unreadable metadata" in signals[0]
    assert str(broken) in signals[0]


def test_an_unreadable_foreign_dist_is_recorded_but_not_ours(monkeypatch, tmp_path):
    """Someone else's broken package is noted, never claimed as a cosmic-cli."""
    broken = tmp_path / "requests-2.0.0.dist-info"
    broken.mkdir()

    class _Unreadable:
        _path = broken

        @property
        def metadata(self):
            raise PermissionError("denied")

    monkeypatch.setattr(
        buildinfo, "_installed_distributions", lambda: [_Unreadable()]
    )
    errors: list = []
    assert buildinfo._distributions(errors) == []
    assert errors, "a failure that was swallowed silently is the defect"


# ---- git ---------------------------------------------------------------------


def test_git_describe_reports_this_checkout():
    git = buildinfo._git_describe()
    assert git["source"] in {"git", "archive", "none"}
    if git["source"] == "git":
        assert git["commit"], "a git checkout must yield a commit id"


def test_git_falls_back_without_raising_when_git_is_unavailable(monkeypatch, tmp_path):
    """No git binary: the report degrades, it does not fail."""
    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir: no `git` to find
    monkeypatch.setattr(buildinfo, "_repo_root", lambda: tmp_path)
    git = buildinfo._git_describe()
    assert git["source"] == "none"
    assert git["detail"] == "not a git checkout"


def test_unexpanded_archival_placeholders_are_not_reported_as_a_commit(
    monkeypatch, tmp_path
):
    """export-subst only expands inside `git archive` output.

    In a working tree the placeholders are literal, and reporting `$Format:%H$`
    as a commit id would be worse than reporting nothing.
    """
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(buildinfo, "_repo_root", lambda: tmp_path)
    (tmp_path / ".git_archival.txt").write_text(
        "node: $Format:%H$\ndescribe-name: $Format:%(describe:tags=true)$\n"
    )
    git = buildinfo._git_describe()
    assert git["source"] == "none"
    assert "$Format:" not in git["describe"] + git["commit"]
    assert "unexpanded" in git["detail"]


def test_an_expanded_archival_file_is_read(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(buildinfo, "_repo_root", lambda: tmp_path)
    (tmp_path / ".git_archival.txt").write_text(
        "node: 0123456789abcdef0123456789abcdef01234567\ndescribe-name: v9.9.9\n"
    )
    git = buildinfo._git_describe()
    assert git["source"] == "archive"
    assert git["describe"] == "v9.9.9"
    assert git["commit"] == "0123456789ab"


# ---- version line format -----------------------------------------------------

# The rendered shape of the identity line, pinned by case matrix. grok pins its
# own the same way (test_display_version_formatting_matrix) because the string
# is the interface: an operator reads it off a screenshot and a script greps it,
# so a stray label or a moved paren is a breaking change with no compiler to
# catch it. format_version_line is pure precisely so this matrix can exist —
# every case here is asserted independently of the machine running the tests.
_VERSION_LINE_CASES = [
    # (version,       commit,                                     expected)
    ("0.9.4", "f1d7ac9", "cosmic-cli 0.9.4 (f1d7ac9)"),
    # Degraded form: no git, no archival file. The line still answers the
    # question it was asked, and does NOT print "unknown" to fill the space.
    ("0.9.4", "", "cosmic-cli 0.9.4"),
    ("0.9.4", None, "cosmic-cli 0.9.4"),
    # The archival fallback hands back 12 chars and describe hands back 7;
    # display is normalised so the line reads the same either way.
    ("0.9.4", "0123456789ab", "cosmic-cli 0.9.4 (0123456)"),
    ("0.9.4", "0123456789abcdef0123456789abcdef01234567", "cosmic-cli 0.9.4 (0123456)"),
    # A clean checkout sitting exactly on a tag yields a bare `v0.9.4` from
    # describe with no -g<sha>, so the commit slot gets the tag name. Printing
    # it would be a false provenance claim; it is dropped instead.
    ("0.9.4", "v0.9.4", "cosmic-cli 0.9.4"),
    # Unexpanded export-subst placeholder: dropped for the same reason.
    ("0.9.4", "$Format:%H$", "cosmic-cli 0.9.4"),
    # Too short to be an abbreviated object id anywhere git would emit one.
    ("0.9.4", "abc", "cosmic-cli 0.9.4"),
    # Prereleases and dev versions render unchanged — the version is whatever
    # __version__ says, never reformatted.
    ("1.0.0rc1", "abcdef1", "cosmic-cli 1.0.0rc1 (abcdef1)"),
    ("0.9.4.dev3+g1234567", "abcdef1", "cosmic-cli 0.9.4.dev3+g1234567 (abcdef1)"),
    # Surrounding whitespace off a file read must not reach the rendered line.
    ("0.9.4", "  f1d7ac9\n", "cosmic-cli 0.9.4 (f1d7ac9)"),
    ("0.9.4", "F1D7AC9", "cosmic-cli 0.9.4 (f1d7ac9)"),
]


@pytest.mark.parametrize("version,commit,expected", _VERSION_LINE_CASES)
def test_version_line_formatting_matrix(version, commit, expected):
    assert buildinfo.format_version_line(version, commit) == expected


def test_version_line_omits_the_commit_argument_entirely():
    """The commit is optional in the signature, not merely empty-able."""
    assert buildinfo.format_version_line("0.9.4") == "cosmic-cli 0.9.4"


def test_version_line_is_exactly_one_line():
    for version, commit, _ in _VERSION_LINE_CASES:
        rendered = buildinfo.format_version_line(version, commit)
        assert "\n" not in rendered and "\r" not in rendered


def test_version_line_reports_this_install():
    """The live resolver, which must agree with __version__ and never raise."""
    line = buildinfo.version_line()
    assert line.startswith(f"cosmic-cli {__version__}")
    assert line == buildinfo.format_version_line(
        __version__, buildinfo._provenance_commit()
    )


def test_version_line_degrades_when_provenance_is_unavailable(monkeypatch, tmp_path):
    """No git and no archival file: the version still answers, alone."""
    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir: no `git` to find
    monkeypatch.setattr(buildinfo, "_repo_root", lambda: tmp_path)
    assert buildinfo.version_line() == f"cosmic-cli {__version__}"


def test_version_line_survives_a_broken_git_resolver(monkeypatch):
    """Identity degrades, never raises — same contract as collect_identity."""
    def _boom():
        raise RuntimeError("git exploded")

    monkeypatch.setattr(buildinfo, "_git_describe", _boom)
    assert buildinfo.version_line() == f"cosmic-cli {__version__}"


# ---- provenance of the running code ------------------------------------------

# git walks UP, so a package installed into a venv nested inside a checkout gets
# the CHECKOUT's commit — a syntactically perfect hex sha naming the wrong build.
_INSTALLED_COPY_CASES = [
    # (package dir,                                              is a copy)
    ("/repo/venv/lib/python3.10/site-packages/cosmic_cli", True),
    ("/usr/lib/python3/dist-packages/cosmic_cli", True),
    ("/opt/homebrew/lib/python3.12/site-packages/cosmic_cli", True),
    # An editable install's package dir IS the checkout — commits are real here.
    ("/repo/cosmic_cli", False),
    ("/Users/someone/cosmic-cli/cosmic_cli", False),
    # Substrings of the marker names are not the marker names.
    ("/repo/my-site-packages-backup/cosmic_cli", False),
]


@pytest.mark.parametrize("package_dir,installed", _INSTALLED_COPY_CASES)
def test_installed_copy_detection_matrix(monkeypatch, package_dir, installed):
    monkeypatch.setattr(buildinfo, "_package_dir", lambda: Path(package_dir))
    assert buildinfo._is_installed_copy() is installed


def test_an_installed_copy_claims_no_commit_even_though_git_answers(monkeypatch):
    """The wheel-rollback state: git replies, and its reply is not this code.

    An offline wheel rollback deliberately leaves the checkout where it was — on
    the commit just rolled AWAY from — while the running bytes come from the
    wheel. git, asked about a site-packages dir inside that checkout, answers
    with the bad commit. The version line has no `package` row beside it to warn
    the operator, so it drops the claim rather than printing a false one.
    """
    monkeypatch.setattr(
        buildinfo, "_package_dir",
        lambda: Path("/repo/venv/lib/python3.10/site-packages/cosmic_cli"),
    )
    monkeypatch.setattr(buildinfo, "_git_describe", lambda: {
        "source": "git", "describe": "v1.0.0-1-g3248788", "commit": "3248788",
        "detail": "", "unexpanded": False,
    })
    assert buildinfo._provenance_commit() == ""
    assert buildinfo.version_line() == f"cosmic-cli {__version__}"


def test_a_checkout_still_reports_its_commit(monkeypatch):
    """The guard must not cost the editable install its provenance."""
    monkeypatch.setattr(buildinfo, "_package_dir", lambda: Path("/repo/cosmic_cli"))
    monkeypatch.setattr(buildinfo, "_git_describe", lambda: {
        "source": "git", "describe": "v0.9.4-8-gf1d7ac9", "commit": "f1d7ac9",
        "detail": "", "unexpanded": False,
    })
    assert buildinfo._provenance_commit() == "f1d7ac9"
    assert buildinfo.version_line() == f"cosmic-cli {__version__} (f1d7ac9)"


def test_installed_copy_detection_degrades_when_the_package_dir_is_unreadable(
    monkeypatch,
):
    """Same contract as everything else here: it answers, it does not raise."""
    def _boom():
        raise OSError("no such file")

    monkeypatch.setattr(buildinfo, "_package_dir", _boom)
    assert buildinfo._is_installed_copy() is False


# ---- the two version surfaces ------------------------------------------------


@pytest.mark.parametrize("args", [["--version"], ["-V"], ["version"]])
def test_every_version_surface_prints_the_same_single_line(args, monkeypatch):
    """`--version`, `-V` and `version` are one answer with three spellings.

    Pinned to the same BYTES, not merely to the same substring: two identity
    surfaces that drift apart are worse than one, because the operator has no
    way to know which of them is lying.
    """
    from click.testing import CliRunner

    from cosmic_cli import main as main_module

    monkeypatch.setattr(main_module.buildinfo, "version_line",
                        lambda: "cosmic-cli 9.9.9 (abc1234)")
    result = CliRunner().invoke(main_module.cli, args)
    assert result.exit_code == 0
    assert result.output == "cosmic-cli 9.9.9 (abc1234)\n"


def test_the_version_flag_never_starts_the_dashboard(monkeypatch):
    """Asking which build is running must not start a server.

    rollback.sh and any wrapper are free to call this on a box mid-repair; a
    side effect there is a side effect nobody asked for. -V answers eagerly,
    before the group body, and `version` is excluded from the dashboard set.
    """
    from click.testing import CliRunner

    from cosmic_cli import main as main_module

    started = []
    monkeypatch.setattr(main_module, "_ensure_dashboard",
                        lambda *a, **k: started.append(True))
    for args in (["--version"], ["-V"], ["version"]):
        assert CliRunner().invoke(main_module.cli, args).exit_code == 0
    assert not started, "the version surface spawned a dashboard"


# ---- doctor --strict ---------------------------------------------------------


@pytest.fixture
def quiet_doctor(monkeypatch):
    """Doctor without the network or the Helix probe.

    doctor calls the live models API; these tests are about the exit contract,
    not about xAI's uptime. Dropping the key short-circuits that whole branch.
    """
    from cosmic_cli import main as main_module

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setattr(main_module.helix_bridge, "available", lambda: False)
    return main_module


def _identity(signals: list, notes: list = None) -> dict:
    return {
        "process": _process(),
        "distributions": [],
        "path_hits": [],
        "git": {"source": "none", "describe": "", "commit": "", "detail": "test",
                "unexpanded": False},
        "signals": signals,
        "notes": notes or [],
        "errors": [],
    }


def _invoke(main_module, monkeypatch, signals, args, notes=None):
    from click.testing import CliRunner

    monkeypatch.setattr(
        main_module.buildinfo, "collect_identity", lambda: _identity(signals, notes)
    )
    return CliRunner().invoke(main_module.cli, args)


def test_doctor_renders_the_identity_block(quiet_doctor, monkeypatch):
    result = _invoke(quiet_doctor, monkeypatch, [], ["doctor"])
    assert result.exit_code == 0
    assert "IDENTITY" in result.output
    # IDENTITY is the top block: it must be readable before anything scrolls.
    assert result.output.index("IDENTITY") < result.output.index("Cosmic doctor")


def test_doctor_is_exit_0_on_a_signal_without_strict(quiet_doctor, monkeypatch):
    """Default behaviour is unchanged: doctor is a diagnostic, not a gate."""
    result = _invoke(quiet_doctor, monkeypatch, ["rogue install on PATH"], ["doctor"])
    assert result.exit_code == 0
    assert "rogue install on PATH" in result.output


def test_doctor_strict_is_exit_1_on_a_signal(quiet_doctor, monkeypatch):
    result = _invoke(
        quiet_doctor, monkeypatch, ["rogue install on PATH"], ["doctor", "--strict"]
    )
    assert result.exit_code == 1
    assert "rogue install on PATH" in result.output


def test_doctor_strict_is_exit_0_when_clean(quiet_doctor, monkeypatch):
    result = _invoke(quiet_doctor, monkeypatch, [], ["doctor", "--strict"])
    assert result.exit_code == 0
    assert "no drift in this install" in result.output


def test_doctor_strict_prints_notes_but_is_exit_0(quiet_doctor, monkeypatch):
    """Notes are the operator's box, not this install.

    rollback.sh verifies with `doctor --strict`. A stale cosmic-cli in
    /Library has no bearing on whether the checkout is at the tag, so it must
    be visible and must not turn a successful rollback red.
    """
    result = _invoke(
        quiet_doctor, monkeypatch, [], ["doctor", "--strict"],
        notes=["another cosmic-cli on PATH: /Library/stale/cosmic-cli"],
    )
    assert result.exit_code == 0
    assert "/Library/stale/cosmic-cli" in result.output
