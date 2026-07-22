# Cosmic CLI

**Temple runtime avionics** for coding cockpits (Grok Build, Claude Code, shell).

Not another agent TUI. Cosmic is the mission protocol a cockpit calls:
tool-shaped actions, T2Helix memory, compass-gated execution, a kernel-backed
governance floor, and an independent review seat.

**Default model:** `grok-4.5` · **v0.9.4** · substrate: [T2Helix](https://github.com/templetwo/t2helix)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21461197.svg)](https://doi.org/10.5281/zenodo.21461197) · Apache-2.0 · [COSMIC-ALLOW RFC v1.1](docs/COSMIC-ALLOW-RFC-v1.1.md)

## What it is

| Layer | Role |
|-------|------|
| **Stargazer** | Grok agent loop — READ / EDIT / CREATE / SHELL / CODE / FINISH |
| **T2Helix** | Local SQLite chronicle + compass (shared with Claude seats) |
| **Review** | Second-pass verdict on a mission (`do --review`) |
| **Cockpit** | You — Grok Build, Claude Code, or plain terminal |

Filesystem is ground truth. Gates exist so claims about the runtime stay true;
the product is finishing work.

## Governance model (three layers)

Enforcement is layered, and the order is deliberate. Higher layers are the
load-bearing ones; classification is the top, not the bottom.

| Layer | What it is | Boundary it holds |
|-------|-----------|-------------------|
| **1 · Kernel floor** | Grok Build's native sandbox (Seatbelt on macOS, Landlock + bwrap on Linux), provisioned by default and verified at launch | Denies reads of the approval-token store no matter how a command is spelled |
| **2 · Privilege ranking** | L3 operator > L2 seat > L1 compass > L0 gated model; approval flows only downward through a channel a lower level cannot reach | The model cannot approve its own actions |
| **3 · Classification** | The compass gate (WITNESS / PAUSE / OPEN) and command scanning | Defense in depth above the floor |

The self-approval class is closed **structurally**: approval is an interactive
operator (L2) action at a TTY, and there is no environment break-glass a lower
level could set.

## Install

```bash
cd ~/cosmic-cli
python3 -m venv venv && source venv/bin/activate
pip install -e ".[test]"
# optional: symlink ~/bin/cosmic-cli → this repo's launcher
```

Requires Python 3.10+ and an [xAI](https://x.ai) API key. `fcntl` is required
(POSIX; Windows is unsupported).

## Config

Load order (later wins): package `.env` → `~/.cosmic-cli/.env` → cwd `.env`

```
XAI_API_KEY=xai-…
COSMIC_GROK_MODEL=grok-4.5
T2HELIX_ROOT=~/t2helix
# optional: share Claude's Helix data dir
# T2HELIX_DATA_DIR=~/.claude/plugins/data/t2helix-templetwo-t2helix
# optional: never auto-start the dashboard
# COSMIC_NO_DASHBOARD=1
```

## T2Helix (memory + compass)

```bash
git clone https://github.com/templetwo/t2helix ~/t2helix
cd ~/t2helix && npm install && npm rebuild better-sqlite3
export T2HELIX_ROOT=~/t2helix

cosmic-cli helix status
cosmic-cli helix boot "what were we building"
cosmic-cli helix recall "path footgun"
```

> Note: `claude plugin update` does not rebuild the native `better-sqlite3`
> addon in the new install dir. After a plugin update, run
> `npm rebuild better-sqlite3` there or the chronicle will not load.

On each `do`, Cosmic boots Helix memory, sets a session goal, witnesses SHELL
and CODE through the compass, and records mission receipts.

**Compass contract**

| Class | Behavior |
|-------|----------|
| **WITNESS** | Hard deny — does not run |
| **PAUSE** | Soft block. A single-use token is minted to an **operator-only** 0600 file; it never reaches the model. The operator approves at a TTY, then the action retries once |
| **OPEN** | Allow |

Exit code **4** = blocked (cockpits can branch). Sensitive paths refuse READ
(including read-capable tools like Grep/Glob/ListDir aimed at the token store);
content is redacted before logs/LLM context.

**Approving a PAUSE** (operator, L2, interactive TTY only):

```bash
cosmic-cli helix show-pause-token   # display the last minted token (TTY only)
cosmic-cli helix accept-pause       # stage one approved retry, then re-run the action
```

A non-interactive shell — the model's own L0 shell included — is refused here by
design: it is not a TTY, so it cannot approve its own level.

## Which build am I running?

```bash
cosmic-cli --version        # cosmic-cli <version> (<short commit>)
cosmic-cli version          # same line, same bytes
cosmic-cli doctor           # the full IDENTITY block
```

The commit is resolved at **run time**, not baked in at build time. grok bakes
version and commit into its binary and has `build.rs` re-run whenever `.git/HEAD`
moves, so the embed cannot go stale; pip has no equivalent trigger, and a commit
baked into a Python package at install time would keep reporting the commit it
was installed from forever. So this asks instead.

When there is no commit to report — no checkout, no expanded archival file, or a
clean tree sitting exactly on a tag, where the version already names the commit —
the line degrades to `cosmic-cli <version>` and says nothing further. It never
prints `unknown`.

**An installed (non-editable) copy reports no commit either, on purpose.** git
walks *up* from the directory it is asked about, and a venv usually lives inside
the checkout, so a wheel installed into `venv/lib/.../site-packages` sits inside
that checkout's work tree and `git describe` answers for it — with the
*checkout's* commit, which has nothing to do with the bytes the installer copied
in. That is exactly the state an offline wheel rollback leaves behind, where the
checkout is still on the commit you just rolled *away* from. A hex sha looks
authoritative and there is no room on a one-line surface to qualify it, so the
line drops the claim rather than making a false one. `doctor` still shows the
raw `git` row, because it prints the `package` row right beside it and a path
under `site-packages` is the tell.

`doctor` prints the same version inside a fuller IDENTITY block: the package
directory, the interpreter, every distribution metadata dir claiming to be this
project, and every `cosmic-cli` on your PATH. That block is where you find out
that something *else* is answering to the name.

## Daily path

```bash
cosmic-cli doctor
cosmic-cli helix status
cosmic-cli init                         # COSMIC.md in this project
cosmic-cli do 'your task'               # Helix on by default
cosmic-cli do --no-helix 'offline'
cosmic-cli do --review 'careful change' # mission + independent review seat
cosmic-cli review
cosmic-cli sessions
cosmic-cli dashboard                     # Mission Control over the chronicle
```

Prefer **relative paths** when cwd is home:

```bash
cosmic-cli do 'CREATE Desktop/my-notes/readme.md explaining X'
```

## Actions

```
GLOB · GREP · LIST · READ · DIFF
MKDIR · CREATE · WRITE · EDIT
SHELL · CODE · TEST · TODO · PASS · FINISH
```

- `CREATE: path|||contents` — parents + file in one shot
- `EDIT: path|||old|||new` — unique match; Python syntax gated; `.cosmicbak`
- WRITE/CREATE observations return `rel=` and `abs=` (less find-loop thrash)
- `PASS: reason` — defer a blocked step and open a T2Helix thread for it

## Avionics stack

In-process policy on top of the Helix compass — one door for SHELL/CODE:

| Module | Role |
|--------|------|
| `policy` | Typed rules, severity lattice, monotonic layer compose |
| `rules` | Load `## Compass Rules` from project `COSMIC.md` |
| `gateway` | `authorize` → receipt → `execute_with_receipt`; single-use PAUSE token via an atomic flock'd claim (exactly-once, action-bound) |
| `checkpoint` | Content-preserving backup + exact rollback |
| `self_correction` | Bounded plan→review→execute→verify loop |

Gate order (SHELL/CODE): **local policy (validate)** → **check_shell** →
**sensitive-path / ranking refusal** → **Helix witness** → **consume PAUSE token**.

Mutations (EDIT/WRITE/CREATE/MKDIR): **gateway authorize + execute_with_receipt**
(checkpoint when available).

## Grok Build bridge

Mount the fail-closed gate under Grok Build's (or Claude Code's) fail-open
PreToolUse hooks. Deny is the ground state.

```bash
cosmic-cli init --grok --force     # installs the hook, wrapper, launcher,
                                   # and provisions the kernel floor by default
cosmic-cli gate --boot-canary      # per-class canary; refuses launch if the gate is dead
cosmic-cli gate --floor-canary     # proves the sandbox floor binds; refuses if it does not
~/.cosmic-cli/hooks/cosmic-launch-grok.sh
```

**Proof-of-allow (COSMIC-ALLOW RFC v1.1).** The gate authorizes an action only
by emitting, as its **entire** stdout, `COSMIC-ALLOW v1 <nonce>`, where the nonce
is a per-invocation value of at least 128 bits passed only through an environment
variable. stdin is treated as opaque bytes and never reflected; the environment
is never printed. Anything else is a deny.

**Kernel floor.** `init --grok` writes a `[profiles.temple]` deny-list for the
approval-token files into Grok Build's native `~/.grok/sandbox.toml` (the token
files only, never the whole `~/.cosmic-cli` tree) and sets it as the default
profile. The launcher runs `gate --floor-canary` and refuses to start the cockpit
if the deny rules do not bind. Runtime-verified on macOS Seatbelt; source- and
test-verified for Linux Landlock.

## Conformance

An invariant-organized red-team battery lives at `tests/battery/` and reports
Attack Success Rate (ASR) per class. It scores the **gate contract** in isolation
(it runs `cosmic-cli gate` as a subprocess and scores stdout), so it measures the
classification layer, not the kernel floor.

```bash
pytest tests/battery/ -q            # prints the ASR table
```

Gate-contract ASR is **3.3%** overall (down from 10.7% before the floor and
fixes), zero on every class except shell-indirection reads of the approval store
(`privilege_escalation`), which the kernel floor covers by design. The number is
reported honestly as two layers; the gate is not claimed to win the class the
floor is there to win.

## Rollback

**Known-good** = the latest release tag whose battery run was green. Green is not
0%: it means the archived ASR for that tag matches the recorded baseline (3.3%
overall, residual confined to `privilege_escalation`, which the kernel floor
owns). `battery-results.json` is local and gitignored, so the durable copy is the
`conformance-results.json` deposited *beside* the code in that version's Zenodo
record — a sibling file, not something inside the code archive and not something
`git show <tag>` can produce.

```bash
scripts/snapshot-release.sh                        # retain the tag HEAD is on
scripts/rollback.sh <tag>                          # restore, then verify
COSMIC_CLI_HOME=/path/to/checkout scripts/rollback.sh <tag>
```

**Retain the artifact; do not reconstruct it.** This is grok's model: it keeps
every downloaded binary in `~/.grok/downloads` and makes `~/.grok/bin/grok` a
symlink into that directory, so rolling back is an atomic repoint to something
already on the disk — upgrade and rollback are the same operation with a
different version. Reconstruction is not rollback. `git checkout <tag> && pip
install -e .` rebuilds the install from source and reaches the network for
dependencies, which is exactly what you do not have on the day you need it.

So `scripts/snapshot-release.sh` builds a wheel of the tag you are on and files
it under `$COSMIC_CLI_HOME/.cosmic-releases/`:

```
wheels/<tag>/cosmic_cli-<version>-py3-none-any.whl   the artifact
tags/<tag>                                           a pointer: its filename
```

**One directory per tag**, because a wheel's filename is not unique per tag.
Wheels are named by PEP 440 version, and two tags can legitimately ship the same
version — this repo's own history already contains such a pair. Filed flat, the
second snapshot silently overwrites the first, and rollback then installs the
wrong tag's code and reports success: you asked for one release and are running
another. Nothing downstream can catch that, either, because the metadata
`doctor --strict` compares against the code came out of the same wrong wheel, so
it agrees with itself. The tag is validated as a flat name before it is used as
a directory.

The pointer file handles the other half: tags are named by whatever the tagger
typed, so stripping a leading `v` happens to work right up until someone cuts a
release candidate, where the tag's `-rc1` becomes `rc1` in the wheel name and the
naive mapping quietly misses. The mapping is recorded once, at snapshot time,
when both names are in hand, and written last — so it doubles as the marker that
this tag is retained at all, and an interrupted snapshot never advertises a
partial artifact.

The wheel is built from **a pristine `git archive` export of the tag**, not from
your working tree. An in-tree build ships more than the tag: setuptools copies
sources into `build/lib` and never prunes files that vanished, so the second
snapshot taken in a checkout is the union of every tree ever built there — a
wheel filed under a tag can end up containing a module added two tags later.
`build/` is gitignored, so the dirty-worktree guard cannot see it, and verifying
that the zip is intact does not verify that it is *correct*. Exporting the tag
sidesteps the whole class, and means snapshot never writes inside your checkout
at all.

`rollback.sh` then **prefers the retained wheel** and installs it with
`--no-index` — no network, no build, and the artifact is the one that was
verified good rather than one rebuilt now and assumed to match. If no wheel is
retained for that tag it says so and falls back to `git checkout` + editable
reinstall, which needs the network. It always names the path it took.

### The two paths leave different states

This is the part to be honest about. An editable install means *the checkout is
the code*. A wheel install means *site-packages is the code* and the checkout is
just a directory that happens to be nearby.

- **Wheel rollback** replaces the installed package and **does not move your
  checkout**. Right for: getting flying again with no network, or when you have
  work in the checkout you do not want rewound. Cost: `git describe` still
  reports the *checkout's* tag, because that is the repo containing the package
  directory — so `doctor`'s `git` row can name a version the running code is
  not. The `package` row is what tells you which state you are in: a path under
  `venv/lib/.../site-packages` means the wheel is the code. `--version` detects
  that same path and omits the commit rather than printing the checkout's, which
  here would be the commit you just rolled away from.
- **Git rollback** moves the checkout and reinstalls it editable, so checkout,
  metadata and `git describe` all agree again. Right for: going back to a tag to
  *work* on it, or any time you want one coherent state. Cost: needs the network,
  rebuilds from source, and rewinds your tree.

The offline install is forced (`--force-reinstall --no-deps`), which is not
belt-and-braces. pip treats *the same version is already installed* as nothing to
do — it says so and exits 0 — so without forcing, a rollback between two tags
that share a version would install nothing and still report success. Forcing
makes it unconditional: the bytes on disk come from the named artifact.

`--no-deps` at both ends, deliberately: the retained artifact is a promise about
*this code*, not about the whole dependency closure, and forcing a reinstall of
that closure is something `--no-index` could not satisfy anyway. A backward
rollback lands in a venv that already satisfies the older pins. When it does not,
the failure surfaces at the **boot canary** — which imports the whole app rather
than asking pip's opinion of it — instead of at the install. Either way it fails
loudly rather than quietly reaching for a PyPI that may not be there, but it does
mean this is not a bare-metal disaster floor. That is what the Zenodo record
below is for.

`.cosmic-releases/` is gitignored, and `snapshot-release.sh` also writes the
ignore rule into `.git/info/exclude`. That second one is load-bearing rather than
tidy: `.gitignore` is *versioned*, so a git-path rollback to a tag cut before this
feature existed brings back a `.gitignore` without the entry, the retention dir
shows up as untracked, and `rollback.sh`'s dirty-worktree guard would then refuse
the next rollback over the artifact it was about to use. `.git/info/exclude` is
local to the clone and survives any checkout, which is the lifetime these wheels
actually have. `rollback.sh` re-writes it too, so an install that predates the
rule heals itself.

Both paths keep every guard, all of them before anything mutates: unknown tag
refused, dirty worktree refused (the wheel path does not move the tree, but the
tree is still where the identity report reads provenance from, so a dirty one
leaves two states with nothing to tell them apart), missing venv refused. Then
the result is proved — `gate --boot-canary` first, because that is the step that
shows the gate is live and discriminating and it is the one verb every tag
understands, then `doctor --strict`. Rollback only ever moves *backward*, so
verification cannot require a flag that only newer code has: against a tag that
predates `--strict` the script says so and runs plain `doctor` rather than
reporting a rollback that worked as a failure.

`doctor --strict` fails on drift in **this** install — metadata that disagrees
with the code, a foreign checkout answering for `$COSMIC_CLI_HOME`, code with no
provenance, or something else winning the PATH race. Another `cosmic-cli` sitting
further down your PATH is printed as a note and is *not* fatal: it cannot change
which code answered, and failing on it would make a successful rollback go red
over unrelated PATH hygiene.

**Disaster floor.** If the remote is gone or the tags are wrong, install from the
Zenodo record. The DOI snapshot cannot rot.

```bash
pip install https://zenodo.org/records/<record>/files/cosmic-cli-<tag>.zip
```

That install has a `venv/bin/cosmic-cli` and no `~/bin` shim, which is a normal
shape, not drift — `doctor --strict` verifies it clean.

**A botched upgrade fails closed.** Deny is the absence of the sentinel, so a
broken gate authorizes nothing, and the boot and floor canaries refuse cockpit
launch. Rollback is therefore a convenience procedure to get flying again, not an
incident procedure — nothing is open while you run it.

Mission-file rollback is a different unit and a different tool: `checkpoint.py`
restores the files one mission edited inside one workspace. This restores the
installed code version.

## Mission Control dashboard

A read-only local dashboard over the shared chronicle.

```bash
cosmic-cli dashboard               # start + open http://localhost:4333
cosmic-cli dashboard --no-open     # start only
```

Binds `127.0.0.1`, serves `SELECT`-only chronicle views (no approval tokens), and
logs to `~/.cosmic-cli-logs/` — never into the protected `~/.cosmic-cli/` token
store. Auto-start is an allowlist (`do`, `stargazer`, `workflow`, `tui`, `chat`),
so `gate` and every future subcommand are side-effect free unless they opt in.
Opt out entirely with `COSMIC_NO_DASHBOARD=1`.

## Horizon

Documented in repo-root `COSMIC.md`: Cosmic is avionics; cockpits fly.
One Stargazer path (Grok). Legacy material under `_archive/`.

## Tests

```bash
pytest tests/ -q                   # core suite
pytest tests/battery/ -q           # red-team battery + ASR table
```

## Cite

> Vasquez, A. J., Sr. (2026). *Fail-Closed Governance for Fail-Open Harnesses: A
> Cross-Vendor Proof-of-Allow Protocol for Coding Agents (COSMIC-ALLOW RFC v1.1)*
> (v0.9.4). The Temple of Two. Zenodo. https://doi.org/10.5281/zenodo.21461197

Concept DOI (all versions): `10.5281/zenodo.21461196` · see [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0 · [Temple of Two](https://thetempleoftwo.com) · Anthony Vasquez Sr.

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
