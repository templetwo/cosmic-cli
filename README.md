# Cosmic CLI

**Temple runtime avionics** for coding cockpits (Grok Build, Claude Code, shell).

Not another agent TUI. Cosmic is the mission protocol a cockpit calls:
tool-shaped actions, T2Helix memory, compass-gated execution, a kernel-backed
governance floor, and an independent review seat.

**Default model:** `grok-4.5` · **v0.9.3** · substrate: [T2Helix](https://github.com/templetwo/t2helix)

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

## Mission Control dashboard

A read-only local dashboard over the shared chronicle.

```bash
cosmic-cli dashboard               # start + open http://localhost:4333
cosmic-cli dashboard --no-open     # start only
```

Binds `127.0.0.1`, serves `SELECT`-only chronicle views (no approval tokens), and
never runs on the `gate` hot path. Opt out entirely with `COSMIC_NO_DASHBOARD=1`.

## Horizon

Documented in repo-root `COSMIC.md`: Cosmic is avionics; cockpits fly.
One Stargazer path (Grok). Legacy material under `_archive/`.

## Tests

```bash
pytest tests/ -q                   # core suite
pytest tests/battery/ -q           # red-team battery + ASR table
```

## License

Apache-2.0 · [Temple of Two](https://thetempleoftwo.com) · Anthony Vasquez Sr.

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
