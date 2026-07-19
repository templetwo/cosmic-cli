# Cosmic CLI

**Temple runtime avionics** for coding cockpits (Grok Build, Claude Code, shell).

Not another agent TUI. Cosmic is the mission protocol a cockpit calls:
tool-shaped actions, T2Helix memory, compass-gated execution, and an
independent review seat.

**Default model:** `grok-4.5` · **v0.8.7** · substrate: [T2Helix](https://github.com/templetwo/t2helix)

## What it is

| Layer | Role |
|-------|------|
| **Stargazer** | Grok agent loop — READ / EDIT / CREATE / SHELL / CODE / FINISH |
| **T2Helix** | Local SQLite chronicle + compass (shared with Claude seats) |
| **Review** | Second-pass verdict on a mission (`do --review`) |
| **Cockpit** | You — Grok Build, Claude Code, or plain terminal |

Filesystem is ground truth. Gates exist so claims about the runtime stay true;
the product is finishing work.

## Install

```bash
cd ~/cosmic-cli
python3 -m venv venv && source venv/bin/activate
pip install -e ".[test]"
# optional: symlink ~/bin/cosmic-cli → this repo’s launcher
```

Requires Python 3.10+ and an [xAI](https://x.ai) API key.

## Config

Load order (later wins): package `.env` → `~/.cosmic-cli/.env` → cwd `.env`

```
XAI_API_KEY=xai-…
COSMIC_GROK_MODEL=grok-4.5
T2HELIX_ROOT=~/t2helix
# optional: share Claude’s Helix data dir
# T2HELIX_DATA_DIR=~/.claude/plugins/data/t2helix-templetwo-t2helix
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

On each `do`, Cosmic boots Helix memory, sets a session goal, witnesses
SHELL and CODE through the compass, and records mission receipts.

**Compass contract**

| Class | Behavior |
|-------|----------|
| **WITNESS** | Hard deny — does not run |
| **PAUSE** | Soft block + single-use token → `cosmic-cli helix confirm <token>` then retry |
| **OPEN** | Allow |

Exit code **4** = blocked (cockpits can branch). Sensitive paths refuse READ;
content is redacted before logs/LLM context.

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

## Artifacts

| What | Where |
|------|--------|
| Echoes | `~/.cosmic_echo.jsonl` |
| Sessions | `~/.cosmic-cli/sessions/<id>.jsonl` |
| Reviews | `~/.cosmic-cli/sessions/<id>.review.json` |
| Backups | `*.cosmicbak` |
| Project law | `COSMIC.md` via `init` |
| Helix DB | `$T2HELIX_DATA_DIR` (or default Claude plugin path) |

## Avionics stack (v0.8)

In-process policy on top of Helix compass — one door for SHELL/CODE:

| Module | Role |
|--------|------|
| `policy` | Typed rules, severity lattice, monotonic layer compose |
| `rules` | Load `## Compass Rules` from project `COSMIC.md` |
| `gateway` | `authorize` → receipt → `execute_with_receipt` |
| `checkpoint` | Content-preserving backup + exact rollback |
| `self_correction` | Bounded plan→review→execute→verify loop |

Gate order (SHELL/CODE): **local policy (validate)** → **check_shell** → **Helix witness** → **consume PAUSE token**.

Mutations (EDIT/WRITE/CREATE/MKDIR): **gateway authorize + execute_with_receipt** (checkpoint when available).

## Horizon

Documented in repo-root `COSMIC.md`: Cosmic is avionics; cockpits fly.
One Stargazer path (Grok). Legacy material under `_archive/`.

## Tests

```bash
pytest tests/ -q
```

## License

MIT · [Temple of Two](https://thetempleoftwo.com) · Anthony Vasquez Sr.
