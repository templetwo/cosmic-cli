# Cosmic CLI

A **coding-interface agent** for the terminal — not a chat wrapper with a shell escape hatch.

**Default model: `grok-4.5`** · **v0.5.0** · local only (no stack)

## What sets this apart

| Typical agent CLI | Cosmic Stargazer |
|-------------------|------------------|
| Shell mkdir + hope | **`CREATE`** one-shot (parents + file) |
| Absolute path thrash | **Relative-path bias** + normalize (no `lstrip` footgun) |
| Monologue plan | **Tool-shaped actions** + discovery thrash steer |
| Blind overwrite | **READ-before-EDIT** + syntax-safe EDIT |
| "Done!" vibes | **FINISH receipts** + optional **review** seat |
| No trail | **Sessions**, echoes, `COSMIC.md` via `init` |

Filesystem is ground truth. The loop + tools + gates are the product.

## Install

```bash
cd ~/cosmic-cli && python3 -m venv venv && source venv/bin/activate
pip install -e .
# optional: ~/bin/cosmic-cli launcher
```

## Config

Later wins: package `.env` → `~/.cosmic-cli/.env` → cwd `.env`

```
XAI_API_KEY=xai-…
COSMIC_GROK_MODEL=grok-4.5
OLLAMA_BASE_URL=http://localhost:11434
```

## Daily path

```bash
cosmic-cli doctor
cosmic-cli init                      # COSMIC.md in this project
cosmic-cli do 'your task'
cosmic-cli do --review 'careful change'
cosmic-cli review
cosmic-cli sessions
```

**Prefer relative paths** when cwd is home:

```bash
cosmic-cli do 'CREATE Desktop/my-notes/readme.md explaining X'
```

## Actions

```
GLOB · GREP · LIST · READ · DIFF
MKDIR · CREATE · WRITE · EDIT
SHELL · CODE · TEST · TODO · PASS · FINISH
```

`CREATE: path|||contents` — mkdir parents + write (best for folder+file).  
`EDIT: path|||old|||new` — unique match; quotes preserved; Python syntax gated.  
WRITE/CREATE observations return **`rel=`** and **`abs=`** so the model stops find-looping.

## Artifacts

| What | Where |
|------|--------|
| Echoes | `~/.cosmic_echo.jsonl` |
| Sessions | `~/.cosmic-cli/sessions/<id>.jsonl` |
| Reviews | `~/.cosmic-cli/sessions/<id>.review.json` |
| Backups | `*.cosmicbak` |
| Project notes | `COSMIC.md` via `init` |

## Tests

```bash
pytest tests/ -q
```

MIT · Temple of Two
