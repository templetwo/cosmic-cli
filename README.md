# Cosmic CLI

A **coding-interface agent** for the terminal — not a chat wrapper with a shell escape hatch.

**Default model: `grok-4.5`** · **v0.4.2** · local only (no stack connector)

## What sets this apart

| Typical agent CLI | Cosmic Stargazer |
|-------------------|------------------|
| Monologue plan → shell spaghetti | **Tool-shaped actions** (search / edit / verify) |
| Guess paths | **GLOB / GREP / LIST** before READ |
| Dump whole files | **Surgical EDIT** with unique match |
| Hope it works | **READ-before-EDIT gate** + auto `py_compile` |
| Re-read forever | **File cache** + repeat → FINISH |
| Soft safety | **Safe-mode blast radius** |
| No trail | **Session transcript** + mission echoes |
| "Done!" | **FINISH receipts** + optional **review** seat |

The model is the reasoner. **The filesystem is ground truth.**

## Install

```bash
cd ~/cosmic-cli
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

Global launcher (if `~/bin` is on your PATH):

```bash
~/bin/cosmic-cli doctor    # COSMIC_CLI_HOME defaults to ~/cosmic-cli
```

## Config

Loaded in order (**later wins**): package `.env` → `~/.cosmic-cli/.env` → cwd `.env`

```
XAI_API_KEY=xai-…
COSMIC_GROK_MODEL=grok-4.5
OLLAMA_BASE_URL=http://localhost:11434
```

## Daily path

```bash
cosmic-cli doctor
cosmic-cli sessions

# Work
cosmic-cli do 'Find X and summarize how it works.'
cosmic-cli do 'Change Y in file Z (valid code).'
cosmic-cli do --review '…'          # build + independent review on diffs

# Review only
cosmic-cli review
cosmic-cli review --path note.py --directive 'change greeting'

# Other
cosmic-cli ask '…'
cosmic-cli chat
cosmic-cli stargazer ollama --model gemma4:e2b '…'
```

## Actions

```
GLOB / GREP / LIST / READ / EDIT / WRITE / DIFF
SHELL / CODE / TEST / TODO / PASS / FINISH
```

`EDIT: path|||old|||new` — old must match once; quotes preserved; Python syntax gated.

## Artifacts

| What | Where |
|------|--------|
| Mission echoes | `~/.cosmic_echo.jsonl` |
| Sessions | `~/.cosmic-cli/sessions/<id>.jsonl` |
| Reviews | `~/.cosmic-cli/sessions/<id>.review.json` |
| Edit backups | `*.cosmicbak` next to files |

## Tests

```bash
pytest tests/ -q
```

MIT · Temple of Two
