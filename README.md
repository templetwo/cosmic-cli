# Cosmic CLI

A **coding-interface agent** for the terminal — not a chat wrapper with a shell escape hatch.

**Default model: `grok-4.5`**

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
| "Done!" | **FINISH receipts** (what / evidence / residual) |

The model is the reasoner. **The filesystem is ground truth.** The loop + tools + gates are the product.

## Install

```bash
cd ~/cosmic-cli
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

## Config

`.env` (gitignored):

```
XAI_API_KEY=xai-…
COSMIC_GROK_MODEL=grok-4.5
```

## Daily use

```bash
cosmic-cli doctor

# Primary
cosmic-cli do 'Find where DEFAULT_MODEL is set and explain how to override it.'

# Make a change (agent will READ → EDIT → verify)
cosmic-cli do 'In scratch/note.txt, change hello to hello world. Create the file if needed.'

# Options
cosmic-cli do --mode safe --max-steps 20 --no-verify -q '…'
cosmic-cli do --model grok-4.3 '…'
cosmic-cli do --review '…'   # build, then independent review seat on diffs

# Review only (latest session or explicit paths)
cosmic-cli review
cosmic-cli review --session 20260713T235943Z
cosmic-cli review --path note.py --directive 'change greeting'

cosmic-cli ask '…'      # one-shot, no tools
cosmic-cli chat
cosmic-cli analyze path
cosmic-cli stargazer ollama --model gemma4:e2b '…'   # local fallback
```

## Action vocabulary

```
GLOB: **/*.py
GREP: pattern | path | glob=*.py
LIST: .
READ: path/to/file
EDIT: path|||exact_old|||new
WRITE: path|||full_contents
DIFF: path
SHELL: command
CODE: python
TEST: tests/ -q
TODO: ["step1","step2"]
PASS: blocker
FINISH: receipt
```

## Artifacts

- Mission echoes: `~/.cosmic_echo.jsonl`
- Session logs: `~/.cosmic-cli/sessions/<id>.jsonl`
- Edit backups: `*.cosmicbak` next to edited files

## Tests

```bash
pytest tests/ -q
```

## Version

**0.4.1** — coding-interface loop + independent `review` seat (no stack bridge yet).

MIT · Temple of Two
