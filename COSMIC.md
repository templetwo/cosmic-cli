# Cosmic CLI — Standing Horizon

**Locked:** 2026-07-14  
**Partners:** Grok Build (builder) · Claude Code (support)

## Horizon (A+B)

**Grok Build is the cockpit. Cosmic is the avionics.**

Cosmic is **not** another agent TUI. Grok Build occupies that seat.

Cosmic is the **Temple runtime**:
- Mission protocol (`do` / Stargazer tool loop)
- **T2Helix** local chronicle as memory + task substrate (shared with Claude seats)
- Receipts, review seat, project law (`COSMIC.md`)
- Compass-aware shell (WITNESS hard-deny, PAUSE token-gate)

Not: prettier TUI than Grok Build.  
Not: direct Sovereign Stack connector (Helix is the joint).  
Yes: headless / scriptable Temple agent the cockpit can call.

## Compass honesty (load-bearing)

Until 2026-07-14, non-hook seats treated **PAUSE as OPEN** in `grok-adapter.js`
(logged, not enforced). Same failure shape as a `safe` flag that is never read.

**Current contract (Cosmic + adapter):**

| Class | Behavior |
|-------|----------|
| **WITNESS** | Hard block. Does not execute. |
| **PAUSE** | Soft block. Mints single-use token. Approve with `cosmic-cli helix confirm <token>`, then retry. |
| **OPEN** | Proceed. |

**`--mode full`:** explicit blast-radius opt-in. Skips local policy + check_shell
network rules + Helix witness. Documented here so the honesty table is not a lie.

**Wire shape (Claude experiment 2026-07-14):** Cosmic must send shell as  
`{tool_name: "Bash", tool_input: {command: cmd}}`.  
Plain `"SHELL: …"` strings are tagged `Grok` in the adapter and **skip every Bash-scoped rule** — enforcement code never fires. Fixed in Cosmic: always Bash-structured.

**Override loop (Claude experiment #2):** Approvals key by `(session_id, action_summary)`.
Cosmic reuses Helix/Claude seat session id (not a fresh mint per `do`) so
`helix confirm` then re-run can consume the approval. On PAUSE/WITNESS block,
Stargazer **stops and surfaces the token** (no thrash-retry). Exit code **4** = blocked.

**Execution coverage (Claude experiment #3):** SHELL **and** CODE both pass through
`_compass_gate` (local `check_shell` + Helix Bash-structured witness). Certifying
only the shell door while CODE ran unguarded was the half-true claim.

**READ / redaction (Claude experiment #4):** Sensitive-path blocking is solid
(`.env`, `.ssh`, key names, resolve+recheck). Content `redact()` must mirror the
t2helix `secrets.js` vocabulary (AWS AKIA, GitHub `ghp_`, Stripe `sk_live_`,
Google `AIza`, Slack `xox…`, npm, SendGrid, …) — not a partial subset. Same
failure shape: path lock claimed complete while ordinary files leaked provider
tokens into model context and mission logs.

**Ollama path removed (2026-07-14):** Project evolved past the local Ollama twin.
One Stargazer runtime (Grok + Helix). No second, less-gated execution path.

**Consciousness + plugins archived (2026-07-14):** Unwired metrics/research layer
and the FileOperations plugin scaffold moved to `_archive/`. Not on the mission
path; not shipped as package surface.

If a README says "compass protects both ✓" without distinguishing PAUSE, treat that claim as **under-specified**. Prefer this table.

## Convergence (load-bearing)

Refinement converges or it consumes. A target that is fixed, external, and binary
can be reached; "make it sharper" cannot, and chasing it removes metal until the
edge is gone. This project refines toward tests, never toward the absence of an
imaginable weakness. This is Compass honesty turned inward: the same rule that
forbids claiming a gate that only logs forbids claiming a defect that only exists
in a reviewer's head.

- **The conformance suite is the terminal reviewer.** Once it is green for a
  version, a describable weakness is not a defect until it is a failing test or a
  working reproducer. Imagination is not a finding.
- **Shipped code freezes on meeting its version's exit criteria.** Frozen code
  unfreezes only for (a) a red conformance test, or (b) an external report with a
  working reproducer. Not for a new review pass, a hypothetical, or a refactor
  itch while someone is already in the file.
- **Out-of-scope ideas go to the backlog for the next version, the moment they
  appear.** The working set holds only current-milestone work. Ideas do not die;
  they queue.

Frozen at v0.8.5 (`aeddf0b`): `policy`, `rules`, `gateway`, `checkpoint`,
`action_bind`, `tools` path safety, and the `agents` gate paths.

**Unfreeze receipt (v0.8.6):** conformance suite (`tests/test_conformance.py`)
seated green. Box 4a (fcntl refuse) and 4b (seal-after-verify) were red
conformance tests — the only legal unfreeze of `gateway.py` under this section —
then fixed and flipped to pass. Terminal reviewer is seated.

Stop signal: if a change is not required by a current exit criterion, it is
next-version work, and the next version has not begun.

Non-goal: sharpening past convergence. A fail-closed gate fails closed against its
own worst habit too.

## Compass Rules

**Status:** human-readable view loaded by `cosmic_cli.rules.load_rules_from_markdown`
on every mission (when this file exists in the work root). Authoritative remote
compass remains T2Helix; this table is the **local** policy kernel door
(`ActionGateway` → then `check_shell` → then Helix witness).

| ID | Type | Scope | Pattern |
|----|------|-------|---------|
| destructive-rm | WITNESS | SHELL,CODE | rm -rf |
| outbound-net | PAUSE | SHELL,CODE,NETWORK | curl |
| example-allow | OPEN | SHELL | safe-command |

**Loader:** 4-column table under this heading. Severity lattice WITNESS > PAUSE > OPEN;
all matches reported; unknown dispositions rejected at load. Local PAUSE mints a
single-use action-bound token (`COSMIC_APPROVAL_TOKEN`); Helix PAUSE still uses
`cosmic-cli helix confirm <token>`.

**Avionics stack (v0.8.5):** **exactly-once PAUSE** (`claim_once` under flock);
canonical length/hash EDIT·WRITE bindings (no newline collision); operator
token file `~/.cosmic-cli/last_pause_token.json` via `helix show-pause-token`
(never model return / stderr / ui_callback).

## Memory

- Substrate: T2Helix SQLite (`T2HELIX_DATA_DIR`, default Claude plugin data dir when present)
- Bridge: `scripts/helix_rpc.js` + `cosmic_cli/helix_bridge.py`
- Mission FINISH → `helix.record`; PASS → `open_thread`; MEMORY → `helix.recall`

## Daily path

```bash
export T2HELIX_ROOT=~/t2helix
cosmic-cli doctor
cosmic-cli helix status
cosmic-cli do '…'
cosmic-cli helix confirm <token>   # after PAUSE
cosmic-cli do --review '…'
```

## Non-goals

- Competing with Grok Build / Claude Code as a fullscreen TUI
- Safe-mode allowlist deep work (Anthony: not worried)
- Lying about gates that only log
- Reviving Ollama / consciousness / plugin twins without a clear capability win
