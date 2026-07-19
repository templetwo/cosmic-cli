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

**TERMINAL REVIEWER SEATED** at v0.8.7 (`ba5287f`). Conformance suite
(`tests/test_conformance.py`) is 8/8 green and is the terminal reviewer for
these properties. Frozen modules: `policy`, `rules`, `gateway`, `checkpoint`,
`action_bind`, `tools` path safety, and the `agents` gate paths.

Prior freeze was v0.8.5 (`aeddf0b`). Legal unfreeze receipts: box 4a/4b red
tests → fixes; sequential-mutation regression (seal) → seal/un-seal pair.
Re-frozen at `ba5287f` after soundness pass clean. Imagination is not a finding.

Stop signal: if a change is not required by a current exit criterion, it is
next-version work, and the next version has not begun. Hardening of this core
is done; remaining work is construction, not refinement.

## Ranking (load-bearing)

Sibling of Convergence. Anthony's insight 2026-07-19: *"you approve the cosmic
cli and I approve you"* — a privilege ranking, not only a collaboration habit.

**Law:** Approval of an action at level N must be issued from a level **strictly
greater than N**, through a channel no actor at level ≤ N can reach. No actor
approves its own action or its own level. (Biba integrity / protection rings.)

| Level | Who | Approves |
|-------|-----|----------|
| **L3** | Anthony (human owner) | push, DOI deposit, stack-policy enactment |
| **L2** | Reviewer seat / operator at a TTY | cosmic-cli behavior, PAUSE approval |
| **L1** | cosmic-cli / compass | gates L0 (WITNESS/PAUSE/OPEN); not self-governance |
| **L0** | Gated model / agent shell | **nothing** — cannot reach any approval channel |

**Role sets the level, not identity.** The same Claude is L2 reviewing Grok and
L0 the moment it drives a gated shell. A human at a TTY is L2+; a non-interactive
agent shell is L0, whoever is driving it.

**PAUSE channel is L2-only by construction** (`cosmic_cli/ranking.py`):

1. `helix accept-pause` / `show-pause-token` / `confirm` require interactive TTY
   (or `COSMIC_L2_OPERATOR=1` for scripted L2 seats). L0 exit 4 otherwise.
2. L1 WITNESS: `check_shell` denies any tool call touching the approval surface
   (`accept-pause`, token files, `COSMIC_APPROVAL_TOKEN=`).
3. Token paths are sensitive: L0 READ/WRITE tools cannot open
   `last_pause_token.json` / `operator_approval_token`.

**Conformance:** `tests/test_privilege_ranking.py` — the exact self-approval
reproducer (L0 runs accept-pause, retries) must return **deny**. When that is
green, the class is closed, not just the instance.

Full write-up: `PRIVILEGE_RANKING.md`. Stack standing-law enactment is Anthony's
(registry via `current_policies()`); this section is project law for the repo.

## Sentinel (v0.9 construction)

**Box 1:** `cosmic-cli gate --hook {grok,claude}` — COSMIC-ALLOW v1 sentinel
(`cosmic_cli/gate.py`).

**Box 2+3 landed:** positive-protocol PreToolUse wrapper + boot canary.

```bash
cosmic-cli init --grok --force          # install global hook + wrapper
cosmic-cli gate --boot-canary           # 4 deny + 1 allow; refuse if silent/broken
~/.cosmic-cli/hooks/cosmic-launch-grok.sh   # canary then exec grok
```

- Wrapper: deny unless entire stdout equals `COSMIC-ALLOW v1 <nonce>` (stderr discarded).
- Canary: shell/write/read/MCP must empty stdout; grep must emit sentinel.
- Live-verified: `rm -rf` → Hook denied (no sentinel); boot canary PASS.

PAUSE at the gate seam landed (box 4) + privilege ranking on the approval
channel. Still construction: checkpoint-in-gate, live Helix leg, deposit packaging.

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
| ranking-accept-pause | WITNESS | SHELL,CODE | accept-pause |
| ranking-show-pause | WITNESS | SHELL,CODE | show-pause-token |
| ranking-token-file | WITNESS | SHELL,CODE | last_pause_token |
| example-allow | OPEN | SHELL | safe-command |

*(Ranking rows are defense-in-depth; `check_shell` + TTY-gate also enforce L2-only.)*

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
