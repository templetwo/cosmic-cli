# Grok Build seam contract — source-verified

**Status:** source-verified (not docs-only)  
**Date:** 2026-07-19  
**Upstream tree:** [xai-org/grok-build](https://github.com/xai-org/grok-build)  
**Pinned monorepo SHA (`SOURCE_REV`):** `ba69d70c2f7d70a130a323b2becdf137af784c7f`  
**Pinned published tree HEAD:** `ba76b0a` (*Synced from monorepo*)  
**Local mirror used for this pass:** `/tmp/grok-build-src`  
**Constraint:** `CONTRIBUTING.md` — external contributions not accepted. Compose via hooks, plugins, skills, MCP, sandbox profiles, ACP. No upstream PRs.

This document is the honesty upgrade both outside audits asked for: the cosmic↔grok contract is pinned to Rust source, not only user-guide prose.

---

## What cosmic depends on (and source confirms)

### 1. PreToolUse is first among runtime gates (after plan-mode)

Call site: `xai-grok-shell` `session/acp_session_impl/tool_calls.rs`

Order for a tool call:

1. Plan-mode edit gate (if plan mode)
2. **File/registry `PreToolUse` hooks** via `dispatch_pre_tool_use`
3. **Client `PreToolUse` hooks**
4. Permission system (allow/deny/ask rules, remembered grants, mode)

Docs claimed hooks run earliest; source matches. Cosmic’s global `~/.grok/hooks/cosmic-pretooluse.json` is on this path.

### 2. Decision aggregation: first explicit deny wins; allow never skips later deniers

`xai-grok-hooks/src/dispatcher.rs` — `dispatch_pre_tool_use`:

- Hooks run **sequentially in config order**
- Only an explicit **`deny`** stops the chain
- Tests: `first_deny_wins_short_circuits`, `allow_then_deny_denies`
- Empty registry → **Allow**

JSON vocabulary (`runner/mod.rs` `GateHookJson`):

```json
{"decision": "allow" | "deny", "reason": "…"}
```

Exit code **2** is the gate deny exit (`GATE_EXIT_CODE`). JSON `deny` wins even if exit is not 2 (with a warning). Exit 2 with JSON `allow` → **deny** (stdout ignored on exit 2).

Cosmic wrapper: emit `{"decision":"deny",…}` + exit 2 unless stdout is exactly `COSMIC-ALLOW v1 <nonce>`.

### 3. Native hooks are fail-open on failure — cosmic’s positive protocol is the counter

Same `dispatch_pre_tool_use` rustdoc (verbatim intent):

> Hook failures (timeouts, crashes, command-not-found, env-var pre-spawn refusals, malformed output) are **fail-open**: the failure is logged … but the tool call continues as if the hook had allowed it.

`HookRunResult::Failed` / `HookRunnerResult::Failed` → continue as Allow.

**Cosmic value-add (not a grok feature):** the wrapper’s positive protocol — *no valid sentinel ⇒ deny* — is **fail-closed** at the policy boundary even though the platform fails open on hook crash. That is a deliberate, source-grounded composition: we only emit the sentinel on a genuine OPEN from `cosmic-cli gate`. A broken gate (empty stdout) denies.

Deposit language: do **not** claim “grok hooks fail closed.” Claim: “cosmic’s positive-protocol wrapper is fail-closed; platform hook failures alone are fail-open.”

### 4. Hook process environment: inherited; no scrub of operator exports

`xai-grok-hooks/src/runner/command.rs`:

- Child `Command` does **not** clear the environment
- Spawn applies `extra_env` then runner-injected keys only:
  - `GROK_HOOK_EVENT`, `GROK_HOOK_NAME`, `GROK_SESSION_ID`, `GROK_WORKSPACE_ROOT`, `CLAUDE_PROJECT_DIR`
- Process env is inherited; `find_unresolved_env_vars` treats process env as a valid resolution source

`env_expand.rs`: config-time `${VAR}` expansion; command paths re-expanded by `sh -c` at runtime for shell-metacharacter commands. No filtering of `COSMIC_*`.

**Live + source agreement:** operator `export COSMIC_APPROVAL_TOKEN=…` before launching `grok` reaches the PreToolUse hook and the gate child. L2 approve path stands.

### 5. Subagents inherit parent PreToolUse gates

- `session/acp_session_tests/client_hooks_tests.rs` — `subagent_inherits_parent_pre_tool_use_client_hook`
- `session/commands.rs` notes PreToolUse over the parent connection
- Envelope carries `subagent_type` for matchers

Cosmic’s global PreToolUse therefore covers subagent tool calls when hooks are active on the parent session.

### 6. Sandbox: process-wide kernel floor; custom `deny` is real

Crate: `xai-grok-sandbox` (Landlock / Seatbelt via `nono`).

- Applied **once at process startup**; irreversible for the process lifetime
- Covers in-process FS **and** child processes (bash inherits)
- Profiles: `workspace`, `devbox`, `read-only`, `strict`, `off`, plus custom from:
  - `~/.grok/sandbox.toml` (global)
  - `.grok/sandbox.toml` (project; **additive only** — cannot redefine a global profile name)
- Custom profile fields: `extends`, `restrict_network`, `read_only`, `read_write`, **`deny`**
- **`deny` overrides grants** — kernel-enforced read + write/rename (macOS Seatbelt sub-actions close `mv` relocate bypass; Linux bwrap bind-over for read-deny)
- Network restrict: **Linux child seccomp only**; macOS network block is a no-op
- Custom profile with non-empty `deny` that cannot be applied → **refuse to start** (fail closed)

Temple compose (live, this machine):

```toml
[profiles.temple]
extends = "workspace"
deny = [
  "/Users/…/.cosmic-cli/last_pause_token.json",
  "/Users/…/.cosmic-cli/operator_approval_token",
  "/Users/…/.cosmic-cli/local_approvals.json",
  "/Users/…/.cosmic-cli/local_approvals.json.lock",
]
```

**Do not deny the whole `~/.cosmic-cli` directory** — PreToolUse wrapper lives at `~/.cosmic-cli/hooks/` and must remain executable under the same profile.

Live receipt: `GROK_SANDBOX=temple` + `cat …/last_pause_token.json` → `Operation not permitted` (Seatbelt). Benign shell still works.

Example shipped: `scripts/grok-sandbox-temple.toml.example`.

---

## Value-add boundary (defensible, not slogan)

| Layer | Grok Build (native) | Cosmic (avionics) |
|-------|---------------------|-------------------|
| Kernel FS floor | Sandbox profiles + `deny` globs | Temple profile compose; also Seatbelt under `cosmic-cli do` shell |
| PreToolUse | Hooks, first-deny, **fail-open on hook failure** | Positive-protocol sentinel wrapper (**fail-closed** if gate silent) |
| Permission modes | allow/deny/ask, modes, remembered grants | Local compass WITNESS/PAUSE/OPEN + check_shell |
| Memory / witness | Session memory (optional) | **T2Helix** chronicle + Helix witness bridge |
| PAUSE | Not a first-class action-bound token | Exactly-once, action-bound `claim_once`, TTY-only L2 channel, ranking law |
| Cross-cockpit | Grok-native | Unified gate verb for grok + Claude hook envelopes |
| Receipts | Tool/session telemetry | Chronicle / deposit-grade receipts |

**Cosmic is avionics, not another agent:** the cockpit owns TUI, tools, sandbox kernel, and permission UI. Cosmic owns mission protocol, Helix memory, compass lattice, ranking, and the positive-protocol seam that hardens fail-open hooks.

---

## Explicit non-claims (honesty)

- We do **not** claim grok’s default hook posture is fail-closed.
- We do **not** claim macOS sandbox blocks child network (source: Linux-only).
- We do **not** send PRs upstream (`CONTRIBUTING.md`).
- Checkpoint cross-pollination with `xai-grok-workspace` is **not** source-verified in this pass (tracked, non-blocking for deposit).
- Published tree is a periodic monorepo sync; pin both `SOURCE_REV` and tree HEAD when citing.

---

## Re-verify recipe (next instance / Claude)

```bash
git clone --depth 1 https://github.com/xai-org/grok-build.git /tmp/grok-build-src
cat /tmp/grok-build-src/SOURCE_REV   # expect ba69d70c2f7d70a130a323b2becdf137af784c7f (until next sync)
# Spot-check:
#   crates/codegen/xai-grok-hooks/src/dispatcher.rs  — fail-open + first deny
#   crates/codegen/xai-grok-hooks/src/runner/command.rs — env inherit, GATE_EXIT_CODE=2
#   crates/codegen/xai-grok-sandbox/src/{lib,profiles,deny}.rs
#   crates/codegen/xai-grok-shell/src/session/acp_session_impl/tool_calls.rs — order
GROK_SANDBOX=temple grok -p '…'   # live floor probe
```

When `SOURCE_REV` moves, re-run this pass and supersede this file (receipts, never quiet edit).
