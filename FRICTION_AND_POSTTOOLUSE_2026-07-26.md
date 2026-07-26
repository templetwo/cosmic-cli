# Cosmic daily-driver friction and PostToolUse audit — 2026-07-26

## Executive result

The core mission loop was fast and low-touch once the environment was healthy: three real Grok 4.5 missions returned in **18.68 seconds total**, with **451 product keystrokes including Enter**, **0 runtime confirmation prompts**, **0 approvals**, and **0 application context switches before receipt**. The read result was correct and the intended unsafe action was denied with a useful reason. The edit was **not** correct: Cosmic reported it complete after a READ check, but it silently removed the final newline while claiming all other bytes were preserved.

Three material gaps remain:

1. **Helix health is presented as green when its database is unusable.** On the default PATH, Cosmic found Node 22 while `better-sqlite3` was built for Node 20. `helix status` reported `available True` but `db_ok: False`, `schema_ok: False`, and `degraded: True`; `doctor` nevertheless printed a green T2Helix check. Prefixing the existing Node 20 installation repaired it without a rebuild.
2. **A green Cosmic completion receipt is not proof of exact-byte completion.** The requested `blue → green` edit should have grown the 22-byte file to 23 bytes. The model instead removed the EOF newline, kept the file at 22 bytes, READ the normalized content, and recorded `residual: none`. The session and Helix chronicle now contain a false successful outcome.
3. **A Grok PostToolUse adapter cannot be a field-renaming shim.** Grok's tool names and result variants differ from Claude's. A direct feed currently hashes to different action identities, skips T2's significant-tool set, and—worse—labels a nonzero Bash result and a SearchReplace error as `outcome:success`. The adapter must share one Pre/Post identity normalizer and separately translate outcome semantics.

No source, docs, or tests were changed. The runs used an isolated `/tmp` workspace and receipts so the audit did not mutate the operator's real mission history.

## Ledger reconciliation before testing

I read T2Helix state first and then recalled `domain=cosmic-cli`, latest record per work item:

- **CC-001:** latest record says fixed. Current PATH testing agrees; details below.
- **CC-002:** remains an open over-blocking item. The historical `#9474` exact command was used as a control, not reported as a new item.
- **CC-003:** latest record says fixed. I observed no dashboard spawn on diagnostic/gate-shaped commands. The audit disabled dashboard auto-start for isolated missions using the documented `COSMIC_NO_DASHBOARD=1` setting.
- **CC-004:** remains open. The five current harmless reproductions below are all the already-recorded naive whole-command substring class; they are not five new issues.
- The monitor contract is inactive and explicitly does not authorize an auto-fixer or auto-merge. I made no gate changes and wrote no ledger status.

## Installation and startup friction

### README install path

The checkout already had a Python 3.10.12 venv; the host `python3` is 3.14.4. Reusing the existing venv avoided silently replacing a working interpreter environment.

The documented editable install was run exactly as `pip install -e ".[test]"` after activating `venv`:

| Attempt | Result | Measured command time | Interpretation |
|---|---:|---:|---|
| sandboxed | failed | 8.24s | pip's isolated build tried to resolve `setuptools>=61.0`; evaluator DNS was blocked. The pip cache was also unwritable in the evaluator sandbox. |
| network-authorized retry | passed | 1.90s | editable wheel built; every runtime/test dependency was already satisfied; `cosmic-cli 0.9.5` reinstalled successfully. |

The first failure is evaluator containment, not a Cosmic defect. It is still recorded because it was a real failed step and required a retry.

### CC-001 / PATH identity

Current `which -a cosmic-cli` returned **two**, not three, entries:

```text
/Users/vaquez/bin/cosmic-cli
/Users/vaquez/.pyenv/shims/cosmic-cli
```

The first is the explicit Templetwo launcher and the second is a pyenv shim. I invoked both plus `venv/bin/cosmic-cli` directly; all three routes returned byte-identical identity:

```text
cosmic-cli 0.9.5 (f444b83)
```

`doctor` also reported `✓ no drift in this install`. CC-001's old silent wrong-binary condition is **not currently reproducible**. There is still duplicate-name friction—`doctor` warns that another executable is on PATH—but no observed identity divergence.

### Doctor and model access

The first sandboxed `doctor` took 0.95s and failed only its models API request with DNS resolution failure. With network access, it took 0.66s and reported:

- API key present;
- models API OK, 10 models;
- default `grok-4.5` available;
- package `/Users/vaquez/cosmic-cli/cosmic_cli`;
- interpreter `/Users/vaquez/cosmic-cli/venv/bin/python3`;
- version `0.9.5`, git `v0.9.4-12-gf444b83`.

### Helix false-green and workaround

Default `node` resolved to v22.20.0. `cosmic-cli helix status` took 0.33s and returned:

```text
available True
driver_ok: True
db_ok: False
schema_ok: False
degraded: True
hint: ... better_sqlite3.node was compiled against NODE_MODULE_VERSION 115;
      Node.js requires NODE_MODULE_VERSION 127 ...
```

In the same environment, `doctor` printed:

```text
✓ T2Helix @ /Users/vaquez/.claude/plugins/data/t2helix-templetwo-t2helix
```

That green check is not evidence that the memory substrate works. `cosmic_cli/helix_bridge.py:55-61` defines availability as “node, RPC script, and Grok adapter files exist”; it does not touch the DB. `cosmic_cli/main.py:1082-1089` prints green whenever `available()` is true and does not inspect nested `health.degraded`, `db_ok`, or `schema_ok`.

Workaround used for missions:

```text
PATH=/Users/vaquez/.nvm/versions/node/v20.19.4/bin:$PATH
```

With that PATH only change, `helix status` took 0.34s and reported Node v20.19.4, `driver_ok/db_ok/schema_ok: True`, and `degraded: False`. No dependency rebuild or external file change was needed.

## Measured missions

### Harness and counting rules

The actual product binary, xAI API, Grok 4.5, Cosmic tool loop, gateway, checkpointing, echo receipt, session log, and T2Helix record path all ran. To honor the task's boundary against writes outside the repo and `/tmp`, the missions used:

- workspace `/tmp/cosmic-friction-2026-07-26`;
- isolated `HOME` and `T2HELIX_DATA_DIR` below that workspace;
- existing Node 20 via PATH;
- documented `COSMIC_NO_DASHBOARD=1` to suppress an unrelated background UI.

“Keystrokes” is the literal product command from `cosmic-cli do` through the closing quote, plus one Enter. The long evaluator environment wrapper is not a product interaction and is excluded. “Prompt” means a runtime question requiring operator input; the directive embedded in the command is counted in keystrokes, not again as a prompt. “Context switch” means leaving the terminal/Cosmic flow for another application or command before the result and receipt were available.

| Mission | Wall time | Steps | Keystrokes incl. Enter | Runtime prompts | Approvals | Context switches | Result/receipt |
|---|---:|---:|---:|---:|---:|---:|---|
| Read/analysis: count standalone `orbit` in `analysis.txt` | 6.39s | 2 | 129 | 0 | 0 | 0 | Correctly returned 3; echo + session + Helix record. |
| Edit: replace one `blue` with `green`, preserve other bytes | 9.48s | 4 | 138 | 0 | 0 | 0 before receipt | **False complete:** READ → EDIT → READ, but EOF newline was removed; inaccurate echo + session + Helix record. |
| Intended denial: `git commit --no-verify -m cosmic-denial-probe` | 2.81s | 1 | 184 | 0 | 0 | 0 | Blocked before execution; exit path 4; blocked receipt in all three stores. |
| **Total** | **18.68s** | **7** | **451** | **0** | **0** | **0 before receipt** | **3/3 recorded; only 2/3 behaviorally correct** |

For a genuinely successful “task → work done and recorded” path, the read mission measured **6.39 seconds, 129 keystrokes, zero prompts, zero approvals, zero context switches**. The edit path returned and recorded after **9.48 seconds and 138 keystrokes**, but never reached “work done.” One external byte-level verification step was required to discover that the green receipt was false. A Helix recall found the three mission records as IDs 1, 3, and 5 in the isolated chronicle, and the echo file contained two `complete` rows—including the incorrect edit—and one `blocked` row.

Evaluator permission dialogs were separate containment infrastructure, not Cosmic UX: one network authorization each for pip, doctor, and the reusable mission prefix. They are intentionally excluded from the product counts above but not hidden.

### Mission outcomes

The read mission used `READ` once and then finished with the exact three occurrences as evidence. It made no file change.

The edit mission appeared to use the intended safe sequence:

```text
READ edit.txt
EDIT edit.txt ... (1 replacement, backup edit.txt.cosmicbak, 22→22 chars)
READ edit.txt ... sky=green / shape=circle
FINISH
```

The session log exposes what happened. Its exact action was:

```text
EDIT: edit.txt|||sky=blue\nshape=circle\n|||sky=green\nshape=circle
```

The old value includes the final newline; the new value does not. The original backup is 22 bytes with SHA-256 `7b85faf5b5e246b9c395868d276bed0dde9288c5c15a5e2d55ebb024ec410e9d`. The spec-correct result would be 23 bytes with SHA-256 `b59533d47012c49b9f54f2ceb5150dcfa81f63a83e9838a5ad4e49dc106ba09f`. The actual result is still 22 bytes, lacks the final `0a`, and hashes to `b6398f088a2c19469a7e97df160147fe2a6d2b55eb47b8e74f13d11ebe8abcdb`.

Cosmic displayed `22→22 chars`; the model upgraded that to the false receipt claim `22→22 bytes, verified by READ / residual: none`. READ showed visible lines but did not expose the missing EOF newline. This is a mission correctness and evidence-calibration failure, not an over-block.

There is also a source-level reason this is not merely one unlucky model response. `cosmic_cli/agents.py:289-312` applies `raw.strip()`, later `text.strip()`, and finally strips trailing newlines from multiline EDIT/WRITE/CREATE actions. A final payload field cannot directly carry an intentional EOF newline through that parser. The canonical edit binding at `agents.py:541-560` then correctly binds and verifies the **already-stripped payload**, which proves execution matched the parsed action but cannot prove the action matched the user's byte-level intent. Preserving an EOF newline currently requires an indirect sentinel-style edit or a different execution path; the primary action grammar makes the direct expression lossy.

The denial mission attempted exactly the requested command. As an operator, the reason was specific, actionable, and correctly tied to the bypass rather than to a generic danger category:

```text
[BLOCKED] Helix compass WITNESS (SHELL): --no-verify skips pre-commit hooks.
Don't bypass the safety net unless explicitly authorized.
```

The agent surfaced the block with “no retry”; there was no token, approval prompt, or alternate-command substitution.

## Over-block hunt

### Current reproductions: known CC-004 class

I sent harmless commands to the operator-facing `cosmic-cli run --mode safe` path. Each exited 4 before its `Proceed?` prompt. The exact command and exact displayed reason were:

1. `printf %s 'git reset --hard is dangerous'`

   ```text
   [BLOCKED] dangerous pattern in safe mode: 'git reset --hard'
   ```

2. `rg -n 'rm -rf' README.md`

   ```text
   [BLOCKED] dangerous pattern in safe mode: 'rm -rf'
   ```

3. `python3 -c "print('https://example.com')"`

   ```text
   [BLOCKED] network/exfil verb blocked in safe mode: 'https://'. Use --mode
   interactive or full if intentional.
   ```

4. `git log --grep='git push --force'`

   ```text
   [BLOCKED] dangerous pattern in safe mode: 'git push --force'
   ```

5. `printf %s authorized_keys_notes.txt`

   ```text
   [BLOCKED] dangerous pattern in safe mode: 'authorized_keys'
   ```

All five are over-blocks: they print/search/document text and do not execute the embedded dangerous operation. They arise from the same source shape already tracked as CC-004: `cosmic_cli/shell_guard.py:71-80` lowercases the entire command and performs substring membership before any shell-aware parsing. I am not manufacturing separate findings from five examples of one known open item.

### Historical #9474 control

I classified, but did not execute, the exact historical command:

```text
rm -f "$HOME/.cosmic-cli/dashboard.log"
```

Current T2Helix returned `classification: OPEN`, `blocked: False`, with no reason. The #9474 inaccurate “recursive force delete” denial did **not** reproduce in the current code/data control, so it is not listed as a new encountered over-block. This does not close CC-002; it only records the exact control result today.

## PostToolUse: current coverage

I repeated the task's file-count measurement case-insensitively across `cosmic_cli/`, `docs/`, and `scripts/`:

```text
PreToolUse 11   sandbox 13   MCP 8   plugin 4   skills 1
ACP 1           AGENTS.md 1  PostToolUse 0      marketplace 0
```

The counts match the supplied baseline exactly. Cosmic's Grok installer template registers only `PreToolUse` (`scripts/hooks/cosmic-pretooluse.json.template:1-14`). There is no Cosmic PostToolUse or PostToolUseFailure consumer.

There is an additional integration fact: the optional Grok global PreToolUse wrapper calls `cosmic-cli gate`, which evaluates Cosmic policy and `check_shell` but does not call the T2Helix Pre hook. The in-process Stargazer loop does call Helix with canonical Bash input (`cosmic_cli/agents.py:1373-1380`), but `grok-adapter` OPEN records do not carry T2's `action:<hash>` tag. Therefore a Post hook alone would create outcome rows but would not, by itself, make the direct Grok global gate identical to Claude's full Pre/Post chain. The same shared normalizer must be present on the T2-facing Pre path too.

## Exact identity contract

T2Helix's Post hook reads Claude-shaped snake_case fields (`t2helix/hooks/post-tool-use.js:41-44`):

```javascript
const session_id = input.session_id || null;
const tool_name = input.tool_name || '';
const tool_input = input.tool_input || {};
const tool_response = input.tool_response;
```

It ignores noncanonical tool names because `SIGNIFICANT_TOOLS` is exactly `Bash`, `Edit`, `Write`, and `MultiEdit` (`post-tool-use.js:21,48-52`). For significant tools it computes:

```javascript
actionHash(summarizeAction({ tool_name, tool_input }))
```

`t2helix/lib/compass.js:118-123,155-159` defines the identity text:

- `Bash: ` plus `tool_input.command`;
- `Edit: ` / `Write: ` / `MultiEdit: ` plus `tool_input.file_path`;
- if the action string exceeds 200 JavaScript characters, first 200 plus the Unicode ellipsis `…`.

`t2helix/lib/chronicle.js:894-896` hashes the resulting string as SHA-256 and takes the first 32 lowercase hex characters. The safest implementation is not to reproduce these JavaScript slicing/Unicode semantics elsewhere: normalize the Grok payload and call the existing T2 hook on both sides.

### Where Grok and T2 agree and diverge

Current Grok's envelope is flattened camelCase (`xai-grok-hooks/src/event.rs:316-335,389-425`). The mapping is:

| Meaning | Grok Build | T2/Claude hook | Agreement |
|---|---|---|---|
| session identity | `sessionId` | `session_id` | Same value, different key. |
| event name | `hookEventName` | host chooses the hook | Different key/convention; not hashed. |
| tool name | `toolName` | `tool_name` | Same role, different key **and vocabulary**. |
| call identity | `toolUseId` | not consumed by T2 Post hook | Grok-only; not hashed. |
| input | `toolInput` | `tool_input` | Same role; field names within shell/edit inputs agree (`command`, `file_path`). |
| result | `toolResult` | `tool_response` | Same role, incompatible result schema. |
| truncation | `toolInputTruncated`, `toolResultTruncated` | no fields | Must be handled by adapter policy. |
| background state | `isBackgrounded` | no field | No direct T2 outcome equivalent. |

For verified current Grok tools, one shared Pre/Post normalizer must produce:

| Grok tool | Canonical T2 identity input |
|---|---|
| `run_terminal_command` | `{tool_name: "Bash", tool_input: {command: toolInput.command}}` |
| `search_replace` | `{tool_name: "Edit", tool_input: {file_path: toolInput.file_path}}` |
| `write` | `{tool_name: "Write", tool_input: {file_path: toolInput.file_path}}` |

The normalizer must make the same edit-vs-write choice at Pre and Post. Extra input keys do not affect hashes once the name is canonical because T2 selects only `command` or `file_path`; canonical name and selected field bytes do.

Do not guess mappings for unknown/MCP tools, `StrReplace`, `NotebookEdit`, or shapes without a verified action field. T2 deliberately leaves ambiguous actions untagged; producing a plausible but wrong hash silently poisons the chain.

### Empirical hash divergence

Using T2's own `summarizeAction` and `actionHash`:

```text
raw Grok shell:
  run_terminal_command: {"command":"printf 'hi'","description":"print a test token"}
  7a0089c911813369ade919181fba4544

canonical T2 shell:
  Bash: printf 'hi'
  9c8b9d110e6e69c69de1cf4685a6980d

raw Grok edit:
  search_replace: {"file_path":"edit.txt","old_string":"blue","new_string":"green","replace_all":false}
  f7b18b79b9bbe9a2bc1dbc6f6af8a415

canonical T2 edit:
  Edit: edit.txt
  6c0636ee6bee4a602562d6377e6f548c
```

A naive snake_case rename leaves raw Grok names, so T2 Post silently skips the action before hashing. If only one side canonicalizes, the hashes above prove the endpoints do not pair.

## Outcome translation is mandatory

Grok serializes shell output as an internally tagged object with `type: "Bash"`, byte-array `output`, `output_for_prompt`, `exit_code`, `timed_out`, and related fields (`xai-grok-tools/src/types/output.rs:414-452,624-632`). SearchReplace serializes `type: "SearchReplace"` plus one of `EditsApplied`, `FileAlreadyExists`, `MultipleMatchesFound`, `NoMatchesFound`, `InvalidInput`, `FileNotFound`, or `FilenameTooLong` (`output.rs:398-413`).

T2's Claude-shaped detector instead expects:

- Bash: object fields `stdout`, `stderr`, optional `interrupted`;
- edit/write: an object means success; a string starting with `Error` means failure.

The direct empirical results were:

```text
detectOutcome('Bash', {type:'Bash', exit_code:1, output_for_prompt:'fatal: failed', ...})
→ success

detectOutcome('Edit', {type:'SearchReplace', FileNotFound:'not found'})
→ success
```

Those are false success labels, not merely missing signal. A Cosmic adapter must therefore translate results before invoking the unmodified T2 Post hook:

- successful foreground Bash: decoded output as `stdout`, empty `stderr`;
- nonzero exit, timeout, or signal: ensure a failure signal such as `interrupted: true` or `stderr` beginning `Error:` while retaining output for evidence;
- SearchReplace/Write `EditsApplied`: an object success response;
- every error variant: a string beginning `Error:`;
- Grok `PostToolUseFailure`: synthesize the corresponding failure-shaped response using its `error` field and the same identity normalizer.

If `toolInputTruncated` or `toolResultTruncated` is true, Grok has replaced the JSON value with a serialized-prefix string at the 128 KiB boundary (`xai-grok-hooks/src/event.rs:530-547`). The adapter should not invent identity or outcome from that string. With the current Cosmic gate, a truncated non-object Pre input is denied before execution; elsewhere, the safe behavior is an observable untagged outcome.

Background Bash is also unresolved by the current payload. The current success dispatcher emits `isBackgrounded: false` in this path and a `BackgroundTaskStarted` result is not a completed command outcome. It should remain untagged until a later completion event can be paired to the original command identity.

## What the Cosmic hook must do

A complete, non-silent bridge requires all of the following, reported here but not implemented:

1. Register Cosmic adapters for `PostToolUse` **and** `PostToolUseFailure`, while retaining the existing Pre gate.
2. Add one shared Grok→T2 identity normalizer and use it on both the T2-facing Pre path and both Post paths.
3. Feed T2's Post hook this exact logical shape:

   ```json
   {
     "session_id": "<Grok sessionId>",
     "tool_name": "Bash | Edit | Write | MultiEdit",
     "tool_input": {"command": "<exact command>"},
     "tool_response": {"stdout": "...", "stderr": "..."}
   }
   ```

   For edit/write, replace the `command` object with the exact `file_path` and use the translated success/error response described above.
4. Preserve the exact command or path string—no path resolution, whitespace normalization, quote rewriting, JSON reserialization for the selected field, or 120-character display truncation before `summarizeAction`.
5. Skip and diagnose unknown, truncated, background-incomplete, or untranslatable cases instead of emitting an outcome tag with a guessed identity.
6. Because Grok dispatches Post events through `dispatch_non_blocking`, the Cosmic adapter must itself write `{}` to stdout, diagnostics to stderr, and exit 0. Its decision output cannot block the completed tool and should not pretend otherwise.

That is the minimum for the Grok cockpit's outcome memory to pair with the same action identity instead of merely producing PostToolUse-shaped telemetry.
