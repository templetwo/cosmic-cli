# Phase-1 PR checklist — Pilot Board + MissionBus v1

Source: Anthony, 2026-09-19. This is the bounded implementation assignment for the Grok Build branch, covering part of UPG-005/006 in [VERSION_UPGRADE_PLAN.md](VERSION_UPGRADE_PLAN.md).

Contract inputs: [Pilot Board layout](PILOT_BOARD_SPEC.md), [MissionBus v1](MISSION_BUS_SPEC.md), finish-line honesty, and L2-only PAUSE approval. This checklist takes precedence over broader blueprint deliverables for this PR. Preserve CLI and Mission Control through compatible event writing/reading. All boxes begin unchecked; no implementation or validation is claimed by this planning commit.

## Definition of done

- [ ] Session JSONL lines carry v, mission, seq, ts, session.
- [ ] Legacy start/step/end and mission.start/step.proposed/mission.end coexist through dual-write or readers that support both; define deduplication if dual-written.
- [ ] finish_basis appears only on finished statuses; verified iff finish_basis=verifier.
- [ ] Echo remains enabled where configured, retains existing keys, and adds only optional mission/ts in this slice.
- [ ] LocalMissionBus drives TUI updates rather than exclusive agent-log polling.
- [ ] `cosmic-cli tui` shows MissionRail, StepTape, and InstrumentStack; minimal instruments are sufficient.
- [ ] Board PAUSE approval/decline uses the shared CLI/gateway authority path; no token body appears on bus or UI.
- [ ] DiffPeek shows the last mutation path from fs.mutate or a temporary log-derived fallback.
- [ ] Finish-line and new bus tests pass; run `pytest tests/ -q` and accurately classify any battery residuals or environment limitations.
- [ ] Headless `cosmic-cli do` preserves status rendering and exit codes, including blocked=4 and existing finish semantics.
- [ ] Two concurrent directives remain supported and do not mix events, selection, or approvals.

## Explicit non-goals

- Command palette, Helix recall drawer, compact-tab polish.
- Full step.finished coverage or every tool event.
- Web-dashboard rewrite; its legacy tail continues to work.
- Unix sockets or multiprocess bus transport.
- Replacing the prompt_toolkit file browser.
- Changes to compass classification rules or the kernel floor.
- UPG-001's full typed shell-result refactor, UPG-002 progress detection, UPG-003 budgets, or UPG-004 consolidated handoff.

Full debrief modal/ReviewDock behavior from the broader board blueprint is not a Phase-1 ship gate; status/basis must still display honestly. Diff bodies, undo, cancellation, and richer instruments follow the backlog below.

## Preflight — read before coding

| File / seam | Purpose |
| --- | --- |
| cosmic_cli/agents.py | _session_write, _append_echo, execute, FINISH and BLOCKED paths |
| cosmic_cli/gateway.py | AuthorizationReceipt, mint/claim ownership, execute_with_receipt |
| cosmic_cli/ui.py | DirectivesUI, refresh, threading, APIKeyScreen |
| cosmic_cli/theme.py | Status, step_bar, palette; reuse these |
| cosmic_cli/main.py | tui/do wiring, dashboard allowlist, operator approval command |
| cosmic_cli/dashboard.py | Current session and echo readers |
| tests/test_finish_line.py | Finish/basis invariants and persistence tests |
| Helix bridge, gate, ranking, and accept-pause path | Reuse actual operator authority and single-use claim/staging logic |
| tests/conftest.py and tests/test_isolation.py | Existing protections against test pollution of live stores |

## Commit 1 — Bus and event foundation

- [x] Add `cosmic_cli/events.py`: statuses, finish bases, event names, validation, and normalize_legacy.
- [x] Single-source FINISHED_STATUSES or maintain a compatibility re-export from agents.
- [x] Add `cosmic_cli/bus.py`: publish, subscribe, unsubscribe; isolate subscriber exceptions from the agent loop.
- [x] Add `tests/test_bus_schema.py`: envelope fields, verified/verifier equivalence, absent basis for blocked/max_steps, and sensitive-data checks where implemented.
- [x] Validate namespace conversion without inventing missing evidence in legacy records.

Acceptance: unit tests pass; agent behavior is unchanged at this step.
Evidence: `.venv/bin/python -m pytest tests/ --ignore=tests/battery -q` → 386 passed (interpreter 3.10.12, import path this worktree). Dual-write aliases share the canonical seq (`compat`/`alias_of`); `iter_canonical` drops them. `unique_mission_stem` uses microseconds so concurrent missions do not share a file; session_id is unchanged.

## Commit 2 — Agent emission and compatibility

- [x] Initialize `_seq = 0` and an injectable LocalMissionBus per agent.
- [x] Add `_emit(event, **payload)` owning envelope creation, redaction, sequence assignment, publication, and raw JSONL append; no duplicate timestamp/session injection.
- [x] Emit compatible start events with identity.
- [x] Emit step.proposed with n, action verb, raw, head; keep legacy step consumers working.
- [x] Emit end with status, optional basis, edited, steps, warnings, outcome, and model.
- [x] Emit finish.declared immediately before the accepted finish end, with status, basis, synthesized, and redacted text.
- [x] On BLOCKED, emit compass.verdict when classification is known; end/echo have blocked status and no basis.
- [x] Add optional mission/ts to echo while retaining existing fields and write_echo behavior.
- [x] Preserve FINISH decision logic and existing BLOCKED return/no-thrash behavior.
- [x] Extend finish-line persistence tests for end/echo agreement.
- [x] Add scripted-agent sequence test and synthesized-finish bus test.

Acceptance: headless do works, and current dashboard-tail behavior remains compatible. If names change before reader migration, dual-write during the intermediate commits.
Evidence: `.venv/bin/python -m pytest tests/test_finish_line.py tests/test_bus_schema.py tests/test_bus_agent_emit.py tests/test_isolation.py -q` → 58 passed (26+22+7+3). `.venv/bin/python -m pytest tests/ --ignore=tests/battery -q` → 406 passed (interpreter 3.10.12, import path this worktree). Bus is canonical-only; JSONL dual-writes start/step/end aliases sharing seq. `unique_mission_stem` + `token_hex(2)` nonce always; session_id unchanged.

## Commit 3 — PAUSE and compass hooks

- [x] Emit gate.pause_minted at the actual mint point with action_summary, action_sha256, expiry, optional pending_id, and only safe opaque correlation information.
- [x] Never emit a full token; supplied token_id_prefix is optional and at most 8 characters, and should be omitted if it exposes credential bytes.
- [x] Emit gate.pause_resolved for approval, decline, expiry, or invalidity using the defined actor/correlation semantics.
- [x] Emit OPEN/WITNESS/PAUSE compass verdicts where authoritative classification is available.
- [x] Prefer the agent seam that knows session/mission; keep gateway free of UI imports.
- [x] If needed, inject an optional gateway event callback defaulting to no-op. (Not needed: agent emits; gateway stays UI-free.)
- [x] Test mint opacity and exactly-once approval claim.
- [ ] Decline semantics matching existing CLI behavior — deferred to commit 6; decline does not exist yet and was not faked.

Acceptance: `helix accept-pause` remains compatible and applicable gate/battery tests retain their expected results. Automatically expired/invalid records must not falsely claim an operator decision.
Evidence: `.venv/bin/python -m pytest tests/test_pause_bus_opacity.py tests/test_local_policy_gate.py tests/test_finish_line.py tests/test_bus_agent_emit.py tests/test_bus_schema.py -q` → 69 passed (6+8+26+7+22). `.venv/bin/python -m pytest tests/ --ignore=tests/battery -q` → 425 passed (interpreter 3.10.12, import path this worktree). Mint seams: `_run_mutation`, local `_compass_gate` PAUSE, Helix PAUSE. Local mint payload keys: `v, event, ts, session, mission, seq, action_summary, action_sha256, expires_at` (no `token` / `token_id_prefix` / `pending_id`). `claim_once` True → `gate.pause_resolved` `decision=approved` `by=operator`; failed claim → `invalid` without `by` (expired not distinguishable without new ApprovalManager API). Decline not emitted (commit 6; not faked). Authoritative `compass.verdict` PAUSE/WITNESS at the gate; OPEN is allow-through and is not invented. Gateway `on_event` skipped; agent emits. Battery not re-run.

## Commit 4 — High-value mutation, shell, and verifier events

- [x] Emit fs.mutate for successful EDIT/WRITE/CREATE through gateway/checkpoint: op, path, optional checkpoint_id/receipt_id.
- [x] Omit diff body initially or cap it at 8 KiB with truncation explicitly marked.
- [x] Emit shell.exec for SHELL/CODE/TEST: kind, redacted command, exit_code parsed only when present, blocked, output_head ≤500.
- [x] Emit verify.result for verify_cmd and auto_verify with distinct role values.
- [x] Never convert auto_verify success into mission verification.
- [x] Test successful mutation path emission and blocked shell output without false exit 0.

Parsing existing result markers is explicitly permitted for this slice. Do not present it as completion of UPG-001's stronger execution-evidence design.
Evidence: `.venv/bin/python -m pytest tests/test_bus_mutate_shell.py tests/test_finish_line.py tests/test_pause_bus_opacity.py tests/test_bus_agent_emit.py -q` → 56 passed (17+26+6+7). `.venv/bin/python -m pytest tests/ --ignore=tests/battery -q` → 453 passed (interpreter 3.10.12, import path this worktree). Seams: `_run_mutation` after `execute_with_receipt` / full-mode executor (success tuple only); `_execute_step` after SHELL/CODE/TEST `_run_shell`/`_run_code` (stubs still emit); `_maybe_auto_verify` and FINISH `_run_shell(self.verify_cmd)` for `verify.result` roles `auto_verify` / `verify_cmd`. `_run_shell` return strings unchanged. Diff body omitted (no `diff` / `diff_truncated`). `exit_code` from leading `[exit N]` only; `[BLOCKED]` ⇒ `blocked=True` and `exit_code` null, never 0. `ok` true only when `exit_code==0`. auto_verify py_compile success does not set `mission.end` `verified` (synthesized + passing stub stays `needs_review` / `finish_basis=synthesized`). FINISH basis assignment unmoved. Not UPG-001. Battery not re-run. Not committed.

## Commit 5 — TUI layout and event-driven state

- [x] Keep `cosmic_cli/ui.py` entry/import compatibility using DirectivesUI or a PilotApp alias; extract a tui package only when useful.
- [x] IdentityBar shows real version/model/Helix flag/status; unknown health is not painted healthy.
- [x] Left DataTable#mission_table: STATUS, STEPS, DIRECTIVE, BASIS.
- [x] Center RichLog#step_tape.
- [x] Right instruments: compass counts, pending list, session meta including cwd/verify_cmd/mode.
- [x] DirectiveBar retains Input, Deploy button, and Enter submission.
- [x] Collapsible DiffPeek starts hidden.
- [x] Reuse theme tokens and APIKeyScreen.
- [x] Add BoardState/Mission dataclasses and a reducer that has no UI or I/O side effects. Define whether it mutates state or returns a new state consistently.
- [x] Subscribe before execution starts; cross from worker thread using call_from_thread.
- [x] Prefer incremental bus updates; keep only a slow status/step reconciliation fallback.
- [x] Minimum bindings: ctrl+k, q, Enter, p, contextual y/n, D, and mission row selection.
- [x] Guard callbacks after unmount and unsubscribe when appropriate.
- [x] Preserve two concurrent directives and selected-mission tape isolation.

Acceptance: stable basic layout at 120×40 and 100×30; two missions can run and be selected independently; shutdown with a late event does not crash. Compact-tab polish remains deferred.
Evidence: `.venv/bin/python -c 'import cosmic_cli, pathlib; print(pathlib.Path(cosmic_cli.__file__).resolve())'` → this worktree `cosmic_cli/__init__.py`. Interpreter 3.10.12. `.venv/bin/python -m pytest tests/test_cosmic_cli.py tests/test_board_state.py tests/test_pilot_board.py -q` → 45 passed (23+12+10). `.venv/bin/python -m pytest tests/ --ignore=tests/battery -o addopts=` → 436 passed in 13.93s. `DirectivesUI = PilotApp`; APIKeyScreen unchanged. Bus subscribe happens before `agent.run()`; `_on_bus_event` uses `call_from_thread(self.apply_bus_event, event)`; `_detach_bus` on unmount. `floor_ok` stays `None` → IdentityBar paints `floor:unknown`, never `floor:ok`. PauseApproveScreen / compact tabs / palette deferred. Reducer still ignores `helix`/`root` on `mission.start` (UI absorbs those fields; not a reducer edit).

## Commit 6 — Operator PAUSE modal

- [ ] Add PauseApproveScreen with action summary, reason/rule, L2/token-opacity hint, APPROVE, DECLINE, and Esc.
- [ ] Notify on mint; optionally open the modal for a selected mission or single mission.
- [ ] Extract/reuse a shared helper behind CLI accept-pause and TUI approval; preserve existing TTY/ranking checks and exact action binding.
- [ ] UI does not implement ad hoc credential-file I/O or pass tokens into widgets, notifications, or agent context.
- [ ] Decline leaves the mission blocked without remint/retry thrash.
- [ ] Test the modal with Textual Pilot where practical; at minimum test the shared approval helper and token opacity.
- [ ] Demonstrate correct selection under two concurrent pending actions; never approve whichever global token happened to be minted last.

Acceptance: approve stages/claims the selected action under the existing contract; decline stops visibly; CLI approval still works. Approval must be functional, not just a painted resolved row. Preserve current headless BLOCKED return; explicitly describe how an operator initiates any approved rerun/retry.

## Commit 7 — Dashboard normalization and documentation

- [x] Normalize legacy and namespaced lifecycle/step events in dashboard readers.
- [x] Prefer step.proposed.head; fall back to legacy action.
- [x] Read basis from end or mission.end; old records without v remain readable.
- [x] Avoid duplicate steps/end rollups from compatibility aliases.
- [x] Document the implemented Phase-1 event subset and compatibility policy in `docs/MISSION_BUS_v1.md` or the README, linking the broader draft specification.
- [x] Document TUI PAUSE bindings in COSMIC.md or a pilot note.
- [ ] Manually verify dashboard `/api/state` using an isolated fixture and `dashboard --no-open`.

Acceptance: isolated JSONL/echo readers; `complete` stays `complete`; no live echo/chronicle reads.
Evidence: `.venv/bin/python -m pytest tests/test_dashboard_normalize.py tests/test_finish_line.py tests/test_board_state.py -q` → 50 passed (12+26+12). `.venv/bin/python -m pytest tests/ --ignore=tests/battery -q` → 425 passed (interpreter 3.10.12, import path this worktree). `session_step_rows` / `session_terminal` / `mission_counts` use `iter_canonical` + `normalize_legacy` + `is_compat_alias`; dual-write aliases collapse to one step; old records without `v` remain readable; finish_basis is never invented. Live `dashboard --no-open` / `/api/state` not invoked (would open operator `chronicle.db`). Echo tiles still status/directive/model/steps; optional `finish_basis` in meta is HTML-escaped.

## Cross-cutting checks

**Safety:** No approval token in bus/log/UI/notifications; free text redacted before emission; sensitive READ/mutation refusals remain intact. Escape dynamic Rich markup. Operator authority comes from the shared control path, not an event field.

**Compatibility:** Echo status chips work; blocked stays exit 4; verified/needs_review retain current exits; write_echo=False and use_helix=False remain supported. No package-version bump until the release target is selected.

**Concurrency:** Worker threads publish; UI changes occur on the UI thread. JSONL has one serialized writer per mission. Prevent event payload mutation from corrupting another sink. Distinct simultaneous missions must have distinct files/sequences without altering session-based approval binding.

**Identity:** Preserve mission/session meanings and token keying. Existing second-resolution mission filenames need explicit collision handling for the required concurrent case; use a compatible unique suffix if necessary, with regression coverage.

**Isolation:** Existing fixture redirects agent echo/session writes and stubs Helix record. It does not prove every RPC/subprocess is isolated. New integration tests must also stub or isolate witness/call and approval stores, and must not read or alter real operator credentials.

## Suggested test matrix

| Test | Assertion |
| --- | --- |
| test_finish_line.py | Existing decision logic remains green |
| Finish-line persistence tests | End and echo agree on status/basis |
| test_bus_schema.py | Envelope, enums, finish combinations, redaction |
| test_bus_agent_emit.py | Scripted execution emits ordered start/step/end |
| test_pause_bus_opacity.py | Serialized mint/resolution data contain no credential bodies |
| test_dashboard_normalize.py | Legacy and v1 both parse without duplicates |
| Textual interaction coverage | Selection, worker updates, modal y/n, shutdown, concurrent missions |
| Manual headless smoke | v1 session output and unchanged CLI exits |
| Manual TUI | Deploy → live steps → forced PAUSE → decline → blocked |
| Separate approval scenario | Shared CLI and board approval preserve one-use behavior |
| Manual dashboard | Old/new sessions and chronicle views load |

Run `pytest tests/ -q`, plus focused finish-line/bus tests while developing. Report actual commands, interpreter, counts, and outcomes. Existing red-team results distinguish gate-contract residuals from kernel-floor enforcement; do not change classification or assertions merely to turn the battery green.

## PR description template

Use after implementation, replacing placeholders with measured results and final scope:

```markdown
## Summary
Phase-1 avionics adds MissionBus v1 from StargazerAgent, a multi-pane pilot TUI
subscriber, and operator-only PAUSE approve/decline. Headless do, echo, and
Mission Control session readers retain compatibility.

## Why
The existing TUI relies on log strings and a table. Structured events expose
finish_basis, step activity, and pending PAUSE actions in the mission board.

## Included
- events.py / bus.py and validation tests
- Compatible mission.start, step.proposed, finish.declared, mission.end records
- PAUSE mint/resolution events without credentials
- Minimal fs.mutate / shell.exec / verify.result
- Pilot grid and PauseApproveScreen
- Dashboard legacy normalization

## Deferred
Palette, Helix drawer, socket bus, dashboard redesign, compact tabs, undo,
cancellation, and full tool-event coverage.

## Validation
- [ ] pytest tests/ -q — insert result
- [ ] Focused finish-line and bus tests — insert result
- [ ] Headless do smoke — insert result
- [ ] TUI PAUSE approval and decline — insert result
- [ ] Dashboard old/new session check — insert result
```

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Dashboard rejects new names | Compatibility writes/readers plus normalization tests |
| Duplicate timestamps/sequences | _emit owns envelope; raw append does not wrap it again |
| UI-thread races / late shutdown events | UI-thread dispatch, unsubscribe, guarded teardown |
| PAUSE remint/retry loop | Preserve BLOCKED return; one unresolved mint per bound action |
| Token exposure | Opacity tests; shared privileged helper never gives credentials to widgets |
| Large diffs inflate JSONL | Omit initially or enforce cap |
| Concurrent approvals choose wrong action | Bind pending selection to exact action/session before staging |
| Green-looking unverified run | Shared finish enums and status/basis rendering tests |

## Ship gate

- [ ] Local suite and required CI results recorded accurately.
- [ ] Headless smoke produces v1 session records with correct finish_basis rules.
- [ ] TUI mission tape updates live from bus.
- [ ] Forced PAUSE modal → decline → blocked; separate CLI accept-pause scenario works.
- [ ] Board approval uses the real shared helper with exactly-once evidence.
- [ ] Mission Control still displays old/new missions and steps.
- [ ] Final diff/PR describes the implemented subset and unresolved limitations honestly.

## After merge — Phase-1.1 backlog

- Remove remaining log-parsing refresh.
- Richer compass aggregation from the bus.
- Diff bodies and checkpoint undo binding `u`.
- TUI mission.cancel with defined child/process cleanup.
- Shared query helpers with dashboard state().

Implementation order: bus/events/tests → agent emission → PAUSE/compass → fs/shell/verify → TUI/state → shared approval/modal → dashboard/docs → reviewable PR. Branch publication/PR creation is a later delivery action; the branch-setup task does not itself claim it occurred.
