# Cosmic CLI version upgrade plan

Created: 2026-09-19
Status: Draft overall roadmap; Phase-1 implementation slice specified by Anthony and prepared for a local Grok Build handoff. Further additions remain open.
Current package version: 0.9.5
Target version: TBD after scope and compatibility decisions are settled.

## Purpose and baseline

Improve mission verification, progress detection, resource limits, handoff evidence, and operator mission control through the Cosmic Pilot Board and MissionBus. The overall release scope remains open. Anthony's [Phase-1 PR checklist](PHASE_1_PR_CHECKLIST.md) defines the bounded implementation assignment for a local Grok Build instance; remaining roadmap items are not implicitly included.

The comparison baseline is main commit `7174de997ff9ea1855e42a313587f5376e4cc820`, which merged the finish-line split. It already provides `verified` / `needs_review`, operator-owned `--verify-cmd`, and persisted `finish_basis`. Both finished statuses still exit 0. Those features are existing behavior, not new work in this plan.

The local checkout at plan creation is `165b0849756d410515c7aea360593a74a05853b1` on `fix/test-isolation-live-stores`, one commit beyond that baseline. Confirm the integration base when implementation begins; this plan does not assume that local commit has merged to main.

This plan complements [EXECUTION_PLAN.md](EXECUTION_PLAN.md). Its existing product boundaries remain the starting point: one Stargazer path, existing gates and kernel floor, and the existing read-only dashboard. Historical roadmap statements and test counts must be reconciled with the eventual release commit rather than copied forward as current evidence.

Anthony's 2026-09-19 Pilot Board addition explicitly extends the earlier execution plan's restriction on new UI. The native Textual board and operator-only PAUSE modal are now requested upgrade scope. The HTTP dashboard remains a read-only sibling. No additional permission question is needed to record this scope change; implementation remains on standby as instructed.

Update: Anthony subsequently requested a Grok Build implementation branch. Phase-1 preparation now proceeds under that instruction. Branch/worktree details and the implementing agent's assignment are in [GROK_BUILD_HANDOFF.md](GROK_BUILD_HANDOFF.md). The broader specifications remain design references; the Phase-1 checklist takes precedence for this PR's deliverables and explicit deferrals.

## Proposed additions

UPG-001 through UPG-004 were proposed by Codex following a source comparison with MartinLoop. UPG-005 and UPG-006 are Anthony's detailed additions. Priority is provisional and can change as additions arrive.

| ID | Proposal | Suggested order | Status |
| --- | --- | --- | --- |
| UPG-001 | Structured verifier evidence | 1 | Proposed |
| UPG-002 | Detect stalled progress across different actions | 2 | Proposed |
| UPG-003 | Mission-wide token, spending, and time limits | 3 | Proposed |
| UPG-004 | Machine-readable mission handoff | 4 | Proposed |
| UPG-005 | Cosmic Pilot Board: evolution of DirectivesUI | Parallel layout work; event-driven wiring after UPG-006 | User-specified; awaiting implementation |
| UPG-006 | MissionBus: typed events and derived persistence | Foundation for UPG-005; coordinate with UPG-001 | User-specified; awaiting implementation |

**First PR scope:** UPG-005/006 Phase-1 subset only, as bounded by [PHASE_1_PR_CHECKLIST.md](PHASE_1_PR_CHECKLIST.md). Full structured verifier execution (UPG-001), progress detection (UPG-002), budgets (UPG-003), and consolidated handoff (UPG-004) remain later slices. Phase-1 may emit verifier events from current result strings while preserving current FINISH decisions; it must not claim the deeper verifier refactor is complete.

### UPG-001 — Structured verifier evidence

**Problem:** The finish path currently interprets shell text beginning with `[exit 0]` as verifier success. A structured execution result would make the verdict and its supporting evidence explicit.

**Proposed work:**

- Introduce a structured shell execution result with launch/completion state, exit code, timeout/crash/block reason, elapsed time, and redacted output.
- Bind verifier evidence to the mission/run, workspace, working directory, and operator-supplied command.
- Persist a versioned verifier receipt with the mission result and expose a compact summary through existing output paths.
- Keep execution on the existing gated shell path, including compass and sandbox enforcement; preserve text rendering for consumers that need it.

**Acceptance:** A verdict requires an actually launched, completed, exit-zero verifier associated with the current run. Blocked, timed-out, crashed, missing, or mismatched evidence cannot produce `verified`. Printed success markers cannot substitute for process status. Synthesized finishes remain unverified. Tests cover these distinctions and redaction through persistence.

**Decision still open:** Receipt schema and backward-compatible transition from string-returning shell helpers. Record what state was checked; determine whether a workspace fingerprint is needed to detect evidence invalidated by later mutations.

### UPG-002 — Detect stalled progress across different actions

**Problem:** Consecutive-action repetition checks can miss alternating reads or edits that repeatedly return to the same failing state.

**Proposed work:**

- Maintain a bounded window of meaningful observations: relevant content hashes, verifier outcomes, and newly acquired discovery evidence.
- Detect unchanged state and repeated cycles across different actions.
- Record a machine-readable stop reason and the evidence that triggered it.
- Favor concrete state over similarity between model-written summaries.

**Acceptance:** Alternating no-progress loops terminate within the configured window. Useful read-only discovery and changing verifier outcomes count as progress. Timestamps, spending counters, and rewritten prose alone do not reset the detector. A stalled run cannot become verified because the harness stopped it.

**Decisions still open:** Window size, thresholds, discovery-progress definition, and whether the result uses a distinct status or an additional reason attached to an existing status. Avoid prematurely stopping a legitimate long check or a large read-only investigation.

### UPG-003 — Mission-wide token, spending, and time limits

**Problem:** Step limits and per-call timeouts do not express cumulative resource limits for a mission.

**Proposed work:**

- Add operator-configured token and elapsed-time limits first; proposed CLI flags are `--max-tokens` and `--max-duration`.
- Account for all model calls in the mission, including auxiliary calls, and include usage and stop reasons in receipts.
- Check remaining allowances before launching more work; bound an in-flight call or subprocess by the remaining deadline where supported.
- Add optional dollar-budget preflight after usage accounting is trustworthy. Persist pricing/model provenance and distinguish provider-reported usage from estimates.

**Acceptance:** Exhausted limits prevent another call, interrupted runs retain their evidence, and missing usage data is explicitly represented. Time limits cover in-flight work rather than only checks between steps. Tests exercise threshold boundaries and missing/late usage. An estimated next-call cost is never advertised as an absolute spending guarantee.

**Decisions still open:** Defaults, units, review-seat accounting, stop/exit semantics, handling unknown usage, provider-supported output caps, and whether dollar budgets belong in the first release slice.

### UPG-004 — Machine-readable mission handoff

**Problem:** A subsequent seat currently has to assemble verification, changed files, remaining work, and recovery information from separate records.

**Proposed work:**

- Define a versioned handoff record containing mission identity, outcome and basis, changed files, verifier evidence, unresolved work, stop reason, and suggested next action.
- Attach checkpoint references and explicit recovery states such as available, attempted, restored, failed, or unavailable.
- Expose the record through the CLI and Helix using existing persistence facilities.
- Reuse Cosmic's content-hashed checkpoint/rollback implementation. Keep mission-file recovery distinct from installed-version rollback.

**Acceptance:** Consumers can distinguish verified work, unchecked work, and interrupted work without parsing prose or relying solely on exit 0. Recovery is marked restored only after successful restore verification. Older receipts remain readable with absent evidence represented as unknown/not recorded. Secret redaction applies to the complete exported record.

**Decisions still open:** Command/output shape, required versus optional fields, migration handling, and which recovery actions remain explicit operator actions. This proposal does not make failed verification trigger automatic rollback.

### UPG-005 — Cosmic Pilot Board

Source: Anthony's complete Textual layout and widget-tree addition, 2026-09-19. Detailed specification: [PILOT_BOARD_SPEC.md](PILOT_BOARD_SPEC.md).

One full-screen board replaces DirectivesUI with IdentityBar, MissionRail, StepTape, InstrumentStack, collapsible DiffPeek, DirectiveBar, and context-sensitive StatusFooter. Reuse the existing theme, API-key modal, runner, checkpoint system, review path, and operator approval path. Add PAUSE and debrief modals; command palette is optional P2 and Helix recall is P3.

Acceptance includes multi-mission selection, honest status/basis presentation, keyboard-only operation, stable 120×40 and 100×30 layouts, 80-column tab access to every pane, safe operator-only approval, and checkpoint-backed undo. The board calls the same mission runner as `do`; the web dashboard gains no approval controls.

### UPG-006 — MissionBus event schema

Source: Anthony's MissionBus addition, 2026-09-19. Detailed specification: [MISSION_BUS_SPEC.md](MISSION_BUS_SPEC.md).

Introduce a process-local typed event stream with a versioned JSONL envelope (`v`, `event`, `ts`, `session`, `mission`, `seq`). Derive session logs, echo, UI state, and eventually Helix receipt tags from a single emission path. Preserve legacy-reader support and finish-line invariants. Redact before publication or persistence and keep approval credentials off the bus.

Smallest useful cut: additive envelopes, enriched mission end, finish declaration, LocalMissionBus subscription, structured step proposals with legacy compatibility, and opaque PAUSE lifecycle events. Tool, mutation, verifier, review, and instrument events follow. UPG-001 supplies authoritative execution evidence to `shell.exec` and `verify.result`; UPG-004 consumes the same stream for handoff records.

## Integration decisions carried forward

These are implementation reconciliation notes, not permission requests or changes to the requested product intent. Full details live in the two specifications.

- Example `max_steps: 20` must use the existing shared default (currently 30); sample version/model/commit strings are display examples, not constants.
- Legacy event-name compatibility needs reader normalization and duplicate handling; renaming `start` to `mission.start` is not additive by itself.
- Resolve post-run review/undo events versus the proposed rule that `mission.end` is the last non-log event. Keep execution terminality and later operator activity unambiguous.
- Use the actual operator staging/claim/consume path for PAUSE, including correct mission/action binding and one retry; an event saying `by: operator` does not grant authority.
- Keep compact-mode tabs and keyboard mappings consistent; supplied sketches assign `m` and `s` both navigation and filtering roles.
- Define cancellation, PAUSE resume, concurrency, and checkpoint-undo behavior before wiring controls. A multi-mission display does not by itself choose concurrent execution.
- Decide subscriber isolation, durable-write failure reporting, sequence allocation, replay deduplication, and unique mission identifiers. Redaction precedes every sink; verification status comes from execution evidence.

**Phase-1 clarifications from Anthony's checklist:** Preserve two concurrent directives. Defer cancellation, undo binding, compact-mode polish, full tool-event coverage, and palette/recall features. Maintain current headless blocked termination with no automatic retry; board approval uses existing one-retry staging/claim semantics. Session identity and approval keying remain compatible. Mission filename collision handling and selected-action approval must not conflate concurrent missions. These refine the broader notes above.

## Integration and release outline

1. Collect Anthony's additions and resolve overlap among the proposals and detailed specifications.
2. Freeze the first release scope, choose its version, and settle compatibility decisions, including `needs_review` exit semantics.
3. Reconcile main, the local test-isolation change, existing CI coverage, and the historical execution plan.
4. Establish the minimal MissionBus contract and structured verifier evidence together. Build the board layout incrementally, then connect state, steps, PAUSE, diffs/undo, debrief, bindings, and compact mode in the board's specified order. Add progress detection, resource limits, and consolidated handoff in agreed slices; palette P2 and recall P3 retain their supplied priorities.
5. Validate behavior with isolated test stores, event-schema and legacy-reader tests, Textual interaction/layout tests, and applicable gate/battery checks. Include a controlled live-model mission for changed finish behavior and identify its limits explicitly. Exercise CLI and board approval paths without exposing credentials.
6. Refresh README, CLI help, receipt-schema documentation, version metadata, and release evidence. Publish measured battery counts with their tested revision; distinguish gate-contract results from kernel-floor verification.

No release date or implementation estimate is assigned while additions are still being collected.

## Additions inbox

Awaiting Anthony's next additions. Assign subsequent feature items `UPG-007` onward. The Phase-1 checklist is the delivery breakdown of UPG-005/006, not a seventh feature.

| ID | Addition | Source | Status / decisions |
| --- | --- | --- | --- |
| UPG-005 | Cosmic Pilot Board layout, widget tree, state, modals, bindings, migration | Anthony, 2026-09-19 | Recorded in linked specification; implementation on standby |
| UPG-006 | MissionBus schema, event catalog, projections, invariants, migration | Anthony, 2026-09-19 | Recorded in linked specification; implementation on standby |
| UPG-005/006 Phase-1 | Seven-commit PR checklist and ship gate | Anthony, 2026-09-19 | Implementation assignment prepared for Grok Build |
| UPG-007 onward | Awaiting further additions | Anthony | Open |

## Source notes

MartinLoop reference revision: `f7fdf993d7680835e6a4caedf0fdf7b1a2e1c7cf`, inspected 2026-09-19. These are architectural inspirations to adapt to Cosmic; this plan adds no dependency or copied implementation.

- [Verifier evidence and handoff](https://github.com/Keesan12/martin-loop/blob/f7fdf993d7680835e6a4caedf0fdf7b1a2e1c7cf/packages/core/src/verified-handoff.ts)
- [Exit policy and progress-state hashing](https://github.com/Keesan12/martin-loop/blob/f7fdf993d7680835e6a4caedf0fdf7b1a2e1c7cf/packages/core/src/exits.ts)
- [Trajectory detection](https://github.com/Keesan12/martin-loop/blob/f7fdf993d7680835e6a4caedf0fdf7b1a2e1c7cf/packages/core/src/trajectory.ts)
- [Budget policy](https://github.com/Keesan12/martin-loop/blob/f7fdf993d7680835e6a4caedf0fdf7b1a2e1c7cf/packages/core/src/policy.ts)
- [Rollback evidence](https://github.com/Keesan12/martin-loop/blob/f7fdf993d7680835e6a4caedf0fdf7b1a2e1c7cf/docs/concepts/rollback-evidence.md)
