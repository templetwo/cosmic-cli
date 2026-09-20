# Cosmic Pilot Board — Textual implementation blueprint

Source: Anthony's addition, 2026-09-19. Tracking: UPG-005 in [VERSION_UPGRADE_PLAN.md](VERSION_UPGRADE_PLAN.md).
Status: Recorded overall design. [PHASE_1_PR_CHECKLIST.md](PHASE_1_PR_CHECKLIST.md) now defines the subset assigned to Grok Build; it takes precedence for initial delivery. Remaining features await later slices.

This specification preserves the requested layout, widget relationships, behavior, rollout, and non-goals. Code sketches are design inputs, not validated Textual implementation. Reconciliation notes are explicitly separated at the end.

## Screen model

```text
PilotApp(App)                    replaces DirectivesUI
├── APIKeyScreen                 existing modal; preserve behavior
├── PauseApproveScreen           new, operator L2 only, P0
├── DebriefScreen                new, post-FINISH
├── CommandPaletteScreen         optional P2
└── HelixRecallScreen            drawer-style modal, P3
```

One full-screen board. Key, PAUSE, and debrief interactions use modals. Instruments and mission control remain the primary surface; no nested chat seats.

## Layout and widget tree

```text
┌──────────────────────────── IdentityBar ──────────────────────────────┐
│ ✦ COSMIC version·commit · model · helix:on · floor:ok · goal: …       │
├──────────────────┬───────────────────────────┬────────────────────────┤
│ MissionRail      │ StepColumn                │ InstrumentStack        │
│ ~28ch            │ 1fr                       │ ~36ch                  │
│ mission list     │ structured StepTape       │ CompassPulse           │
│ status chips     │ tool / path / class       │ PendingGates           │
│ step bars        │ selected mission          │ ReviewDock             │
│                  │                           │ SessionMeta            │
├──────────────────┴───────────────────────────┴────────────────────────┤
│ DiffPeek: collapsible, 0 | auto | ~12 rows, selected mutation         │
├──────────────────────────────────────────────────────────────────────┤
│ DirectiveBar: [directive input…] [▸ DEPLOY] [Review] [Verify…]         │
├──────────────────────────────────────────────────────────────────────┤
│ StatusFooter: bindings · PAUSE count · last receipt · finish_basis   │
└──────────────────────────────────────────────────────────────────────┘
```

Minimum useful target: 100×28. Acceptance layouts: 120×40 and 100×30. Below roughly 90 columns, provide tabs for missions, instruments, and steps; 80 columns remains usable. IdentityBar, DirectiveBar, and footer stay visible.

```text
PilotApp
├── IdentityBar(Static) #identity
│   reactive: version, commit, model, helix_state, floor_state, goal
├── MainGrid(Horizontal) #main
│   ├── MissionRail(Vertical) #missions .rail
│   │   ├── Static("MISSIONS")
│   │   ├── MissionFilter(Horizontal): all | run | review | block
│   │   ├── MissionTable(DataTable) #mission_table
│   │   │   columns: STATUS · STEPS · DIRECTIVE · BASIS
│   │   └── MissionActions(Horizontal): Cancel · Re-run · Logs
│   ├── StepColumn(Vertical) #steps .column
│   │   ├── StepHeader(Static) #step_header
│   │   ├── StepTape(RichLog) #step_tape, markup=True
│   │   └── ActionLegend(Static), optional OPEN/PAUSE/WITNESS glyphs
│   └── InstrumentStack(Vertical) #instruments .rail
│       ├── CompassPulse(Static | DataTable) #compass_pulse
│       ├── PendingGates(DataTable) #pending
│       ├── ReviewDock(Static | RichLog) #review_dock
│       └── SessionMeta(Static) #session_meta
├── DiffPeek(Vertical) #diff_peek .-hidden
│   ├── DiffHeader(Static) #diff_header
│   └── DiffBody(RichLog | Static) #diff_body
├── DirectiveBar(Horizontal) #directive_bar
│   ├── DirectiveInput(Input) #directive_input
│   ├── DeployButton(Button) #deploy_btn
│   ├── ReviewToggle(Button) #review_btn
│   └── VerifyButton(Button) #verify_btn
└── StatusFooter(Footer | Static + bindings)
```

Reuse APIKeyScreen, theme colors/status_markup/step_bar/log formatting, and existing DataTable/RichLog patterns.

## Textual CSS sketch

The supplied CSS uses rail widths 32/38, while the diagram suggests approximately 28/36; tune at the target sizes rather than treating either pair as fixed requirements.

```python
CSS = f"""
Screen {{ background: {theme.PAGE}; color: {theme.TEXT}; }}
#identity {{ height: 1; background: {theme.SURFACE}; color: {theme.MUTED}; padding: 0 1; }}
#identity .accent {{ color: {theme.CYAN}; text-style: bold; }}
#identity .good {{ color: {theme.GOOD}; }}
#identity .warn {{ color: {theme.WARN}; }}
#identity .crit {{ color: {theme.CRIT}; }}
#main {{ height: 1fr; }}
.rail {{ width: 32; background: {theme.PAGE}; border-right: tall {theme.BORDER}; }}
#instruments {{ border-right: none; border-left: tall {theme.BORDER}; width: 38; }}
.column {{ width: 1fr; }}
DataTable {{ background: {theme.PAGE}; }}
DataTable > .datatable--header {{ color: {theme.MUTED}; text-style: bold; }}
DataTable > .datatable--cursor {{ background: {theme.CURSOR}; }}
#step_tape {{ height: 1fr; background: {theme.PANEL}; border: round {theme.BORDER}; margin: 0 1; }}
#diff_peek {{ height: 12; background: {theme.PANEL}; border-top: tall {theme.BORDER}; padding: 0 1; }}
#diff_peek.-hidden {{ display: none; }}
#directive_bar {{ height: auto; padding: 0 1 1 1; background: {theme.SURFACE}; }}
#directive_input {{ width: 1fr; border: tall {theme.BORDER}; }}
#directive_input:focus {{ border: tall {theme.CYAN}; }}
#deploy_btn {{ background: {theme.BLUE}; color: #fff; text-style: bold; min-width: 12; }}
Footer {{ background: {theme.SURFACE}; }}
"""
```

Resize hook toggles `compact` below 90 columns. Supplied compact selectors hide `#instruments` and expand `#missions`; the implementation must also provide the tab switcher that makes the hidden pane accessible, and put the class on the object targeted by `Screen.compact`.

## Single state model

Use dataclasses and reactive projection from one BoardState. Retain raw logs only as a migration aid.

| Type | Fields |
| --- | --- |
| MissionStatus | ready, running, verified, needs_review, blocked, error, max_steps; include passed from the canonical MissionBus enum; legacy complete is read-only paint |
| StepEvent | ts (UI arrival HH:MM:SS), kind, summary, optional path, compass, receipt_id, detail |
| Mission | key, directive, status=ready, model, steps_taken=0, max_steps, optional finish_basis, review_mode=False, optional verify_cmd, logs=[], steps=[], last_mutation_path, last_checkpoint, last_diff, agent |
| BoardState | missions={}, selected_key, filter=all, pending_pauses=[], compass_today={}, compass_total={}, goal, helix_on=True, floor_ok=None, review_report, verify_cmd_default, review_default=False |

Mission keys must be stable and unique across reruns. Use the shared max-step default; the supplied `20` is an example. Event timestamps retain UTC in persistence; UI arrival stamps are presentation only.

Migration boundary:

```python
def on_mission_event(mission_key, event):
    self.call_from_thread(self.apply_event, mission_key, event)
```

Keep `ui_callback=self.thread_safe_refresh` as a thin adapter until typed events replace log parsing. Target the [MissionBus contract](MISSION_BUS_SPEC.md); avoid separate competing event schemas for agent and UI.

## Pane rendering

- MissionTable: `theme.status_markup(status)`, `theme.step_bar(taken, max, status)`, truncated directive, and finish_basis or `—`. Selection updates StepHeader, StepTape, DiffPeek, ReviewDock, and SessionMeta.
- StepTape: timestamp, action/class, path/command, checkpoint or rule, and terminal status/basis. OPEN uses GOOD, PAUSE WARN, WITNESS CRIT.
- CompassPulse: today and all-time counts with Mission Control semantics.
- PendingGates: age, tool, summary, and approval focus target.
- ReviewDock: empty, active spinner, verdict bullets, or needs-review prompt.
- SessionMeta: session id, cwd, execution mode, model, verify command, Helix root.
- DiffPeek: selected mutation's path, checkpoint id, diff or CREATE preview, optional diff stats, and undo hint. Visible with last_diff or `D`.

```text
12:04:01  ● READ    cosmic_cli/gateway.py
12:04:02  ● EDIT    cosmic_cli/gateway.py       checkpoint:ck_9f3a
12:04:03  ⏸ PAUSE   SHELL rm -rf build/        rule:destructive_rm
12:04:11  ✓ OPEN    SHELL pytest -q
12:04:40  ● FINISH  needs_review (model_declared)
```

## Modals

**PauseApproveScreen (P0):** `ModalScreen[bool]`, title “⏸ PAUSE — operator approval”, action summary, reason/rule, hint “token never shown to the model · TTY L2 only”, APPROVE and DECLINE buttons. Approval invokes the existing `helix accept-pause` authority path for one approved retry. Never render the raw token. Decline resolves the pending row and leaves the mission blocked or follows its explicit PASS path.

**DebriefScreen:** status, finish_basis, what/evidence/residual, session path, copy-Markdown, close. `needs_review` is visibly distinct from verified; green debrief is optional for verified.

**CommandPaletteScreen (optional P2):** fuzzy verbs invoking shared functions also used by Click commands; avoid spawning CLI subprocesses where shared functions suffice.

**HelixRecallScreen (P3):** drawer-style memory lookup.

## Bindings supplied

| Keys | Action |
| --- | --- |
| ctrl+k | API key |
| ctrl+c, q | Quit |
| ? | Help |
| enter, d | Deploy |
| r | Toggle review |
| v | Set verify command |
| c | Cancel selected mission |
| R | Re-run selected mission |
| j, k | Next/previous step or tape scroll |
| p | Focus pending |
| y, n | Approve/decline only for focused approval modal or selected pending item |
| D | Toggle diff |
| u | Undo checkpoint |
| h | Helix recall |
| g | Edit goal |
| s | Focus missions (as supplied in bindings sketch) |
| i | Focus instruments |
| m | Cycle mission filter (as supplied in bindings sketch) |
| :, ctrl+p | Command palette |

Footer paints a context-sensitive subset. Resolve the compact-tab sketch's `m missions / i instruments / s steps` conflict before implementation. Single-letter shortcuts must not hijack typing in DirectiveInput; Enter behavior must respect focus and modal controls.

## Composition and data flow

`compose()` yields IdentityBar, a Horizontal main grid containing the three columns above, hidden DiffPeek, DirectiveBar, and Footer. MissionActions includes cancel, rerun, and logs; section labels identify MISSIONS and INSTRUMENTS. Input placeholder: `directive… (: palette · ctrl+k key)`.

1. Deploy creates Mission with review/verifier settings, invokes the shared Stargazer runner, and inserts its rail row.
2. Worker-thread events enter `call_from_thread(apply_event)`; update mission state, append the tape, refresh selected diff, and update compass/pending instruments.
3. PAUSE mints in the operator store; the board shows a pending row/modal. Approval takes the shared operator path and permits exactly one retry; decline resolves visibly.
4. FINISH updates status/basis and opens or offers debrief/review as appropriate.
5. A light 1-second reconciliation timer can refresh steps/status; optional read-only polling of Helix pending confirmations complements events.
6. Dashboard remains a read-only sibling. Share query helpers later; do not make Textual depend on the HTTP server.

## Suggested modules and migration

Start in `ui.py`, stabilize the grid, then extract incrementally:

```text
cosmic_cli/ui.py                  PilotApp entry + compatibility re-exports
cosmic_cli/tui/app.py             PilotApp
cosmic_cli/tui/state.py           BoardState, Mission, StepEvent/reducer
cosmic_cli/tui/format.py          step formatting using theme
cosmic_cli/tui/widgets/
  identity.py missions.py steps.py instruments.py diff_peek.py directive_bar.py
cosmic_cli/tui/screens/
  api_key.py pause.py debrief.py palette.py helix_recall.py
```

## Implementation order and acceptance

| Step | Deliverable | Acceptance |
| --- | --- | --- |
| 1 | Three-pane grid, IdentityBar, DirectiveBar | Stable at 120×40 and 100×30 |
| 2 | MissionTable driven by BoardState | Multiple missions selectable |
| 3 | Structured StepTape, temporary log adapter allowed | Compass class visible when available |
| 4 | PAUSE modal using real operator approval path | No token reaches model context; CLI approval still works |
| 5 | DiffPeek and checkpoint undo | EDIT exposes path and usable undo |
| 6 | finish_basis and DebriefScreen | needs_review never looks verified |
| 7 | Bindings and contextual footer | Keyboard-only deploy → select → diff |
| 8 | Compact tabs | Every pane usable at 80 columns |

## Non-goals and visual hierarchy

No primary multiline chat transcript, raw approval token display, web-dashboard approval buttons, replacement for `cosmic-cli do`, or second toolkit in the hot path. Keep prompt_toolkit file browsing separate; peek using step paths.

Primary focus: DirectiveInput while idle, MissionTable while running. Secondary: StepTape. Tertiary: instruments. Interrupt: PAUSE modal on top. Epilogue: DebriefScreen.

## Reconciliation notes for implementation

- Preserve operator authentication and action binding in the shared approval implementation. The current CLI describes `accept-pause` as staging one approved retry; document exactly where atomic claim and consumption occur instead of assuming the UI itself consumes a token.
- Define cooperative cancellation, child termination, pause/resume state, and rerun identity. Decide whether missions execute serially or concurrently; make selected-mission data and reviews unambiguous either way.
- Checkpoint undo must detect later/concurrent modifications and avoid erasing another mission's or the operator's edits. Report the real restore outcome through MissionBus.
- Escape dynamic Rich markup in logs, diffs, paths, and directives while retaining trusted status styling. Redaction does not itself escape markup.
- Render unavailable/unverified Helix and floor health as unknown/degraded, not green. Static sample identity values are not health evidence.
- Reconcile `passed` and legacy `complete` with the shared enums; resolve post-end review/undo lifecycle in MissionBus.
