# Grok Build handoff — Pilot Board / MissionBus Phase-1

Prepared for Anthony on 2026-09-19. This is a coding assignment for a separately launched local Grok Build instance. The preparation task creates the branch and commits planning documents; it does not launch Grok, implement the runtime changes, or publish the branch.

## Checkout

- Branch: `grok/pilot-board-phase1`
- Dedicated worktree: `/Users/vaquez/cosmic-cli-worktrees/grok-pilot-board-phase1`
- Repository: `https://github.com/templetwo/cosmic-cli`
- Fetched remote main: `7174de997ff9ea1855e42a313587f5376e4cc820`
- Implementation base: `165b0849756d410515c7aea360593a74a05853b1`

The base adds the existing `test: isolate the suite from the operator's live stores` commit above main. It changes only tests/conftest.py and tests/test_isolation.py, redirecting agent session/echo output and stubbing Helix record. Keep this prerequisite visible in review; it was not authored as part of this planning task. It may be merged independently before the feature PR.

The original checkout stays on `fix/test-isolation-live-stores`. Do implementation work in this dedicated worktree, not the original checkout. The unrelated untracked `docs/COSMIC_RECONCILE_STATUS_2026-07-30.md` is not part of the handoff.

## Assignment and reading order

Implement one mergeable Phase-1 slice: typed MissionBus v1, compatibility session/echo writes, an event-driven three-pane Textual Pilot Board, and operator-only PAUSE approve/decline through the existing authority path. Preserve headless do and read-only Mission Control behavior.

Read in this order:

1. [PHASE_1_PR_CHECKLIST.md](PHASE_1_PR_CHECKLIST.md): authoritative PR scope, seven implementation commits, acceptance and ship gate.
2. [MISSION_BUS_SPEC.md](MISSION_BUS_SPEC.md): detailed event contract, projection and compatibility rules, known reconciliation decisions.
3. [PILOT_BOARD_SPEC.md](PILOT_BOARD_SPEC.md): widget/layout/state design; implement only the Phase-1 subset.
4. [VERSION_UPGRADE_PLAN.md](VERSION_UPGRADE_PLAN.md): broader roadmap and dependencies, not blanket authorization to implement every item in this PR.
5. COSMIC.md, existing source/test seams listed in the Phase-1 preflight, and applicable repository instructions.

Anthony's explicit board request supersedes the older EXECUTION_PLAN.md restriction on new native UI. The web dashboard remains read-only. No need to ask again whether the requested board belongs in scope.

## Execution boundaries

- Follow the seven-commit order where dependencies allow; keep each change reviewable and add evidence to the checklist as completed.
- Keep DirectivesUI/import compatibility or provide an alias. Reuse APIKeyScreen, theme, Stargazer runner, gateway, and existing approval code.
- Do not change FINISH decision logic, CLI exits, compass rules, or kernel-floor behavior in this slice.
- Do not implement budgets, a new progress detector, full typed verifier execution, palette, Helix drawer, compact-tab polish, undo, cancellation, or a dashboard redesign as incidental work.
- Two concurrent directives must remain supported. Test mission id/file uniqueness and event/approval isolation. Session approval keying must remain compatible.
- Keep the existing headless BLOCKED return. Operator approval stages/claims one retry under existing semantics; document the explicit rerun/retry workflow instead of adding an automatic loop.
- A global last-token file alone is not enough to identify the selected pending action under concurrency. Inspect the actual gate/Helix/local-policy channels and verify binding before extracting a shared helper. Never expose credentials to events or widgets.
- Migration choices are allowed within the checklist: dual-write with deduplication, or compatible readers that accept both. Preserve working intermediate commits when practical.
- Treat schematic snippets as design inputs: reserved envelope fields cannot be overridden, unknown health is not green, dynamic Rich markup must be escaped, and post-end review/undo must not violate the defined lifecycle.

## Environment and validation

Python requires 3.10+. Use an isolated virtual environment in this worktree and install the project's test extras if needed. The existing original-checkout venv may be useful for inspecting installed dependencies, but its editable install can point to the wrong source tree. Confirm `cosmic_cli.__file__` resolves inside this worktree before relying on any test run or smoke result.

An example setup, to run when implementation begins:

```sh
cd /Users/vaquez/cosmic-cli-worktrees/grok-pilot-board-phase1
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -c 'import cosmic_cli; print(cosmic_cli.__file__)'
```

Inspect the available interpreter/version first; the commands above are a setup outline, not a claim the environment already exists. The branch-preparation task did not install dependencies or run the code suite.

Run targeted tests during each slice, then the requested full suite and meaningful interaction checks:

```sh
.venv/bin/python -m pytest tests/test_finish_line.py tests/test_bus_schema.py -q
.venv/bin/python -m pytest tests/ -q
```

Add scripted bus tests, dashboard legacy/v1 normalization tests, opacity/claim tests, and Textual interaction coverage as specified in the checklist. Test without real model calls where deterministic stubs suffice. Keep API/model-backed smoke results separate from scripted evidence and report unavailable live checks honestly.

The inherited fixture protects agent echo/session writes and Helix record, not every RPC. Ensure new witness/call, approval, dashboard, and subprocess integration tests use isolated stores/mocks. Do not use a real operator token as a test fixture. Existing live stores and the unrelated reconciliation document are not cleanup targets.

Known baseline claims, not freshly rerun by the preparing agent:

- `165b084` commit records 364 non-battery tests passed.
- That commit records 228 battery tests passed with 5/185 successful attack probes (2.7%), in the known gate-layer residual class.
- Required remote CI at fetched main covers token stateful/build-identity checks, not the whole requested Phase-1 suite.

Do not change expected residuals or claim a clean security result by suppressing probes. Record exact commands, outcomes, and limitations at the final implementation revision.

## Delivery

Leave a reviewable implementation on this branch with small commits, the checklist reflecting actual work, and a PR description using the supplied template. Report final commit, changed behavior, test evidence, and unresolved issues. Do not merge, tag, deploy, change the package version, or publish a release as part of this implementation slice. Prepare PR content; branch push/PR publication can follow Anthony's delivery instruction.

## Prompt to paste into the local Grok Build session

> Work in `/Users/vaquez/cosmic-cli-worktrees/grok-pilot-board-phase1` on branch `grok/pilot-board-phase1`. Read `docs/GROK_BUILD_HANDOFF.md`, then implement `docs/PHASE_1_PR_CHECKLIST.md` in its seven reviewable slices. The MissionBus and Pilot Board specs are supporting designs; the Phase-1 checklist governs scope. Preserve headless CLI behavior, operator-only single-use PAUSE authority, token opacity, two concurrent directives, and legacy dashboard/echo compatibility. Use isolated tests and verify imports resolve to this worktree. Commit completed work and report evidence and residuals. Do not implement the later roadmap or merge, tag, deploy, or release. Begin with the preflight source reads and proceed with implementation.
