# MissionBus v1 — implemented Phase-1 subset

This file describes what the code on this branch actually does. The broader
catalog, open design questions, and later slices live in
[MISSION_BUS_SPEC.md](MISSION_BUS_SPEC.md). When they conflict, this file is
the compatibility policy for Mission Control and other readers.

## Envelope

Canonical events and new session JSONL lines carry:

| Field | Contract |
| --- | --- |
| `v` | Schema major version; currently `1` |
| `event` | Catalog type (namespaced for new writers) |
| `ts` | UTC ISO-8601 timestamp |
| `session` | Shared Helix/seat session id (approval key) |
| `mission` | Unique mission file stem (JSONL identity) |
| `seq` | Monotonic per mission, beginning at 0 |

Reserved envelope keys cannot be overridden by payload. Old records without
`v` / `mission` / `seq` remain readable; readers do not fill them in.

## Events the agent emits

`StargazerAgent._emit` publishes to `LocalMissionBus` (canonical names only)
and appends the same record to the mission JSONL.

| Event | When |
| --- | --- |
| `mission.start` | First event, seq=0. Identity: directive, model, exec_mode, root, helix, verify_cmd, review, max_steps, cosmic_version, cosmic_commit |
| `mission.status` | `status=running` immediately after start |
| `step.proposed` | Each model step: `n` (1-based), `action` (verb), `raw` (redacted), `head` (first line ≤120) |
| `finish.declared` | Immediately before an accepted finish `mission.end`. status, finish_basis, synthesized, redacted text |
| `mission.end` | Last execution event. status, optional finish_basis, steps, edited, warnings, outcome, model; `block_message` when blocked |
| `compass.verdict` | Authoritative PAUSE/WITNESS at the gate; also on `[BLOCKED]` when the class is known. OPEN is not invented. |
| `gate.pause_minted` | After `mint_token`: action_summary, action_sha256, optional expires_at / opaque pending_id. Never a token. |
| `gate.pause_resolved` | `approved`+`by=operator` on successful `claim_once`, or TUI/CLI approve/decline. Failed claim is `invalid` without `by`. |
| `fs.mutate` | After successful EDIT/WRITE/CREATE/MKDIR. Path, optional checkpoint/receipt. No diff body in this slice. |
| `shell.exec` | SHELL/CODE/TEST. `exit_code` only from `[exit N]`; blocked is never 0. |
| `verify.result` | Distinct `role` (`verify_cmd` / `auto_verify`). auto_verify cannot certify a mission. |

`finish.declared` is omitted on blocked / passed / max_steps / error paths.

### Not emitted in this slice

- `gate.receipt`, `fs.read` / `fs.rollback`
- `step.started` / `step.finished`
- `pass.declared` / `review.completed` / `steer` / `mission.cancel`

Approve stages `operator_approval_token` and does not `claim_once`. Consume is the retry. Concurrent pauses require an explicit `action_sha256`; `last_pause_token.json` is not the selector.

### Post-terminal gate decisions

`mission.end` ends execution, not the lifetime of the mission's audit stream.
A blocked run can return before the operator decides. Its later
`gate.pause_resolved` event retains the mission/session envelope and continues
the sequence, correlated with the pending gate by `action_sha256` and optional
`pending_id`. Local gates do not require a `pending_id`.

Readers must continue past `mission.end` to process these decisions. Resolving
a gate clears the pending indicator; it does not change the terminal status,
finish basis, or echo rollup. Approval stages an existing unused credential;
it neither mints a new one nor executes the action. A fresh same-session retry
has its own mission tape and consumes the credential on the execution path.
`gate.pause_resolved` with `decision=approved` alone is not execution evidence.

## Dual-write / compatibility

JSONL is dual-written for one minor version so pre-bus readers keep working.

- Canonical line first (`mission.start`, `step.proposed`, `mission.end`).
- Compatibility alias immediately after: `start`, `step`, `end`.
- Aliases **share the canonical `seq`**. They set `compat: true` and
  `alias_of` to that seq. They do not consume a new seq.
- Aliases are **not published on the bus**. Subscribers see canonical names
  only.
- Legacy `step` aliases keep the redacted expression in `action` (pre-bus
  `review.load_session` keys on `FINISH:` in that field). Canonical
  `step.proposed.action` is the verb only; `raw` / `head` hold the expression.

Readers:

- `iter_canonical` drops aliases (`compat` or `alias_of` on a legacy name)
  then `normalize_legacy` maps `start`/`step`/`end` onto namespaced types.
- `normalize_legacy` never invents `finish_basis`.
- Dashboard `session_step_rows` treats `step.proposed` \| `step` as steps,
  prefers `head` else legacy `action`, and drops start/end/`finish.declared`
  from the step tape. One action is one row.
- Dashboard `session_terminal` reads the last `mission.end` or `end` after
  alias drop.

Unknown event types are ignored by tolerant readers. Strict `validate_event`
is a separate path and is not applied to old files.

## Status and finish_basis

| Category | Values |
| --- | --- |
| Nonterminal | ready, running |
| Finish-path terminal | verified, needs_review |
| Other terminal | blocked, passed, max_steps, error |
| Legacy read-only | complete — never emitted by new writers |

| finish_basis | Status |
| --- | --- |
| verifier | verified |
| model_declared | needs_review |
| synthesized | needs_review |
| verifier_blocked | needs_review |

Rules the writers and readers share:

- `status == "verified"` iff `finish_basis == "verifier"`.
- Finished canonical terminals carry a basis. Non-finish statuses omit the
  key (including blocked / passed / max_steps / error).
- Synthesized FINISH is never `verified`. auto_verify / `py_compile` never
  certifies a mission.
- Legacy records missing the key stay legacy. Normalization does not invent
  verification. `complete` stays `complete` and is counted separately on
  Mission Control; it is never folded into `verified`.

Echo is a derived record (not a bus event). It keeps existing keys and adds
optional `mission` / `ts`. Echo `finish_basis` is present only when the
canonical end carried it.

## Token opacity

Approval credentials never appear on the bus, in JSONL, in echo, or in the
dashboard. Forbidden payload keys include `token`, `token_id`,
`approval_token`, `COSMIC_APPROVAL_TOKEN`, and related names.
`token_id_prefix`, if ever used, is at most 8 characters and must not be a
credential body.

Operator approval remains the existing L2 TTY path
(`cosmic-cli helix accept-pause`). An event field is not authority. The
Pilot Board must not render a token; see COSMIC.md for TUI bindings.

## Mission identity vs session_id

- `session` / `session_id` is the seat/Helix session. Approvals stay keyed
  by `(session_id, action_summary)`.
- `mission` / `mission_id` is the JSONL file stem from `unique_mission_stem`:
  `{session_id}__{UTC microseconds}[_{nonce}]`. Concurrent starts in the same
  second do not share a file. A 2-hex nonce is always applied today.
- Two missions may share a session_id; they must not share a mission stem.

## Mission Control

Read-only HTTP dashboard. `/api/state` still returns status / directive /
model / steps for echo tiles. When `finish_basis` is present on an echo row
it is shown in the tile meta (escaped). `mission_counts` keys off status
strings only (`verified`, `needs_review`, `complete`, `blocked`).

Session helpers used by the dashboard (and by tests without sqlite):

- `session_step_rows` / `session_steps`
- `session_terminal`
- `mission_counts`

`state()` still queries chronicle.db for compass / insights / pending gates.
Those queries are not part of the JSONL normalize path.
