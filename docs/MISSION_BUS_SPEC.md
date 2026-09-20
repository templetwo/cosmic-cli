# MissionBus — event schema and integration blueprint

Source: Anthony's addition, 2026-09-19. Tracking: UPG-006 in [VERSION_UPGRADE_PLAN.md](VERSION_UPGRADE_PLAN.md).
Status: Recorded overall design. [PHASE_1_PR_CHECKLIST.md](PHASE_1_PR_CHECKLIST.md) now defines the subset assigned to Grok Build; it takes precedence for initial delivery. Remaining features await later slices.

The catalog below preserves the supplied contract and examples in a compact reference. Reconciliation notes identify design conflicts to resolve before implementation; they do not silently replace the requested schema.

## Goals and boundaries

One typed agent event stream supplies session JSONL, echo projections, the [Pilot Board](PILOT_BOARD_SPEC.md), and later Helix receipt tags. Events carry honest status and optional finish_basis, redact before emission, and never carry approval credentials. Mission Control can continue reading files and existing Helix sources.

Initial transport is in-process publication plus append-only files. It does not replace Helix tables, promise guaranteed-delivery RPC, or expose token-store contents. Sockets are a possible later transport, not initial scope.

## Envelope v1

Every canonical event and session JSONL line includes:

```json
{
  "v": 1,
  "event": "mission.start",
  "ts": "2026-09-19T21:03:00.123456+00:00",
  "session": "20260919T210300Z",
  "mission": "20260919T210300Z__20260919T210300123Z",
  "seq": 0
}
```

| Field | Contract |
| --- | --- |
| v | Required schema major version; initially 1 |
| event | Required catalog type |
| ts | Required UTC ISO-8601 timestamp |
| session | Shared Helix/seat session when supplied; otherwise minted |
| mission | Unique mission file stem, maps to session_path |
| seq | Monotonic per mission, beginning at 0; authoritative order independent of clock skew |

Current session records already inject `ts` and `session`; `v`, `mission`, and `seq` are additive. Readers ignore unknown fields and normally unknown event types; strict schema validation may reject unknown types. Event-name changes need the migration path below.

`mission.start` carries identity once: directive, model, exec_mode, root, helix, verify_cmd (nullable), review, max_steps, cosmic_version, cosmic_commit. Consumers cache identity instead of repeating it on each event. Use actual runtime identity and the shared step default; sample `grok-4.5`, `0.9.5`, and `20` are illustrative values.

## Status and finish_basis

| Category | Values |
| --- | --- |
| Nonterminal | ready, running |
| Finish-path terminal | verified, needs_review |
| Other terminal | blocked, passed, max_steps, error |
| Legacy read-only | complete; never emitted by new writers |

| finish_basis | Meaning | Status |
| --- | --- | --- |
| verifier | Model declared FINISH and operator verifier exited 0 through gated shell | verified |
| model_declared | Model declared FINISH without a successful verifier | needs_review |
| synthesized | Harness generated FINISH to stop a loop | needs_review |
| verifier_blocked | Verifier gated, declined, or lacking usable exit evidence | needs_review |

For canonical new finish events: `status == "verified"` iff `finish_basis == "verifier"`. Finished terminal events require the appropriate basis. Non-finish statuses omit the key entirely, including ready/running and blocked/passed/max_steps/error. Legacy records missing this evidence remain legacy; never invent verification during normalization.

## Event catalog

The envelope is required for every row below. Payload fields follow the supplied examples; nullable and optional distinctions must be formalized in `validate_event` during implementation. Do not infer successful execution from a display string when structured execution evidence is available.

### Lifecycle

| Event | Payload / behavior |
| --- | --- |
| mission.start | directive, model, exec_mode, root, helix, verify_cmd, review, max_steps, cosmic_version, cosmic_commit; first event, seq=0 |
| mission.status | status, optional/nullable detail; lightweight transition notification |
| mission.end | status, finish_basis only for finish terminals, steps, edited[], warnings[], model, outcome (redacted ≤2000), block_message only when blocked |
| mission.cancel | reason (e.g. operator), status=error in supplied design; final end-record semantics must be settled |

Example end payload, in addition to envelope:

```json
{
  "status": "needs_review",
  "finish_basis": "model_declared",
  "steps": 7,
  "edited": ["cosmic_cli/gateway.py"],
  "warnings": [],
  "model": "grok-4.5",
  "outcome": "redacted summary"
}
```

The supplied status sketch mentions running → blocked without ending a mission while also listing blocked as terminal. Resolve whether a pending approval is a separate waiting state/reason or an already-ended blocked run; do not allow ambiguous revival of a terminal run.

### Steps

| Event | Payload / behavior |
| --- | --- |
| step.proposed | n (1-based), action (normalized verb), raw (redacted full step ≤2000), head (first line ≤120) |
| step.started | Optional execution-start event correlated by n and action; exact required payload to define |
| step.finished | n, action, ok, duration_ms, result_head (redacted ≤200), optional/nullable error |

Normalized verbs use the existing action vocabulary: GLOB, GREP, LIST, READ, DIFF, MKDIR, CREATE, WRITE, EDIT, SHELL, CODE, TEST, TODO, PASS, FINISH. Verify the actual supported parser vocabulary before freezing the enum.

Legacy shape: `{"event":"step","n":3,"action":"EDIT: path|||old|||new"}`. Canonical `action` is only the verb; `raw` retains the redacted expression. Readers must normalize this distinction rather than treating both fields identically.

### Filesystem and recovery

| Event | Payload / behavior |
| --- | --- |
| fs.read | n, path, cached, bytes, blocked |
| fs.mutate | n, op (EDIT/WRITE/CREATE/MKDIR), path, rel, abs, checkpoint_id, receipt_id, optional diff_stat with `+`/`-` counts, optional diff and diff_truncated |
| fs.rollback | checkpoint_id, path, ok; failure detail may be added when defining schema |

Emit fs.mutate after successful gateway execution. Cap diffs at about 8 KiB and explicitly mark truncation. Redact content before every sink. Checkpoint and receipt references support the selected mutation preview and audited undo. Byte counts must be byte counts rather than normalized text lengths; define directory semantics separately.

### Shell, code, and verification

| Event | Payload / behavior |
| --- | --- |
| shell.exec | n, kind, cmd, exit_code (integer or null), blocked, output_head |
| verify.result | n, cmd, role, exit_code, ok, blocked, output_head |

`kind`: SHELL, CODE, TEST, VERIFY_CMD, AUTO_VERIFY.
`role`: verify_cmd or auto_verify. AUTO_VERIFY/py_compile success never establishes mission verification.

UPG-001 supplies launched/completed/crashed/timed-out state, command/workspace/run binding, and authoritative process exit status. Extend these events additively with that evidence or a receipt reference. The supplied string-marker interpretation (`[exit 0]`, `[BLOCKED]`) is a transitional adapter for existing helpers, not the final source of truth. Missing execution evidence produces unknown/not-run/blocked as appropriate, never assumed success.

### Compass, gateway, and PAUSE

| Event | Payload / behavior |
| --- | --- |
| compass.verdict | n, classification (OPEN/PAUSE/WITNESS), tool_name, action_summary, rule_matched, reason, optional/nullable receipt_id |
| gate.pause_minted | action_summary, action_sha256, optional token_id_prefix (≤8 hex) or opaque correlation id, expires_at, optional pending_id; never token body |
| gate.pause_resolved | decision (approved/declined/expired/invalid), by=operator as supplied, action_summary, optional opaque correlation/token_id_prefix |
| gate.receipt | Optional safe audit projection: receipt_id, disposition, action_sha256, executor |

Tokens are minted and stored through existing operator-only facilities. Event payloads do not grant permission. The approving operation must still authenticate the operator, bind the exact action, and preserve the existing single-use staging/claim/consume semantics.

The supplied `by=operator` convention needs an explicit policy for automatic expiry/invalidity: do not falsely attribute a timer or validation failure to an operator. Keep this as a schema decision before emitting those cases. An independent opaque pending id is preferable if a prefix would reveal part of a secret credential.

### Finish, PASS, review, and system

| Event | Payload / behavior |
| --- | --- |
| finish.declared | n, status, finish_basis, text (redacted FINISH body), synthesized; emitted after verification/finish acceptance |
| pass.declared | n, reason, helix_thread (boolean in supplied example) |
| review.completed | verdict, summary, model, target_mission |
| steer | kind (discovery_streak/repeat/create_hint), message |
| log | level (info/warn/error), message; migration escape hatch |

Prefer structured events over generic log entries. A review verdict is advisory and cannot silently upgrade an unverified mission. The supplied `finish.declared` name also covers synthesized FINISH; retain `synthesized` and basis to prevent confusion about who declared it.

## Echo and Helix projections

Echo is a derived record, not another bus event. Build it from canonical mission.end plus cached start identity:

```json
{
  "directive": "fix the failing test",
  "outcome": "redacted summary",
  "status": "needs_review",
  "finish_basis": "model_declared",
  "model": "grok-4.5",
  "steps": 7,
  "edited": ["foo.py"],
  "session": "S",
  "mission": "S__T",
  "ts": "2026-09-19T21:03:00+00:00"
}
```

`mission` and `ts` are additive. Echo/end status and finish_basis key presence must agree. Respect existing settings disabling echo; define projection equality for records actually requested and successfully persisted.

Helix tags retain `source:cosmic-cli`, `status:{status}`, `model:{model}`, and `finish_basis:{basis}` only when present. MissionBus feeds existing receipts rather than redefining the chronicle schema.

## Consumer mapping

| Consumer | Events / sources |
| --- | --- |
| IdentityBar | mission.start; external goal and verified health sources |
| MissionRail and step bar | mission.status, step events, mission.end |
| StepTape | step.proposed/finished, compass.verdict, shell.exec, fs events, finish.declared, pass.declared, steer |
| DiffPeek | fs.mutate, fs.rollback |
| CompassPulse | compass.verdict aggregates; scope/deduplication reconciled with existing Helix counts |
| PendingGates | gate.pause_minted minus resolved/expired entries |
| ReviewDock | review.completed; needs_review finish |
| DebriefScreen | mission.end plus last finish.declared |
| Mission Control tiles | Echo rollups and existing Helix compass sources |
| Mission Control steps | Session JSONL, normalized step.proposed and legacy step |

## Current-code mapping

| Current seam | Canonical target |
| --- | --- |
| _session_write(start) | mission.start |
| _session_write(step, n, action) | step.proposed and compatibility projection if needed |
| _execute_step completion | step.finished and fs/shell event |
| Compass decision / blocked outcome | compass.verdict; gate.pause_minted only when actually minted |
| FINISH verification | verify.result → finish.declared → mission.end |
| _append_echo | Pure projection of end plus identity |
| _session_write(end_event) | mission.end |
| _log | Temporary log event where a typed event does not exist |

## Emission and in-process API sketches

Supplied agent seam:

```python
def _emit(self, event: str, **payload) -> None:
    rec = {
        "v": 1,
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": self.session_id,
        "mission": self.mission_id,
        "seq": self._seq,
        **payload,
    }
    self._seq += 1
    self._bus.publish(rec)
    self._session_write_raw(rec)
```

Keep `_session_write` temporarily as a wrapper adding missing envelope fields. Before implementation, add validation/redaction, prohibit payload overrides of reserved envelope keys, serialize sequence allocation, and decide persistence versus subscriber ordering. The sketch is not the final error/durability policy.

```python
class MissionBus(Protocol):
    def publish(self, event: dict) -> None: ...
    def subscribe(self, fn: Callable[[dict], None]) -> Callable[[], None]: ...

class LocalMissionBus:
    def __init__(self):
        self._subs = []

    def subscribe(self, fn):
        self._subs.append(fn)
        return lambda: self._subs.remove(fn)

    def publish(self, event):
        for fn in list(self._subs):
            try:
                fn(event)
            except Exception:
                logger.exception("bus subscriber")
```

Subscriber failures must not raise into the agent loop. TUI worker-thread subscribers dispatch through `app.call_from_thread(app.apply_bus_event, event)`. No network is required. Agent ownership of JSONL versus an optional bus tee must be chosen explicitly to prevent double writes.

## Redaction, privilege, and size contract

- Redact every free-text field before publication, file append, echo projection, or callback. This includes directive, paths when sensitive, commands, raw, head, outcome, output_head, diff, reason, text, warnings, and error details.
- Exclude raw approval credentials and sensitive token-store paths. Opaque ids may correlate events; any prefix allowance must not disclose credential bytes.
- Existing sensitive-path refusals stay in place. Rendering an approval modal does not authorize reading the store through a model tool.
- Suggested caps: raw/outcome 2000 characters; head 120; result_head 200; output_head 500; diff 8 KiB. Specify units and truncation flags, and redact before truncation so cutting text cannot defeat secret detection.
- Operator approval remains an authenticated control path; `by` is descriptive data, not an authority check.
- Escape untrusted Rich markup at rendering. Subscriber error diagnostics must not leak unsanitized payloads.

## Migration phases

**A — envelope and compatibility:** Add v/mission/seq to session lines; enrich end with outcome and basis; add finish.declared; establish LocalMissionBus and TUI subscription. Introduce mission.start/end and step.proposed with legacy compatibility. Echo gains optional mission/ts.

**B — structured events:** Add fs.mutate, shell.exec, compass.verdict, verify.result, PAUSE lifecycle, and other catalog events. Board consumes structure instead of log parsing. Dashboard understands namespaced lifecycle/step names.

**C — cleanup:** Retire legacy names only after supported consumers migrate, or retain explicit aliases. Extend finish-line and applicable battery tests to assert emitted evidence as well as final result objects.

Readers normalize `mission.start|start`, `step.proposed|step`, and `mission.end|end`, including the changed shape of action. Unknown event types are ignored by tolerant readers.

The user supplied dual-write for one minor version or reader-first support as alternatives. Select one migration strategy before implementation. If dual-writing in one file, define alias identity/ordering and deduplication so panes, counters, and echo never process an action or terminal event twice. Do not assume a new event name is compatible with a legacy name-only reader.

## Invariants and acceptance tests

1. `verified` requires `finish_basis=verifier` and bound, successful operator-verifier evidence. Synthesized finish is needs_review; auto_verify cannot certify a mission.
2. Non-finish statuses omit finish_basis; finished canonical terminal records carry it. Legacy normalization never invents a basis.
3. Sequence is monotonic per mission; identities do not collide for simultaneous starts or reruns.
4. `mission.end` is the last execution event, not necessarily the last audit event. A later operator decision may append `gate.pause_resolved` for that mission's pending gate, correlated by `action_sha256` and optional `pending_id`, with sequence continuing. This does not reopen the mission or rewrite its terminal status/echo; a retry has a new mission identity. See [the implemented v1 contract](MISSION_BUS_v1.md#post-terminal-gate-decisions). Review/undo semantics remain a later design question.
5. Echo status and basis presence match the canonical end used to derive it; repeated delivery/replay cannot create duplicate rollups.
6. PAUSE bodies and sensitive-store paths never appear in emitted or persisted data. Test actual seeded credentials and nested/free-text fields, not just a broad token-looking regex.
7. At most one unresolved pause mint per bound action in the relevant mission/session approval scope; resolution permits the intended next lifecycle, not a retry mint storm.
8. Schema major version is 1; breaking renames require an explicit migration/version decision.
9. Subscriber exceptions do not break the agent loop; mutation of one subscriber's object cannot alter another subscriber's evidence or the persisted record.
10. Log-write failures and partial/torn tail lines have documented observable behavior. Replay is tolerant where appropriate and cannot falsely report durable evidence.

`validate_event` rejects missing envelope fields, invalid finish combinations, and forbidden sensitive payloads. Normalization and strict validation are separate concerns; legacy data must not be rejected merely for lacking new fields.

## Example event tapes

These compact examples show ordering and payload relationships. Placeholder timestamps/hashes are illustrative, not validation fixtures.

```jsonl
{"v":1,"event":"mission.start","ts":"…","session":"S","mission":"S__T","seq":0,"directive":"fix tests","model":"grok-4.5","exec_mode":"safe","helix":true,"verify_cmd":"pytest -q","max_steps":30}
{"v":1,"event":"mission.status","ts":"…","session":"S","mission":"S__T","seq":1,"status":"running"}
{"v":1,"event":"step.proposed","ts":"…","session":"S","mission":"S__T","seq":2,"n":1,"action":"READ","raw":"READ: tests/test_x.py","head":"READ: tests/test_x.py"}
{"v":1,"event":"fs.read","ts":"…","session":"S","mission":"S__T","seq":3,"n":1,"path":"tests/test_x.py","cached":false,"bytes":800,"blocked":false}
{"v":1,"event":"step.finished","ts":"…","session":"S","mission":"S__T","seq":4,"n":1,"action":"READ","ok":true,"duration_ms":12}
{"v":1,"event":"step.proposed","ts":"…","session":"S","mission":"S__T","seq":5,"n":2,"action":"EDIT","raw":"EDIT: …","head":"EDIT: foo.py"}
{"v":1,"event":"fs.mutate","ts":"…","session":"S","mission":"S__T","seq":6,"n":2,"op":"EDIT","path":"foo.py","checkpoint_id":"ck_1","receipt_id":"r1","diff_stat":{"+":3,"-":1},"diff":"@@ …"}
{"v":1,"event":"step.proposed","ts":"…","session":"S","mission":"S__T","seq":7,"n":3,"action":"SHELL","raw":"SHELL: rm -rf build","head":"SHELL: rm -rf build"}
{"v":1,"event":"compass.verdict","ts":"…","session":"S","mission":"S__T","seq":8,"n":3,"classification":"PAUSE","tool_name":"SHELL","action_summary":"rm -rf build","rule_matched":"destructive_rm","reason":"…"}
{"v":1,"event":"gate.pause_minted","ts":"…","session":"S","mission":"S__T","seq":9,"action_summary":"rm -rf build","action_sha256":"…","pending_id":42,"expires_at":"…"}
{"v":1,"event":"gate.pause_resolved","ts":"…","session":"S","mission":"S__T","seq":10,"decision":"declined","by":"operator","action_summary":"rm -rf build","pending_id":42}
{"v":1,"event":"mission.end","ts":"…","session":"S","mission":"S__T","seq":11,"status":"blocked","steps":3,"edited":["foo.py"],"warnings":[],"model":"grok-4.5","outcome":"[BLOCKED] …","block_message":"[BLOCKED] …"}
```

Verified finish, abbreviated intermediate sequence:

```jsonl
{"v":1,"event":"verify.result","ts":"…","session":"S","mission":"S__V","seq":30,"n":7,"role":"verify_cmd","cmd":"pytest -q","exit_code":0,"ok":true,"blocked":false}
{"v":1,"event":"finish.declared","ts":"…","session":"S","mission":"S__V","seq":31,"n":7,"status":"verified","finish_basis":"verifier","synthesized":false,"text":"…"}
{"v":1,"event":"mission.end","ts":"…","session":"S","mission":"S__V","seq":32,"status":"verified","finish_basis":"verifier","steps":7,"edited":["foo.py"],"outcome":"…"}
```

## Module map and smallest useful cut

```text
cosmic_cli/bus.py          LocalMissionBus and subscription API
cosmic_cli/events.py       enums, constants, validate_event, normalize_legacy
cosmic_cli/agents.py       emission seam and migration wrappers
cosmic_cli/tui/state.py    apply_bus_event reducer
cosmic_cli/dashboard.py    normalized session-tail reader
tests/test_bus_schema.py   schema, lifecycle, redaction, replay invariants
tests/test_finish_line.py  end/echo basis and verifier evidence
```

First cut requested by Anthony:

1. Envelope on all session lines: v, mission, seq.
2. Mission end with current fields plus outcome.
3. finish.declared immediately before execution end, with status/basis.
4. LocalMissionBus and TUI subscription.
5. step.proposed with the chosen legacy compatibility mechanism.
6. gate.pause_minted/resolved, without token bodies, for PendingGates.

## Decisions to resolve before implementation

- **End versus later work:** The supplied review example has seq=50 after an example mission.end at seq=40; board undo also occurs after completion. Either model review/recovery as separate activity streams referencing target_mission, or explicitly permit post-execution events. Choose one and revise the end invariant, projections, and debrief timing together.
- **Cancellation:** Decide whether mission.cancel is a control request followed by mission.end, or a terminal event itself. Keep one authoritative execution outcome. Its supplied status=error needs a distinct operator-cancellation reason so it does not imply a crash.
- **Identifiers:** Current agent mission stems use second-resolution stamps, so add collision-resistant identity for multiple missions/reruns. Do not use directive hash alone.
- **Write and publish order:** Current session writes swallow OSError. Define visible persistence failures and whether the UI can display events that failed to persist. Choose one owner for JSONL and echo projection; do not promise atomicity across separate files without implementing it.
- **Thread and replay safety:** Choose synchronization and immutable/copy-on-publish semantics, idempotent unsubscribe, and event identity for deduplication. Slow UI subscribers must not indefinitely hold the agent worker. Handle late subscription through explicit snapshot/replay rather than guessing missing state.
- **Health and aggregates:** Start identity does not prove current floor/Helix health; define query/reconciliation sources. Distinguish per-run live compass events from all-time Helix counts to avoid double counting.
- **UI authority:** An approval or undo button routes through shared privileged operations; the event bus never becomes an approval API. Resolve pending approval and retry behavior across process restarts and already-ended missions.
- **Schema finalization:** Define required fields per event, caps and units, structured unknown outcomes, correlation ids, and typed payload validation. The plan's code sketches omit some safeguards deliberately for readability; implementation must satisfy the stated invariants.
