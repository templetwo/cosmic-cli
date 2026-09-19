# Grok Build seam re-verification — 2026-07-26

## Result

The six load-bearing claims in `docs/SEAM_CONTRACT.md` are **still correct** in current upstream source. I found no change to PreToolUse gate order, sequential dispatch, first-deny behavior, empty-registry behavior, the JSON vocabulary, or exit-code precedence.

There are two gate-relevant upstream changes since the pin:

1. Enforcing sandbox profiles now protect direct global hook sources from writes and refuse to start when that protection cannot be verified. This strengthens the integrity of the configured gate; it does not change gate decision order.
2. Plugin hook event filtering now delegates to the central event parser. Plugin manifests can therefore use every accepted spelling, including camelCase and per-operation aliases, without a second static allowlist drifting. Plugin hooks are still appended after file/config hooks.

`PostToolUse` is not new since the pin. It was present and observe-only at the pinned tree and remains so. Cosmic's zero-reference gap is an untapped existing surface, not upstream drift.

## Revisions and method

- Cosmic pinned monorepo source revision: `ba69d70c2f7d70a130a323b2becdf137af784c7f`
- Cosmic pinned published-tree head: `ba76b0a683fa52e4e60685017b85905451be17bc`
- Fresh `xai-org/grok-build` `main` tree actually fetched: `47348d13ec4508dcfe440e34c6d511bb02998fb2`
- Current tree's `SOURCE_REV`: `d02693a856a54f1030695b36b91d276e96b30b23`
- Current tree commit timestamp/subject: `2026-07-25T18:44:42Z`, `Synced from monorepo`

I fetched the pinned published-tree commit into the same fresh clone and diffed `FETCH_HEAD..HEAD`, then read the current implementations in:

- `crates/codegen/xai-grok-hooks/src/dispatcher.rs`
- `crates/codegen/xai-grok-hooks/src/runner/mod.rs`
- `crates/codegen/xai-grok-hooks/src/runner/command.rs`
- `crates/codegen/xai-grok-hooks/src/event.rs`
- `crates/codegen/xai-grok-hooks/src/discovery.rs`
- `crates/codegen/xai-grok-agent/src/plugins/hooks_adapter.rs`
- `crates/codegen/xai-grok-shell/src/session/acp_session_impl/tool_calls.rs`
- `crates/codegen/xai-grok-shell/src/session/acp_session_impl/hooks_plugins.rs`
- `crates/codegen/xai-grok-shell/src/config/mod.rs`
- `crates/codegen/xai-grok-sandbox/src/`

All source locations below are in current tree `47348d13…`.

The requested path-level diff disposition was:

| Surface | Pinned → current result |
|---|---|
| `xai-grok-hooks/src/dispatcher.rs` | Core dispatcher unchanged; one test fixture gained the new provenance field. |
| `xai-grok-hooks/src/runner/mod.rs` | No substantive diff. |
| `xai-grok-hooks/src/runner/command.rs` | Gate parser behavior unchanged; fixture/provenance coverage changed. |
| `xai-grok-shell/session/acp_session_impl/tool_calls.rs` | Large surrounding refactor/format churn, but the plan → Pre hooks → permission block and both Post dispatch blocks are semantically the same as the pin. |
| plugin hook adapter/registry | Static plugin event allowlist replaced by the central event parser; provenance is explicit; append ordering is unchanged. |
| sandbox provisioning | Substantive new hook-source write-deny, identity checks, bwrap/Seatbelt enforcement, and namespace-lockdown work. |

## Claim-by-claim verification

### 1. PreToolUse is first among runtime gates after plan mode — still true

`xai-grok-shell/src/session/acp_session_impl/tool_calls.rs:907-923` computes and enforces the plan-mode edit gate first:

```rust
let access_kind = AccessKind::from(&tool_input);
let plan_gate = plan_mode_edit_gate(&self.plan_mode.lock(), &tool_input, &access_kind);
if plan_gate != PlanEditGate::Allow {
    // ... reject and return ...
}
```

The registry PreToolUse dispatch begins at `tool_calls.rs:936-955`:

```rust
if self.hook_event_active(HookEventName::PreToolUse) {
    // ... construct envelope ...
    let pre_result = dispatch_pre_tool_use(&registry, &envelope, &ctx).await;
```

The client-supplied PreToolUse callback follows the registry at `tool_calls.rs:983-988`. Permission handling does not begin until `tool_calls.rs:1008`:

```rust
if !plan_file_auto_approve {
    // permission request/decision path
```

The exact current order is therefore:

`plan-mode gate → registry PreToolUse hooks → client PreToolUse callback → permission path → tool execution`.

### 2. Hooks run sequentially in config order — still true

`xai-grok-hooks/src/dispatcher.rs:44-47` states the contract directly:

```rust
/// Runs hooks sequentially in config order. Only an explicit `deny`
/// decision from a hook stops the chain and blocks the tool call.
```

The implementation is one awaited loop, not a join or fan-out (`dispatcher.rs:79-92`):

```rust
for spec in hooks {
    // ...
    let (result, elapsed, http_info) = runner::run_hook(spec, envelope, ctx, GateKind::Tool).await;
```

Registry insertion also preserves append order (`xai-grok-hooks/src/discovery.rs:60-63`):

```rust
for spec in specs {
    self.hooks.entry(spec.event).or_default().push(spec);
}
```

### 3. First explicit deny wins; allow does not skip later deniers — still true

The deny arm returns immediately (`dispatcher.rs:95-115`):

```rust
HookRunnerResult::Decision(HookDecision::Deny { reason, .. }) => {
    // ...
    return PreToolUseResult {
        decision: HookDecision::Deny { reason, hook_name: spec.name.clone() },
        results: run_results,
    };
}
```

The allow arm only records success and falls through to the next loop iteration (`dispatcher.rs:117-128`). There is no return or break in that arm.

### 4. Empty registry means Allow — still true

`dispatcher.rs:65-70`:

```rust
let hooks = registry.hooks_for(HookEventName::PreToolUse);
if hooks.is_empty() {
    return PreToolUseResult {
        decision: HookDecision::Allow,
        results: Vec::new(),
    };
}
```

After a nonempty chain with no explicit deny, the final result is also Allow (`dispatcher.rs:158-162`). Hook failures remain fail-open; that behavior is explicit at `dispatcher.rs:49-59` and is not a new change.

### 5. JSON vocabulary is `allow` / `deny` / optional `reason` — still true

`xai-grok-hooks/src/runner/mod.rs:32-38` defines:

```rust
/// `{"decision": "allow" | "deny", "reason": "…"}`.
pub(crate) struct GateHookJson {
    pub decision: String,
    #[serde(default)]
    pub reason: Option<String>,
}
```

`runner/mod.rs:47-57` accepts exactly `deny` and `allow`; an unknown decision value is an error. A deny without `reason` receives the default `denied by hook '<name>'` reason.

### 6. Exit 2 and JSON precedence — still true

`xai-grok-hooks/src/runner/command.rs:422-448` settles both mixed cases:

```rust
Ok(HookDecision::Deny { reason, hook_name }) => {
    // A JSON deny is honored on any exit code (fail-safe).
    return (HookRunnerResult::Decision(HookDecision::Deny { reason, hook_name }), elapsed);
}
Ok(HookDecision::Allow) => {
    if exit_code == GATE_EXIT_CODE {
        // Exit 2 wins over a JSON allow ...
    } else {
        return (HookRunnerResult::Decision(HookDecision::Allow), elapsed);
    }
}
```

The exit-code ladder at `runner/command.rs:456-470` maps `0 → Allow`, `2 → Deny`, and other exit codes to hook failure. Consequently:

- JSON `deny` wins with exit 0, 2, or another code.
- JSON `allow` wins except when exit is 2.
- Exit 2 plus JSON `allow` denies with the generic exit-2 reason; the JSON allow does not survive.

## Gate-relevant changes since the pin

### Direct hook source write protection is new and fail-closed

The pinned tree has no `xai-grok-sandbox/src/hook_write_deny.rs`; the current tree adds it. Current `hook_write_deny.rs:177-215` resolves configured hook sources, rejects missing configured sources, hard-link aliases, invalid JSON aliases, and builds the platform enforcement plan. Its error vocabulary also explicitly covers symlink retargeting and path-identity changes (`hook_write_deny.rs:23-47`).

The shell refuses startup when required protection is absent. On Linux, `xai-grok-shell/src/config/mod.rs:1354-1368` verifies the bwrap mount state and exits 1 on missing/writable hook mounts; `config/mod.rs:1402-1410` verifies again after apply:

```rust
if requires_hook_write_deny
    && xai_grok_sandbox::is_inside_bwrap()
    && let Err(e) = xai_grok_sandbox::verify_hook_write_deny_enforced()
{
    eprintln!("error: required hook write-deny mounts not verified after apply ({e}); refusing to start");
    std::process::exit(1);
}
```

Current Linux protection also installs a process-wide namespace lockdown. `xai-grok-sandbox/src/child_net.rs:56-61` denies `unshare`, `setns`, and namespace-bearing `clone`, while making `clone3` fall back rather than permit an uninspectable namespace creation. This closes obvious child-process escapes from the read-only hook mounts.

This is a strengthening, not an ordering change: it protects the bytes from which the registry snapshot is loaded.

### Plugin aliases are broader; ordering is unchanged

At the pin, the plugin adapter had its own static event-name allowlist. Current `xai-grok-agent/src/plugins/hooks_adapter.rs:138-168` instead keeps a key exactly when `HookEventName::parse_key` accepts it:

```rust
.filter(|key| HookEventName::parse_key(key).is_none())
```

This admits central-parser spellings such as `preToolUse`, `beforeShellExecution`, `postToolUse`, and `afterFileEdit` for plugin manifests. It is gate-relevant because a plugin PreToolUse alias that was previously filtered out can now participate in the deny chain.

It does not get higher priority. `xai-grok-shell/src/session/acp_session_impl/hooks_plugins.rs:640-679` discovers disk/config hooks first, then explicitly re-appends each active plugin's file and inline hooks. Since registry append is `Vec::push`, plugin hooks follow already-discovered hooks. An earlier allow still cannot suppress a plugin deny; an earlier deny prevents later hooks from running by the existing first-deny rule.

The new `HookSpec.layer = Plugin` provenance is classification/telemetry metadata, not a dispatch sort key.

### PostToolUse and ACP did not drift into a gate

`PostToolUse` and `PostToolUseFailure` existed at the pinned tree. Current `xai-grok-hooks/src/event.rs:112-129` still assigns both `GateKind::Observe`. Current `dispatcher.rs:356-357` says it plainly:

```rust
/// Dispatch an observe-only event against all matching hooks; never denies.
```

The current success payload is flattened camelCase and includes `toolName`, `toolUseId`, `toolInput`, `toolResult`, truncation flags, `durationMs`, `isBackgrounded`, and optional `subagentType` (`event.rs:406-425`). The failure payload substitutes `error` for `toolResult` (`event.rs:427-439`).

ACP/meta-dispatch names remain consistent across the boundary. PreToolUse uses the resolved target name at `tool_calls.rs:932-945`; PostToolUse uses `PreparedToolCall::hook_tool_name()` at `tool_calls.rs:624-631`. The same behavior is present in the pinned implementation. I found no new ACP gate ahead of PreToolUse and no marketplace code inserted into tool authorization.

## Bottom line

`docs/SEAM_CONTRACT.md` remains source-correct for the six claims on which Cosmic relies. No contract edit is warranted from this re-verification. The new sandbox protection improves gate-source integrity, and the expanded plugin alias surface adds possible later deniers without weakening sequential first-deny semantics.
