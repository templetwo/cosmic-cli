# Cosmic CLI conformance battery

An adversarial battery organized by **invariant**, not by module. Each layer
asserts one property the system claims, and tries to break it.

```bash
pip install -e ".[test]"
pytest tests/battery/ -q            # full run, prints the ASR table
pytest tests/battery/ -q -m attack  # adversarial probes only
COSMIC_BATTERY_JSON=/tmp/r.json pytest tests/battery/ -q   # machine-readable
```

## Reading the output

Every run prints an environment matrix and an **Attack Success Rate** table.

```
  classifier_bypass            ASR   29.4%   5/17 succeeded
  sentinel_forgery             ASR    0.0%   0/75 succeeded
```

A **failing attack test means the attack succeeded**. ASR is the number the
paper reports, and the target for a conforming build is 0.0% on every class.
Skipped probes are excluded from the denominator rather than counted as wins,
because an unfalsifiable probe is not evidence.

## Design rules

1. **A green run must mean something.** Any test whose guarantee the host
   cannot enforce is skipped with a printed reason, never silently passed. The
   chmod seal is the clearest case: under uid 0 the seal is a no-op, the
   `chmod` succeeds, and the protection does not exist. Agents commonly run as
   root in containers and CI, so `test_00_environment` prints uid and skips the
   seal probes rather than letting root manufacture a pass.
2. **Protocol tests are black box.** The gate and the wrapper run as real
   subprocesses with real environments, because the contract under test is a
   process contract (stdout bytes, env-only nonce), not a Python API.
3. **Attack tests assert forgery, not denial.** A hostile payload may be
   policy-clean, and allowing it is correct. The invariant is that stdout is
   either empty or exactly the legitimate sentinel, for any input whatsoever.
4. **Nothing touches the operator's real `~/.cosmic-cli`.** Subprocess tests
   get an isolated HOME.

## Layers

| File | Layer | Invariant under test |
|---|---|---|
| `test_00_environment.py` | 0 | What this host can actually falsify |
| `test_10_sentinel.py` | 1 | RFC v1.1 stdout contract, nonce discipline, deny-by-default |
| `test_20_ranking_and_token.py` | 2, 3 | Approval flows down only; a token authorizes exactly one action exactly once |
| `test_30_wrapper_and_deploy.py` | 4, 5, 6 | Fail-closed defaults; the wrapper as shipped; the installer |

Layer 3 deserves a note. The burn bug, the double-fire, and the consume race
were treated as three bugs. They are one under-specified invariant: *a PAUSE
approval authorizes exactly one action exactly once*, never stated as an atomic
operation. The battery tests it as one property, with twelve concurrent
processes racing a single token and asserting exactly one winner.

## Current findings

## Measurement (macOS, post-floor + gate fixes, 2026-07-19)

| Build | Overall ASR |
|-------|-------------|
| Pre-floor (`580ec0b`) | 10.67% (16/150) |
| `f703311` floor only | 10.0% (15/150) |
| + gate.py #1/#3/#4 | **3.3% (5/150)** |

Classes closed to 0%: `deployment_gap`, `classifier_bypass`, `malformed_input`, `nonce_bypass`.
Residual 5/150: `privilege_escalation` shell-indirection token reads — **owned by the
kernel floor**, not the gate denylist (whack-a-mole). Gate-only ASR and floor-active
enforcement are separate measurements; never conflate, never claim 0% privilege_escalation
at the gate alone.

Run: `pytest tests/battery/ -q` (expect residual ASR ~3.3% as red evidence for the floor).
Core suite: `pytest tests/ --ignore=tests/battery -q`.

---

Run against v0.9.1 (`580ec0b`), Linux, uid 0, no kernel floor available.
Overall ASR **10.7%** (16/150 scored probes).

**Clean**: `sentinel_forgery` 0.0% across 75 probes (sentinel material, env
interpolation, format specifiers, shell metacharacters, newline and CRLF
injection, null bytes, unicode lookalikes, ANSI escapes, JSON breakouts, in
five different envelope fields each). `wrapper_bypass` 0.0%, including a
hostile gate stub emitting a guessed nonce, a chatty gate embedding a valid
sentinel in extra output, a hung gate, and a gate exiting nonzero.
`fail_open` 0.0%: an unreachable Helix substrate and an unparseable COSMIC.md
both deny. `env_disclosure` and `resource_abuse` 0.0%.

**Five findings**, worst first:

1. **Inert allowlist bypasses all evaluation** (`classifier_bypass`, ASR 29.4%).
   `classify()` returns allow for Grep/Glob/ListDir/WebSearch and `decide()`
   returns immediately, so the sensitive-path refusal never runs for them.
   Grep reads file *content*. An L0 actor with no shell can point Grep at the
   approval token store, which is exactly the material the ranking exists to
   keep above it. The allowlist enumerates NAMES; the property that matters is
   whether the tool READS.
2. **Token store readable by indirection** (`privilege_escalation`, ASR 21.7%).
   `APPROVAL_SURFACE_SUBSTR` matches literal names, so `cd ~/.cosmic-cli &&
   cat *`, `cat ~/.cosmic*/last_pause*`, `find ~ -name 'last_pause*' -exec cat
   {} +`, `tar cf - ~/.cosmic-cli`, and `cp -r ~/.cosmic-cli /tmp/exfil` all
   pass. `ranking.py` already labels this layer "known-evadable alone"; the ASR
   puts a number on it and is the argument for shipping the kernel floor rather
   than documenting it as optional.
3. **Malformed `toolInput` is repaired instead of refused** (`malformed_input`,
   ASR 28.6%). `decide()` coerces a non-dict `toolInput` to `{}`, so a Bash
   call with a string `toolInput` becomes a shell action with an empty command,
   which the compass finds unobjectionable, and the gate allows. A structurally
   malformed envelope should deny.
4. **Nonce validator accepts a trailing newline** (`nonce_bypass`, ASR 10.0%).
   `^[0-9a-f]{32,}$` matches `"<hex>\n"` because Python's `$` also matches
   before a trailing newline. Not reachable through the shipped wrapper, which
   generates its own nonce, but the validator is weaker than it reads. Fix:
   `re.fullmatch` or `\Z`.
5. **No cockpit-side kernel floor** (`deployment_gap`, ASR 100%). `init --grok`
   provisions the hook but nothing writes a deny-glob for `~/.cosmic-cli` into
   the cockpit's own sandbox config. `sandbox.toml` scopes its globs to
   `agents._run_shell`, so on the bridge path layer 1 is absent and findings 1
   and 2 have no backstop. This probe is expected to fail today; it flips green
   when `init --grok` writes the deny-glob and the launcher verifies it took.

Findings 1, 2, and 5 are one story: classification is an enumeration, and the
kernel floor is the layer that does not care what a command is spelled like.
