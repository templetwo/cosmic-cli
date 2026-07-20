# Execution Plan: A Fail-Closed Proof-of-Allow Gate with a Kernel-Enforced Approval Floor for AI Coding-Agent Cockpits

*(Retitled from "Making cosmic-cli the First True Reference Monitor…". Redline of the v1 plan, incorporating an independently-verified feasibility review — Anthony + code seat, 2026-07-20.)*

## What changed from v1 (read this first)

1. **Title / framing retired.** "First True Reference Monitor" is gone. The deposited paper already proved complete mediation is **structurally unreachable** from userspace here — a wrapper cannot catch its own timeout kill — and retired the claim. The earned, adversarially-durable claim is **"a fail-closed proof-of-allow gate with a kernel-enforced approval floor."** Anderson-1972 stays in the paper as the **yardstick, not the title**: here is the classical bar (complete mediation), here is the proof userspace cannot clear it (the timeout hole), here is the maximal honest approximation. Nobody can take the earned claim away — the battery and the runtime verification back every word.
2. **The contract is over observable PROPERTIES, not topology.** v1's `execguard` assumed "cosmic-cli spawns the model's tool child under a profile." Verified false on the bridge: `install_grok_hooks.py` does `exec grok`, so cosmic **replaces its own process** and is not the spawner — it provisions grok's native `~/.grok/sandbox.toml` and verifies the floor with a canary. Fix (Pillar 3): write the conformance contract over **deny-by-default floor keyed to verdicts, fail-closed on absence** — properties a cockpit exhibits — not over who-spawns-whom. A property contract survives both spawn models; a topology contract dies on the first cockpit that execs. **`temple_floor.py` provisioning grok's own sandbox.toml IS bridge-path enforcement — a first-class path, not a stopgap.**
3. **TLA+ demoted to dessert.** Verified: zero of the five real bugs this project hit live where a model checker sees them, and the one concurrency bug (`fcntl` silently absent → race) is precisely where TLC's atomicity assumption would have **hidden** the defect. Load-bearing proof = **Hypothesis `RuleBasedStateMachine` against the real gate, wired into CI**. Keep a 30-60 line token-invariant TLA+ spec as an optional weekend item (checks in seconds, buys one sentence). Dessert, not dinner.
4. **Effort re-sized.** The macOS-floor line is **double-counted** (already shipped + runtime-verified). The full remainder is **2-4 months** at 5-10 h/wk, not a sprint. Linux drops to a **footnote**: detect-and-degrade, never assume Landlock present (two real boxes falsified "assume present"); if the Linux proof is wanted, run it on a **cloud VM (Ubuntu 24.04, kernel 6.8+)**, not the Jetson.

## Status: already banked (do NOT re-plan these)

- **DOI minted, permanent:** 10.5281/zenodo.21461197 (concept 10.5281/zenodo.21461196), Apache-2.0, sole author (ORCID 0009-0000-6440-1506). Paper + RFC + code zip + conformance-results.json deposited.
- **Repo:** `paper/cosmic-allow-v1` merged to `main`; tag `v0.9.4`; README/CITATION/.zenodo.json carry the DOI; thetempleoftwo.com features it.
- **Red-team findings #1/#3/#4 fixed** (`f5dac8b`), guarded by `tests/battery/`. **Gate-contract ASR 3.3%** (not v1's 10.7%), residual = the floor-covered `privilege_escalation` class.
- **macOS kernel floor: provisioned by default and runtime-verified** (`temple_floor.py` Seatbelt canary). Grok-build seam source-verified, pinned `ba69d70`.

## The thesis (earned form)

A small, always-invoked, fail-closed proof-of-allow gate with a kernel-enforced approval floor and a **published, property-based conformance suite** — the combination the big vendors are not building because their incentives favor per-product native controls. The moat is the proofs and the contract, not more features.

## Pillar 1 — Kernel floor (the "always invoked" property)

**Two enforcement domains, kept distinct (do not conflate):**
- **Cosmic's own agent loop.** `agents._run_shell` wraps tool children in `sandbox-exec` (`sandbox.py`). Here cosmic IS the spawner; a per-child profile fits. **On Linux this wrap is currently a no-op** (`sandbox_available()` → False) — that is the one place `py-landlock` genuinely fills a gap, and only for cosmic's own children.
- **The bridge.** `temple_floor.py` writes `[profiles.temple]` deny-globs into grok's native `~/.grok/sandbox.toml` + default profile; the launcher runs a floor canary then `exec grok`. Cosmic is a **config-provisioner + canary verifier**, not the spawner. Bridge levers: (i) provision the cockpit's sandbox config (shipped), or (ii) sandbox the parent before exec so children inherit. **Document (i) as the first-class bridge path.**

**macOS (primary dev target — DONE).** `sandbox-exec`/Seatbelt is deprecated but functional on macOS 26; Codex CLI, Gemini CLI, and Anthropic's sandbox-runtime all rely on it. Our floor is already runtime-verified here. Standing risk: Apple could remove `sandbox-exec` with no notice — mitigate by keeping the floor behind an interface. (Endpoint Security is ruled out: it needs a discretionary Apple-granted entitlement, unshippable solo.)

**Linux/Jetson (FOOTNOTE — not the critical path on Apple Silicon).** `py-landlock` (PyPI, MIT, Py≥3.10, kernel 5.13+, x86_64/aarch64) is real and its `Landlock().allow_read().apply()` API exists — but it is pre-1.0/single-author and **shares its name with two unrelated packages; pin the exact dist**. Hard lesson, verified: **Landlock is routinely compiled-in-but-disabled** — it needs `landlock` in `CONFIG_LSM`/`lsm=` at boot, returns `EOPNOTSUPP` when off, and `/sys/kernel/security/landlock` is then absent (our 5.10 box predates it entirely; our 6.12 box had it off). **Treat Landlock as detect-and-degrade, never assume-present.** The **Jetson is a trap for a quick proof**: 5.15 caps Landlock at ABI v1 (filesystem-only, no REFER/rename, no network), and NVIDIA's tegra kernel likely needs a rebuild to enable it — a separate hardware-support track, not the reference verification. **Run the Linux runtime proof on a cloud VM (Ubuntu 24.04)** where it is genuinely an hour.

## Pillar 2 — Proven core (the "verifiable" property)

**Load-bearing:** Hypothesis `RuleBasedStateMachine` driving random operation sequences against the real `flock`-guarded gate, wired into CI so spec/code drift breaks the build. This touches the real code and thus the platform/representation seams where every real bug actually lived.

**Optional dessert:** a 30-60 line TLA+/TLC spec of the single-use-token invariant only (unused/consumed, atomic consume, nonce, parse-or-deny). Checks exhaustively in seconds; buys a clean "the single-use safety property is machine-checked" sentence. **Do not** attempt "the gate core" — the value question resolves against it (the `fcntl`-absent race is exactly what an atomicity-by-fiat model masks). This matches the paper's §8 already.

## Pillar 3 — A standard others implement (the property contract)

Publish COSMIC-ALLOW RFC v1.1 (**done, in the repo + deposit**) and the battery **as a property-based conformance kit**. The contract is written over **observable properties**, so any cockpit conforms by exhibiting the behavior:
- **Proof-of-allow:** entire stdout == `COSMIC-ALLOW v1 <nonce>` (+ single newline); env-only nonce, whole-string match; stdin opaque; env never printed.
- **Fail-closed on absence:** a broken/hung/absent gate produces no sentinel → deny. Deny-wins aggregation.
- **Kernel floor keyed to verdicts:** the approval-token store is unreadable from the model's execution context regardless of who spawns the child (provisioned into the cockpit's own sandbox, or the parent sandboxed before exec).
- **Privilege ranking:** approval flows only downward through a channel the lower level cannot reach; TTY-gated; no env break-glass.

**Grow the thin probe classes** (`fail_open` n=2, `deployment_gap` n=1, `env_disclosure`/`resource_abuse` n=2) before any aggregate ASR is called canonical. Publish invariants + a probe **schema** as the normative bar; the 150-probe battery is one conforming test-vector set. **Acceptance bar for the kit: a stranger can pass it against the property contract without ever reading cosmic's source. If interpreting the kit requires cosmic internals, Pillar 3 did not actually land.** Recruit **one** external GATE implementer (grok-build is already a source-verified second COCKPIT; the gap is a second gate). Recruitment, not RFC prose, is the real mitigation for "no second implementer."

**Responsible disclosure.** Own-gate findings (our ASR, our five findings) publish openly — good-faith transparency. Any probe implicating a third-party cockpit's specifics waits for coordination with that vendor before publication. This plan and the battery are clear to publish (own-gate findings plus public facts only).

## Pillar 4 — Cross-architecture review seat (advisory only)

PAUSE-class actions get a cross-family second read; **reviewers inform, only the TTY consumes** — this sidesteps the LLM-as-judge reliability failure the literature documents. Report agreement with a chance-corrected metric (Cohen's kappa), never raw. State the advisory-only posture explicitly in any writeup.

## Re-sequenced execution (deposit removed — it's done)

Effort = total engineering hours at 5-10 h/wk. **Realistic calendar for the whole remainder: 2-4 months.**

1. **Framing redline (free, do first).** Retitle; fold Pillars 1/3 property-contract language; align to the paper's honest claim. (This document.)
2. **Phase 0 hygiene residual (~4-6h).** Optional: make `evaluate_rules` fail-closed on no-match (currently pinned fail-open with the classifier structural-deny as backstop — a battery test asserts the current behavior, so this is a coordinated change, not a hole). Low priority.
3. **Linux floor runtime-verification on a cloud VM (~an hour of work + VM setup).** Ubuntu 24.04 (kernel 6.8+, Landlock in the default LSM); exercise **detect-and-degrade in BOTH directions** — floor present (bind + prove the token read is denied) AND floor absent (prove the launcher refuses rather than running unprotected). Its entire output is **one sentence in the spec**. Removes the paper's one live caveat → ship as **v0.9.5** under the concept DOI, flip paper §7 to "runtime-verified on Linux (kernel/ABI …)." NOT the Jetson.
4. **Hypothesis stateful test → CI (~10-15h).** The load-bearing proof. Optionally the 30-60 line token TLA+ spec as a weekend add.
5. **Battery → property-based conformance kit + probe schema; grow thin classes (~10-20h).**
6. **Recruit one external gate implementer (ongoing).** Claude-Code-only minimal gate, or the OWASP/Microsoft agent-governance community.
7. **arXiv cs.CR (optional).** Needs an endorsement (independent researcher); the DOI is the citable fallback and gates nothing.

## Do-not-build (unchanged, ruthless)

No new Stargazer actions; no new model integrations; **no dashboard/UI beyond the read-only localhost viewer already shipped** (it is excluded from the deposit snapshot and must stay localhost-only, never an approval/mutation surface); no network-proxy content-inspection product; no MCP gateway; no premature multi-platform breadth (macOS + one cloud-VM Linux target; the Jetson is a separate optional track).

## Risk register (updated)

- **macOS `sandbox-exec` deprecation** (medium/high): shared by every agent CLI; keep the floor behind an interface. Trigger: an Apple removal date.
- **Landlock availability** (NEW, high on Linux): compiled-in ≠ enabled; two real boxes proved it. Mitigation: detect-and-degrade; verify on a modern cloud VM; treat the Jetson as a separate track.
- **No second implementer** (medium-high): ship the property-based conformance kit; recruit one. Claim "specification + conformance suite," not "standard," until a second gate exists.
- **Spec/code drift** (medium): the Hypothesis CI binding fails the build on drift.
- **Solo-maintainer bus factor** (high over time): the DOI trail + property contract + conformance kit are designed to survive the author.
- **Vendor commoditization** (high): win by being the small, verified, cross-vendor, always-invoked gate with a public conformance bar. If a vendor ships a verified cross-vendor equivalent with a public suite, pivot to conformance-suite/audit authority.

## Standing framing law

Always claim **"a fail-closed proof-of-allow gate with a kernel-enforced approval floor,"** never "reference monitor." Always report the **gate-contract ASR and the kernel floor as two separate layers**; never quote 3.3% as if it covered the floor, and never claim 0% on `privilege_escalation` at the gate layer.

## Citation fixes (verified; apply wherever these are cited)

- Altman/OpenAI MCP adoption is **March 26, 2025** (not ~March 25).
- The "≥95% same-verdict at temp 0 / as low as 70% at temp 1" figures are **Stureborg et al. 2024** (arXiv 2405.01724); use Haldar & Hockenmaier 2025 as corroboration, not co-source.
- Keep Google Cloud "Agent Gateway" distinct from the open-source `agentgateway.dev`/kagent project.
- Everything else in v1's citation set (both CVEs, Hassabis quote, OWASP 2026, competitive landscape, JudgeBiasBench, the AWS/TLA+ quote) verified real on primary sources.
