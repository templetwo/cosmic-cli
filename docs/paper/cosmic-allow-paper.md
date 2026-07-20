# Fail-Closed Governance for Fail-Open Harnesses: A Cross-Vendor Proof-of-Allow Protocol for Coding Agents

**Anthony J. Vasquez Sr.**
The Temple of Two · ORCID 0009-0000-6440-1506
COSMIC-ALLOW RFC v1.1 · cosmic-cli v0.9.4 · Apache-2.0

*Draft for review. Version 0.9.4, 2026-07-20.*

---

## Abstract

AI coding agents now edit files and run shell commands with limited human
oversight, and the harness that wraps them tends to fail open: when a governance
hook errors, times out, or is bypassed, the action usually proceeds. We present
COSMIC-ALLOW, a cross-vendor sentinel protocol (RFC v1.1) that mounts a
fail-closed governance gate under fail-open cockpit hook systems, specifically
Claude Code hooks and the newly open-sourced Grok Build hooks, which are
deliberately compatible. The design makes deny the ground state. Proof-of-allow
is strict: the gate's entire standard output must equal `COSMIC-ALLOW v1 <nonce>`
(plus a single trailing newline), where the nonce is a per-invocation value of at
least 128 bits passed only through an environment variable; the gate treats standard input as opaque bytes, never
reflects it, and never prints the environment. Enforcement is layered by design:
a kernel sandbox floor, provisioned by default into the cockpit's native
Landlock/Seatbelt configuration and verified at launch by a floor canary (a live
sandboxed probe on macOS; a configuration check pending live Linux verification)
that refuses to start the cockpit if the floor is not in place, is the
load-bearing boundary; a privilege ranking makes approval flow only from a higher trust level
through a channel a lower level cannot reach; and command classification is
defense in depth above them. We measure conformance as Attack Success Rate (ASR)
against a hostile-payload battery that scores the gate contract in isolation: the
gate-contract layer scores 3.3% overall, and its one residual class,
shell-indirection reads of the approval-token store, is by design the kernel
floor's responsibility. We are deliberate about scope. This is a fail-closed
proof-of-allow gate with a kernel-enforced approval floor; it is not a
complete-mediation reference monitor over all model actions, and we say so. The
findings come from a four-seat adversarial review method extending the author's
prior IRIS Gate work. The reference implementation, the RFC, and the conformance
battery are released under Apache-2.0.

---

## 1. Introduction and threat model

A coding cockpit is a program that runs a language model in a loop: the model
proposes tool calls (read a file, edit a file, run a shell command), the cockpit
executes them, and the results feed the next turn. Modern cockpits — Claude Code,
the newly open-sourced Grok Build, and compatible shells — expose a PreToolUse
hook so a third party can inspect a proposed action before it runs.

The problem is that these hook systems fail open. A hook that times out, crashes,
is not found, or emits malformed output is, by the cockpit's own runtime
semantics, treated as non-blocking, and the action proceeds. This is a reasonable
default for a productivity tool and a dangerous one for a governance layer: it
means the governance layer's *absence* is indistinguishable from its *assent*. A
wrapper cannot catch its own kill; a governance process wedged past the hook
timeout cannot itself deny.

Our threat model is an untrusted L0 model with the cockpit's tool surface,
including the ability to run shell commands, that may be steered by adversarial
content it reads (prompt injection, poisoned repositories, tool-poisoning). We
assume the operator is honest but not continuously watching. We want three
properties: (P1) no action executes without an affirmative, unforgeable
proof-of-allow; (P2) the model cannot approve its own actions; and (P3) the
material that authorizes actions (the approval-token store) cannot be read by the
model regardless of how a command is spelled.

Our thesis is that these are achievable by inverting the default: make **deny the
ground state**, require an affirmative cryptographic proof for every allow, and
put the load-bearing boundary in the kernel rather than in a cooperating hook.

## 2. Background and related work

The 2026 execution-security literature systematizes policy-enforcement fragility
and time-of-check-to-time-of-use (TOCTOU) as first-class failure categories and
documents production agent-harness CVEs. Runtime governance for agents has
emerged as a category: Microsoft's Agent Governance Toolkit (open-sourced April
2026) ships an Agent OS policy engine that intercepts agent actions at the
application middleware layer with sub-millisecond decision latency; IBM and others
frame an "Agent Control Plane" that enforces policy at runtime. Guardrail and
human-in-the-loop frameworks (NeMo Guardrails, guardrails-ai, LangGraph's
interrupt-on-sensitive-tools) provide cooperative, in-harness controls. A wave of
"agent firewalls" and MCP gateways (Pipelock, Docker MCP Gateway, and others)
focus on network and tool-traffic inspection.

What is missing from this landscape, and what we target, is a small,
always-invoked, fail-closed proof-of-allow contract with a kernel-enforced floor
and a published conformance bar — a design that does not depend on the cockpit
cooperating. We do not claim to be the first runtime governance layer; Microsoft's
middleware engine predates us and mediates more. We claim a specific, verifiable
combination: a fail-closed sentinel contract, a kernel floor over the approval
store, and a public conformance suite, positioned so a second vendor can implement
to a testable bar.

Attack Success Rate (ASR), the fraction of attack instances that achieve the
attacker's objective, is the canonical metric from Agent Security Bench (Zhang et
al., arXiv:2410.02644); we adopt it. The EU AI Act Annex III high-risk
obligations, including Article 14 human oversight and Article 15 robustness, were
scheduled to apply from August 2, 2026; the May 2026 Digital Omnibus political
agreement defers stand-alone Annex III obligations to December 2, 2027 upon formal
adoption, expected at this writing before the August date. Either way, they
motivate a demonstrable, auditable oversight mechanism.

## 3. The COSMIC-ALLOW protocol

The protocol is specified normatively in the companion RFC v1.1; we summarize the
load-bearing parts.

**Proof-of-allow.** Deny is the ground state. On a genuine allow (OPEN
disposition), the gate writes to standard output exactly `COSMIC-ALLOW v1 <nonce>`
followed by a single newline, and nothing else; this is the only write to standard
output in any code path. The
wrapper checks the whole output for equality against the sentinel it expects. A
sentinel embedded in extra output is a deny; the gate's standard error is
discarded from the decision. Exit codes are not the signal — presence of the exact
sentinel is.

**The nonce.** The nonce is at least 128 bits of fresh randomness, generated by
the wrapper per invocation and delivered to the gate only through an environment
variable. The gate validates it with a whole-string match, so a trailing newline
is rejected, and never reflects it except in the sentinel. This closes replay and
guessing: a stub gate that prints a plausible sentinel with the wrong nonce is
denied.

**Envelope discipline.** Standard input is opaque bytes, never interpolated into
shell code and never reflected into model-visible output. A structurally malformed
envelope denies — a `toolInput` that is present but is not an object denies rather
than being coerced into a permissive empty object — and any unrecognized or
MCP-qualified tool name denies by default in the classifier before any rule engine
runs.

**Aggregation and fail-closed composition.** When multiple hooks run, a single
explicit deny wins. Because the cockpit's hook runner is fail-open, the positive
protocol is what supplies the fail-closed property: allow requires a sentinel the
wrapper can verify against its own nonce, so a broken, hung, or absent gate emits
no sentinel and therefore denies.

**Boot canary.** Before a session begins, the launcher exercises the gate with one
probe per action class (shell, write, read-sensitive, MCP), bound to a per-boot
nonce, and refuses to start if the gate is absent, if a deny-class probe emits a
sentinel, or if an allow-class probe does not. Absence of proof is not proof of
absence; an ungoverned session does not begin.

## 4. Implementation: cosmic-cli under two cockpits

The reference implementation, cosmic-cli, mounts the same gate under both Claude
Code and Grok Build PreToolUse hooks, whose surfaces are deliberately compatible.
Its avionics layer sits on a shared SQLite chronicle (the T2Helix substrate) and a
compass with three dispositions — WITNESS (hard deny), PAUSE (deny pending one
human approval), OPEN (allow) — plus a typed policy engine, a gateway that
produces execution receipts, content-preserving checkpoint and rollback, and an
independent review seat.

Enforcement is three layers, ordered by how load-bearing they are.

**Layer 1, the kernel floor.** `cosmic-cli init --grok` writes a deny-list for the
approval-token files into the cockpit's native sandbox configuration
(`~/.grok/sandbox.toml`; Seatbelt on macOS, Landlock plus a bubblewrap bind-over
on Linux) and sets it as the default profile. It denies exactly the token
artifacts, never the whole tool-state directory (which would disable the hook).
The launcher runs a floor canary; on macOS it plants a token and confirms a
sandboxed process cannot read it, and on Linux it verifies the profile is
installed and defers live kernel verification (Section 7). If the check fails, the
cockpit does not start.
This is what makes the guarantee independent of cockpit cooperation: a cockpit
that ignores the sentinel contract still runs its shells under a profile that
kernel-denies the approval store.

**Layer 2, the privilege ranking.** Levels run L3 operator, L2 seat, L1 gate, L0
model. Approval flows only downward, through a channel a lower level cannot reach:
the approval surface (the approve command, the token files, and the approval
environment variable) is reachable only at L2 or above and requires an interactive
operator channel that the model's own non-interactive shell does not have. There
is no environment break-glass a lower level could set to raise itself; role, not
identity, sets level.

**Layer 3, classification.** The compass and command scanning are defense in depth
above the floor. We treat classification as an evadable denylist by construction,
and rely on the kernel floor, not the scanner, to protect the approval store.

**The single-use approval token.** A PAUSE mints a token bound to a hash of the
evaluated input, written only to an operator-readable 0600 file, never to standard
output or model-visible text. Consumption is atomic under an `flock`-guarded
compare-and-swap: given many processes racing one token, exactly one succeeds, and
a token presented for a mismatched action is refused and burned. This is one
invariant — a PAUSE approval authorizes exactly one action exactly once — that had
previously surfaced as three separate bugs (a burn bug, a double-fire, a consume
race) before it was stated as a single property.

## 5. Method: four-seat adversarial review and a red-team battery

The findings come from a review method in which independent seats catch failures
the others structurally cannot: a documentation seat drafting the spec from public
material, a source seat verifying claims against implementation and, for the
cockpit under test, against its Rust source, a runtime seat running live tests, and
an independent red-team battery. This extends the cross-architecture method of the
author's prior IRIS Gate work; we are careful to call it engineering adversarial
review, not a claim about model cognition.

The method's value is concrete. Every real defect this project found lived where a
single seat would have missed it and where a design-level model checker could not
see it: an EDIT newline-serialization collision (a representation bug), an
`fcntl`-absent path that silently turned the atomic token claim back into the race
it prevents (a platform bug), a cockpit silently overriding an "ask"-style decision
(a harness bug), read-capable tools skipping evaluation entirely (a classifier
bug), and a citation that resolved to an unrelated paper (a provenance bug). None
of these is a protocol-logic bug. This shaped our position on formal verification
(Section 8).

The red-team battery is organized by invariant, not by module. Each layer asserts
one property and tries to break it; a failing attack test means the attack
succeeded. The battery scores the gate contract in isolation — it invokes the gate
as a subprocess and scores its standard output — so it measures the classification
layer, not the kernel floor. This separation is deliberate and is reported as such.

## 6. Conformance and results

We report Attack Success Rate per class against the battery. At the reference
release (v0.9.4), the overall gate-contract ASR is **3.3%** (5 of 150 scored
probes), down from 10.7% before the kernel floor and the classifier fixes.

Every class scores 0% except one. `sentinel_forgery` is 0/75 across sentinel
material, environment interpolation, format specifiers, shell metacharacters,
CRLF/newline/null-byte/unicode-lookalike/ANSI/JSON-breakout injection in five
envelope fields each; `wrapper_bypass` is 0/4 (a guessed-nonce stub, a chatty gate
embedding a valid sentinel, a hung gate, a nonzero-exit gate all deny);
`fail_open`, `env_disclosure`, and `resource_abuse` are 0; `classifier_bypass`
0/17; `malformed_input` 0/14; `nonce_bypass` 0/10; `deployment_gap` 0/1.

The single residual is `privilege_escalation` at 21.7% (5 of 23): shell-indirection
reads of the approval store (`cd ~/.cosmic-cli && cat *`, `tar`, `cp -r`, `find
-exec cat`). The battery scores these at the gate deliberately, to demonstrate that
command classification is an evadable denylist and that the kernel floor is the
actual control. With the floor active, those token files are kernel-denied
regardless of how the command is spelled; on macOS this is runtime-verified (the
reference launcher's floor canary plants a token and confirms a Seatbelt-sandboxed
process is denied the read), and the cockpit's own end-to-end test (Grok Build's
`deny_paths_e2e.rs`) independently confirms that a shelled-out read of a denied
path cannot read the bytes. We report the gate-contract number and the floor result as two separate
measurements, and we do not claim 0% on `privilege_escalation` at the gate layer,
because that class is structurally the floor's.

## 7. Limitations and honest disclosures

We are deliberate about what is not claimed.

Mediation is partial by construction. The gate mediates only actions the cockpit
routes through its hooks; a cockpit that ignores the sentinel is constrained only
by the kernel floor. The timeout hole is structural: a wrapper cannot catch its
own kill, and the kernel floor, not the wrapper, is the control for that residue.
This is why we describe the system as a fail-closed proof-of-allow gate with a
kernel-enforced approval floor, and not as a complete-mediation reference monitor.

Platform coverage is asymmetric. The floor is runtime-verified on macOS Seatbelt
and source-and-test-verified for Linux (Landlock plus bubblewrap), not yet
runtime-tested on the author's own Linux hardware; closing that gap is the
immediate next step.

The conformance battery's per-class probe counts are uneven (some classes have as
few as one or two probes), so the battery is best read as one conforming
test-vector set rather than the definition of conformance; the invariants and a
probe schema, not a single aggregate number, are the intended normative bar.

Cross-model review, where used, is advisory only: a second model's read of a PAUSE
action is input to the human, never a consumer of the token, because
LLM-as-judge outputs are not authoritative. The gate requires POSIX `fcntl`
(Windows is unsupported), and we make no formal claim about the isolation of the
nonce channel beyond the empirical battery.

## 8. Future work

Three directions follow, in order. First, runtime-verify the kernel floor on Linux
and on an edge target (an NVIDIA Jetson), which retires the platform-asymmetry
limitation and proves the non-Apple leg. Second, extract the battery into a
cockpit-agnostic conformance kit with a normative probe schema, grow the thin
probe classes, and recruit a second independent gate implementation, since a
runnable conformance kit, not prose, is what recruits implementers. Third, and
scoped narrowly, a machine-checked model of the single-use-token invariant (the
one property whose concurrency is subtle) bound to the implementation by
property-based stateful testing; we scope this to the token invariant rather than
"the gate core," because this project's own bug history shows the risk lives at
representation, platform, and harness seams a design-level model checker cannot
see.

## Acknowledgments

This work was authored solely by Anthony J. Vasquez Sr., who is the sole author
and sole copyright holder of this paper, of the COSMIC-ALLOW specification, and of
the cosmic-cli reference implementation.

The research and its artifacts were developed with the assistance of AI systems
used as tools under the author's direction and final review: **Claude (Anthropic)**
contributed to specification drafting, source-level verification, and adversarial
code review; **Grok and Grok Build (xAI)** contributed to implementation and live
runtime testing, and Grok Build is the open-source cockpit under test;
**ChatGPT (OpenAI)** contributed an independent review perspective. An independent
automated red-team battery was used to measure the reported Attack Success Rate.

These acknowledgments record the use of these tools and the collaborative method
by which the work was produced. They do not constitute, and shall not be construed
as, a claim of authorship, inventorship, or copyright by any AI system or by its
provider (Anthropic, xAI, or OpenAI). All authorship and all rights vest solely in
the named human author. This disclosure is made in the interest of research
transparency and to state the human-authorship basis of the work plainly for the
record.

This work extends the cross-architecture method of the author's prior IRIS Gate
research.

## References

1. Anderson, J. P. (1972). *Computer Security Technology Planning Study.* (The
   reference-monitor concept: always invoked, tamperproof, small enough to verify.)
2. Zhang et al. (2024). *Agent Security Bench (ASB).* arXiv:2410.02644. (ASR
   definition.)
3. OWASP GenAI Security Project (2025). *OWASP Top 10 for Agentic Applications
   (2026).*
4. Microsoft (2026). *Agent Governance Toolkit / Agent OS policy engine.*
5. Bradner, S. (1997). *RFC 2119*; Leiba, B. (2017). *RFC 8174.*
6. xai-org/grok-build (2026). Open-source coding cockpit under test. Apache-2.0.
7. Vasquez, A. J., Sr. (2026). *Entropic Relational Computing: The Universal
   Alignment Attractor* (IRIS Gate). OSF, 10.17605/OSF.IO/T65VS.
8. COSMIC-ALLOW RFC v1.1 and cosmic-cli reference implementation (this deposit),
   Apache-2.0.

*Citations marked with venue/date are anchors verified where possible at draft
time; final citation verification is a pre-publication review item.*
