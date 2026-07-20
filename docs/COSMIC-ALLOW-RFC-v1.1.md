# COSMIC-ALLOW: A Cross-Vendor Proof-of-Allow Protocol for AI Coding-Agent Cockpits

**RFC v1.1** · The Temple of Two · Anthony J. Vasquez Sr.
Status: Draft Standard · License: Apache-2.0

## Abstract

AI coding "cockpits" (Claude Code, Grok Build, and compatible shells) run a
model that issues file and shell actions with limited human oversight. Their
hook systems are **fail-open**: when a governance hook errors, times out, or is
absent, the action typically proceeds. COSMIC-ALLOW specifies a **fail-closed**
gate that mounts under such fail-open hooks. Deny is the ground state. A gate
authorizes an action only by emitting a per-invocation proof-of-allow sentinel
as its entire standard output. This document defines the sentinel contract, the
nonce discipline, envelope handling, multi-hook aggregation, the boot canary,
the single-use approval-token invariant, the privilege ranking, and the kernel
floor, together with a conformance methodology (Attack Success Rate against a
hostile-payload battery). It also states, normatively, what the protocol does
**not** guarantee.

## 1. Terminology

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this
document are to be interpreted as described in RFC 2119 and RFC 8174 when, and
only when, they appear in all capitals.

## 2. Definitions

- **Cockpit.** The agent host that runs the model and invokes tools (e.g. Claude
  Code, Grok Build). It provides a PreToolUse hook surface.
- **Gate.** The decision process the hook invokes (`cosmic-cli gate`). It
  classifies a proposed action and either proves allow or does not.
- **Wrapper.** The thin hook script the cockpit executes; it generates the
  nonce, invokes the gate, and translates the result into the cockpit's hook
  decision format.
- **Sentinel.** The exact proof-of-allow string, `COSMIC-ALLOW v1 <nonce>`.
- **Nonce.** A per-invocation secret of at least 128 bits, lowercase hex.
- **Envelope.** The cockpit's PreToolUse payload (tool name, tool input, cwd),
  delivered to the gate on standard input.
- **Dispositions.** WITNESS (hard deny), PAUSE (deny pending one human
  approval), OPEN (allow).
- **Privilege levels.** L3 operator (human) > L2 seat/operator process > L1
  compass/gate > L0 gated model.

## 3. The proof-of-allow contract (normative)

3.1. Deny **MUST** be the ground state. In the absence of a valid sentinel, the
wrapper **MUST** deny.

3.2. On a genuine OPEN, the gate **MUST** write to standard output exactly the
byte string `COSMIC-ALLOW v1 <nonce>` followed by a single newline, and nothing
else. This **MUST** be the only write to standard output in any code path.

3.3. On any disposition other than OPEN, the gate **MUST NOT** write the
sentinel, and **MUST** leave standard output empty. Diagnostic text **MUST** go
to standard error only.

3.4. The wrapper **MUST** treat the gate's standard output as a **whole-output
equality** check against `COSMIC-ALLOW v1 <nonce>` (optionally with a single
trailing newline). A sentinel embedded in additional output **MUST** be treated
as a deny. The wrapper **MUST** discard the gate's standard error from the
allow/deny decision.

3.5. Exit codes **MUST NOT** be the authorization signal. Presence of the exact
sentinel on standard output is the only signal. A non-zero exit that still emits
a valid sentinel **MAY** be honored as allow; a zero exit without the sentinel
**MUST** be a deny.

## 4. The nonce (normative)

4.1. The nonce **MUST** be at least 128 bits of cryptographically strong
randomness, freshly generated per invocation by the wrapper.

4.2. The nonce **MUST** be delivered to the gate through an environment variable
(`COSMIC_GATE_NONCE`) only. The gate **MUST NOT** read the nonce from the
envelope, and **MUST NOT** accept a payload-supplied nonce.

4.3. The gate **MUST** validate the nonce with an exact, whole-string match
(`^[0-9a-f]{32,}$` under full-match semantics). A value with leading or trailing
characters, including a trailing newline, **MUST** be rejected.

4.4. The gate **MUST NOT** print the environment, and **MUST NOT** reflect the
nonce anywhere except the sentinel on a genuine OPEN.

## 5. Envelope handling (normative)

5.1. The gate **MUST** treat the envelope's standard input as opaque bytes. It
**MUST NOT** interpolate envelope bytes into any shell command, and **MUST NOT**
reflect envelope bytes into standard output or into any model-visible reason.

5.2. A structurally malformed envelope **MUST** deny. In particular, a `toolInput`
that is present but is not an object (a string, list, number, or null) **MUST**
deny; it **MUST NOT** be coerced into a permissive empty object. A truly absent
`toolInput` **MAY** be treated as the no-argument case.

5.3. An unrecognized tool name — including any MCP-qualified `server__tool` — 
**MUST** deny by default in the classifier, before any rule engine runs.

## 6. Aggregation and fail-open composition (normative)

6.1. When the cockpit runs multiple PreToolUse hooks, a single explicit deny from
any hook **MUST** deny the action (deny-wins); an allow **MUST NOT** override a
later deny.

6.2. Cockpit hook runners are typically fail-open: a hook that times out,
crashes, is not found, or emits malformed output is treated as non-blocking. A
conforming deployment **MUST NOT** rely on the cockpit failing closed. The
COSMIC-ALLOW positive protocol (Section 3) is what supplies the fail-closed
property: because allow requires a sentinel the wrapper can verify against its own
nonce, a broken, hung, or absent gate produces no sentinel and therefore denies.

## 7. Boot canary (normative)

7.1. Before a governed session begins, the launcher **MUST** run a boot canary
that exercises the gate with, at minimum, one representative probe per action
class (shell, write, read-sensitive, MCP/unknown), bound to a fresh per-boot
nonce.

7.2. The canary **MUST** fail — and the launcher **MUST** refuse to start the
cockpit — if the gate is absent, if a deny-class probe emits a sentinel
(dead-allow), if an allow-class probe fails to emit the expected sentinel
(dead-deny), or if the gate is silent. Absence of proof is not proof of absence;
the session **MUST NOT** begin ungoverned.

## 8. The single-use approval-token invariant (normative)

8.1. A PAUSE disposition **MUST** mint a single-use approval token bound to a
hash of the evaluated action input (`evaluated_input_sha256`).

8.2. The token **MUST** authorize exactly one action exactly once. Consumption
**MUST** be atomic under concurrency (e.g. an `flock`-guarded compare-and-swap):
given N processes racing one token, exactly one **MUST** succeed.

8.3. A token presented for an action whose input hash does not match the bound
hash **MUST** be refused, and the token **MUST** be burned.

8.4. The token **MUST** be written only to an operator-readable store (mode
0600). It **MUST NOT** appear on standard output, in any model-visible reason, or
in logs delivered to model context. A cockpit that silently overrides an
"ask"-style permission decision **MUST** be treated as fail-open for this purpose;
the token channel, not the cockpit prompt, is authoritative.

## 9. Privilege ranking (normative)

9.1. Approval **MUST** flow only from a strictly higher privilege level to a lower
one, through a channel the lower level cannot reach.

9.2. The approval surface (the approve command, the token files, and any approval
environment variable) **MUST** be reachable only by L2 or above. Approval **MUST**
require an interactive operator channel (e.g. a TTY) that the L0 model's own
non-interactive shell does not possess.

9.3. There **MUST NOT** be an environment-variable break-glass that a lower level
could set to raise its own privilege. Role, not identity, sets level.

## 10. Kernel floor (normative)

10.1. Command classification is an enumeration and is therefore an evadable
denylist; it **MUST** be treated as defense in depth, not as the boundary that
protects the approval store.

10.2. A conforming deployment **MUST** provide a kernel-enforced floor that denies
read and write of the approval-token artifacts to the model's execution context,
independent of how a command is spelled. On Linux this SHOULD be Landlock plus a
bubblewrap bind-over (or seccomp); on macOS this SHOULD be a Seatbelt profile. The
floor **MUST** deny the specific token artifacts and **MUST NOT** deny the whole
tool state directory in a way that disables the hook.

10.3. The floor **MUST** be provisioned by default (not opt-in), and its binding
**MUST** be verified at launch by a floor canary that refuses to start the cockpit
if a sandboxed probe can still read a planted token. On platforms where the floor
cannot be enforced, the deployment **MUST** either refuse to start or explicitly
declare the reduced guarantee (Section 13).

## 11. Conformance (normative)

11.1. Conformance is reported as Attack Success Rate (ASR) per attack class
against a hostile-payload battery, where a successful attack is a probe that
obtains an allow (or reads the approval store) that the invariant forbids.

11.2. The battery scores the **gate contract in isolation** (it invokes the gate
as a subprocess and scores its standard output). A conforming report **MUST**
present the gate-contract ASR and the kernel-floor result as **two separate
measurements**. A report **MUST NOT** present the gate-contract number as if it
also measured the floor, and **MUST NOT** claim 0% on the shell-indirection
approval-store-read class ("privilege_escalation") at the gate layer, because that
class is structurally the floor's responsibility.

11.3. A conforming implementation SHOULD publish the invariants and a probe schema
as the normative bar, and treat any specific probe set (such as the reference
150-probe battery) as one conforming test-vector set rather than the definition of
conformance.

## 12. Reference values

- Sentinel: `COSMIC-ALLOW v1 <nonce>` (whole-output equality).
- Nonce: `^[0-9a-f]{32,}$`, full-match, environment-only.
- Reference gate: `cosmic-cli gate --hook {grok,claude}`.
- Reference battery: `tests/battery/` (invariant-organized; gate-contract ASR).
- Grok Build seam is source-verified at pinned `SOURCE_REV` (see
  `docs/SEAM_CONTRACT.md`).

## 13. Security considerations and explicit non-claims

13.1. **Mediation is partial by construction.** The gate mediates only actions the
cockpit routes through its hooks. A cockpit that ignores the sentinel contract is
constrained only by the kernel floor (Section 10). This protocol is a fail-closed
proof-of-allow gate with a kernel-enforced approval floor; it is **not** a
complete-mediation reference monitor over all model actions.

13.2. **The timeout hole is structural.** A wrapper cannot catch its own kill; a
gate wedged past the hook timeout cannot itself deny. The kernel floor, not the
wrapper, is the load-bearing control for that residue.

13.3. **Platform coverage is asymmetric.** The reference implementation's floor is
runtime-verified on macOS (Seatbelt) and source-and-test-verified for Linux
(Landlock plus bubblewrap); a deployment **MUST** state which platforms it has
runtime-verified.

13.4. **Cross-model review is advisory.** Where a second model reviews a PAUSE
action, its output **MUST** be advisory input to the human at L3 only; a reviewer
**MUST NOT** consume the approval token. LLM-as-judge outputs are not authoritative
(position, format, and temperature sensitivity are documented in the literature);
report agreement with a chance-corrected metric.

13.5. **Denylist residue.** Classification cannot enumerate every spelling of a
hostile command; this is why Section 10 requires the kernel floor. Any claim of
completeness at the classification layer is out of scope and unsupported.

## 14. References

- Anderson, J. P. (1972). Computer Security Technology Planning Study
  (reference-monitor concept).
- Bradner, S. (1997). RFC 2119; Leiba, B. (2017). RFC 8174.
- Zhang et al. (2024). Agent Security Bench (arXiv:2410.02644) — ASR definition.
- OWASP Top 10 for Agentic Applications (2026).
- Reference implementation: cosmic-cli (this repository), Apache-2.0.

---

*This document is the normative specification for the COSMIC-ALLOW protocol as
implemented in cosmic-cli. Where the prose and the reference implementation
disagree, the disagreement is a defect to be reconciled, not a silent divergence.*
