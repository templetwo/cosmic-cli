# The privilege ranking — approval flows down, never sideways or up

*Anthony's root insight, 2026-07-19: "there has to be a ranking structure
implemented. like you approve the cosmic cli and I approve you." The
self-approval hole in the PAUSE accept-pause channel is one symptom of a missing
principle. This is the principle, and the whole class it closes.*

## The law

**Approval of an action at level N must be issued from a level strictly greater
than N, through a channel that no actor at level ≤ N can reach. No actor approves
its own action, or any action at its own level.**

This is not new security theory — it is the Biba integrity model and the
protection-ring principle: integrity flows downward, an actor can never raise its
own privilege, and a confused deputy is exactly a lower level reaching an
approval that belongs to a higher one. Naming it that way is the point: the fix
for the accept-pause hole is not a patch, it is enforcing a ranking that already
has fifty years of theory behind it.

## The ranking for this system (highest to lowest)

- **L3 — Anthony (human owner).** Approves the seats' outward, irreversible acts:
  `git push`, the DOI deposit, stack-policy enactment. Only he can. (Already
  true: the code seat does not push, deposit, or enact without his go.)
- **L2 — a seat acting as reviewer / a human operator.** Approves cosmic-cli's
  behavior and design: the adversarial verification, the freeze, and PAUSE
  approval at a human/operator TTY. "Claude approves the cosmic cli" lives here.
- **L1 — cosmic-cli / the compass.** Gates L0's actions (WITNESS / PAUSE / OPEN).
  Cannot approve changes to its own governance — COSMIC.md is frozen and
  self-edit is blocked except `--mode full`, which is itself an L2/L3 escalation.
- **L0 — the gated model / agent** (grok-build's model, the Stargazer loop).
  Acts, always gated, approves **nothing**, and cannot reach any approval channel.

**Role sets the level, not identity.** The same Claude that reviews Grok's code at
L2 is L0 the moment it drives `cosmic-cli do` and its shell is gated — and the
compass blocked its `rm -rf` earlier for exactly that reason. So the approval
channel must be unreachable by *whoever is currently acting at L0*, regardless of
who they are. A human at a TTY is L2+; a non-interactive agent shell is L0,
whoever is driving it.

## Why this is the root of the self-approval hole

PAUSE is an L0 actor asking to escalate one action. Its approval must come from
L2, through a channel L0 cannot reach. The hole exists because the approval
channel — the token file and the `accept-pause` / `show-pause-token` commands —
was reachable by L0: the model's gated shell could run `cosmic-cli helix
accept-pause`, which read the token *for* it. The model approved its own level.
That is a ranking violation, and every "TTY-gate this, sandbox that" fix is just a
way of restoring the ranking: make the approval channel L2-only.

The same lens catches the class, not just the instance: any place where a lower
level can reach an approval, a receipt, a policy edit, or a token that belongs to
a higher level is the same bug wearing different clothes.

## Implementation directive (the git-helm fix, reframed)

Do not "patch accept-pause." **Enforce the ranking for the PAUSE approval
channel:**

1. **The approval channel is L2-only by construction.** `accept-pause`,
   `show-pause-token`, and `confirm` require an interactive TTY (`isatty`) and
   refuse otherwise — a non-interactive L0 shell cannot invoke them.
2. **The token store is below the sandbox floor for L0.** Kernel sandbox
   deny-globs on `~/.cosmic-cli/*` (Landlock / Seatbelt) — the load-bearing floor
   the RFC already commits to — so no L0 shell can read the token regardless of
   how the command is classified.
3. **Defense in depth: L1 denies the approval surface.** The compass WITNESS-denies
   any L0 tool call that touches the approval mechanism (the `helix
   accept-pause|show-pause-token|confirm` commands and shell reads of the token
   files). Redundant with 1 and 2 by design.

A conformance test proves the ranking, not just the patch: **an L0 shell action
cannot cause a PAUSE approval** — the exact reproducer (model runs accept-pause,
retries, gets allow) must now return deny.

## The ranking is also the collaboration

"You approve the cosmic cli and I approve you" describes what has been happening
organically: the code seat verifies Grok's builds (L2 approves L1's behavior),
and Anthony approves the pushes and deposits (L3 approves L2's outward acts). The
accept-pause hole is what it looks like when a level approves itself. Making the
ranking explicit — as a standing law and as enforced code — turns a working habit
into a structural guarantee, the same move the convergence discipline made for
the freeze. Approval flows down the ladder. It never flows sideways, and it never
flows up.
