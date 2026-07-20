"""Pillar 2 — the load-bearing proof of the PAUSE-token exactly-once invariant.

A Hypothesis ``RuleBasedStateMachine`` drives random sequences of operations
(mint / validate / claim / claim-wrong-action / expire) against the REAL
``ApprovalManager`` — real ``flock``, real JSON store on a real temp path — and
asserts every outcome against a tiny reference model. The invariant
"**exactly one action, exactly once**" is proved structurally, in every
interleaving Hypothesis can find, not by a handful of hand-picked examples.

Why against the real object and not a model checker (paper Sec. 8): every real
bug in this project lived at a platform/representation seam a model checker
assumes away. The exactly-once guarantee rests on ``fcntl`` flock; TLC's
atomicity-by-fiat would *hide* the one concurrency defect. So the two seams a
checker cannot see get their own real tests here:
  * ``test_claim_refuses_without_fcntl`` — no lock => refuse, never race.
  * ``test_concurrent_claim_is_exactly_once`` — N real processes race one
    token through the real flock; exactly one wins.

Wire: this file is the required CI gate (.github/workflows/ci.yml). A change
that breaks the exactly-once contract fails the build.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

import cosmic_cli.gateway as gateway
from cosmic_cli.gateway import ApprovalManager

try:
    import fcntl as _fcntl  # noqa: F401

    HAS_FCNTL = True
except ImportError:  # pragma: no cover — Windows
    HAS_FCNTL = False

requires_fcntl = pytest.mark.skipif(
    not HAS_FCNTL, reason="fcntl (flock) unavailable — exactly-once cannot hold"
)

# A small fixed pool of 64-hex action bindings, so a "claim with the wrong
# action sha" actually collides against real minted tokens often enough to
# exercise the wrong-action burn branch.
SHAS = ["a" * 64, "b" * 64, "c" * 64]
_shas = st.sampled_from(SHAS)


@requires_fcntl
class ApprovalStoreMachine(RuleBasedStateMachine):
    """Model the store as {token_id: {sha, used, expired}} and assert the real
    manager agrees on every operation's return value and resulting state."""

    tokens = Bundle("tokens")

    def __init__(self) -> None:
        super().__init__()
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        store = Path(self._tmp.name) / "local_approvals.json"
        self.am = ApprovalManager(store_path=store)
        self.model: dict[str, dict] = {}
        self.claim_true_count: dict[str, int] = {}

    # --- rules ------------------------------------------------------------
    @initialize(target=tokens, sha=_shas)
    def seed(self, sha):
        # Guarantee a non-empty bundle from step one, so token-consuming rules
        # fire immediately instead of being filtered out until the first mint.
        tok = self.am.mint_token(sha)
        self.model[tok] = {"sha": sha, "used": False, "expired": False}
        self.claim_true_count[tok] = 0
        return tok

    @rule(target=tokens, sha=_shas)
    def mint(self, sha):
        tok = self.am.mint_token(sha)
        assert tok not in self.model, "mint returned a colliding token id"
        self.model[tok] = {"sha": sha, "used": False, "expired": False}
        self.claim_true_count[tok] = 0
        return tok

    @rule(target=tokens, sha=_shas)
    def mint_already_expired(self, sha):
        tok = self.am.mint_token(sha, ttl_seconds=-1)  # expiry in the past
        assert tok not in self.model
        self.model[tok] = {"sha": sha, "used": False, "expired": True}
        self.claim_true_count[tok] = 0
        return tok

    @rule(tok=tokens, sha=_shas)
    def validate_is_non_consuming(self, tok, sha):
        m = self.model[tok]
        expected = (not m["used"]) and (not m["expired"]) and (m["sha"] == sha)
        assert self.am.validate(tok, sha) is expected
        # validate must NOT mutate: the model is unchanged, and the
        # store_matches_model invariant re-checks that a later claim still holds.

    @rule(tok=tokens, sha=_shas)
    def claim(self, tok, sha):
        m = self.model[tok]
        if m["used"]:
            expected = False
        elif m["expired"]:
            expected = False  # expired => refused BEFORE the burn branch
        elif m["sha"] != sha:
            expected = False  # wrong action => refused AND burned
        else:
            expected = True
        got = self.am.claim_once(tok, sha)
        assert got is expected, f"claim mismatch: got {got}, model {m}, sha {sha[0]}"
        # mirror the real state transition into the model:
        if not m["used"] and not m["expired"]:
            if m["sha"] != sha:
                m["used"] = True  # wrong-action burn
            elif got:
                m["used"] = True  # successful exactly-once consume
        if got:
            self.claim_true_count[tok] += 1

    @rule(sha=_shas)
    def claim_of_unminted_token_is_false(self, sha):
        assert self.am.claim_once("tok-never-minted-sentinel", sha) is False

    # --- invariants -------------------------------------------------------
    @invariant()
    def exactly_once(self):
        for tok, n in self.claim_true_count.items():
            assert n <= 1, f"{tok} claimed True {n}x — exactly-once violated"

    @invariant()
    def store_matches_model(self):
        # Re-read through validate: a used or expired token must never look
        # claimable; an untouched one must. Catches any silent store drift.
        for tok, m in self.model.items():
            claimable = (not m["used"]) and (not m["expired"])
            assert self.am.validate(tok, m["sha"]) is claimable

    def teardown(self):
        self._tmp.cleanup()


ApprovalStoreMachine.TestCase.settings = settings(
    max_examples=100,
    stateful_step_count=25,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
TestApprovalStore = ApprovalStoreMachine.TestCase


# --------------------------------------------------------------------------
# The two seams a model checker cannot see — real, not modeled.
# --------------------------------------------------------------------------

def _claim_worker(store_str: str, tok: str, sha: str) -> bool:
    from pathlib import Path as _P

    from cosmic_cli.gateway import ApprovalManager as _AM

    return _AM(store_path=_P(store_str)).claim_once(tok, sha)


@requires_fcntl
def test_concurrent_claim_is_exactly_once(tmp_path):
    """N separate processes race one valid token through the real flock;
    exactly one may win. This is the guarantee TLC's atomicity would fake."""
    store = tmp_path / "local_approvals.json"
    am = ApprovalManager(store_path=store)
    sha = "d" * 64
    tok = am.mint_token(sha)

    n = 16
    ctx = mp.get_context("fork")  # fork inherits everything; no import-path games
    with ctx.Pool(n) as pool:
        results = pool.starmap(_claim_worker, [(str(store), tok, sha)] * n)

    wins = sum(1 for r in results if r)
    assert wins == 1, f"exactly-once violated under concurrency: {wins} wins in {results}"
    # And the token is now spent: a fresh serial claim also fails.
    assert am.claim_once(tok, sha) is False


@requires_fcntl
def test_claim_refuses_without_fcntl(tmp_path, monkeypatch):
    """No exclusive lock => refuse the claim rather than race unlocked
    (box 4a / conformance). A silently-unlocked store is the fcntl-absent race
    the paper flags; here it must fail closed, not succeed."""
    store = tmp_path / "local_approvals.json"
    am = ApprovalManager(store_path=store)
    sha = "e" * 64
    tok = am.mint_token(sha)  # mint while the lock is available
    monkeypatch.setattr(gateway, "fcntl", None)
    assert am.claim_once(tok, sha) is False
