"""cosmic-cli gate --hook {grok,claude} — the COSMIC-ALLOW sentinel gate (box 1).

Reads a cockpit PreToolUse envelope on stdin, classifies the tool call, runs it
through the FROZEN compass (policy rules from COSMIC.md, then check_shell for the
shell class; MCP and unknown tools are deny-by-default in the classifier BEFORE
the rule engine, because evaluate_rules is fail-open on no-match). Emits the
RFC v1.1 proof-of-allow sentinel `COSMIC-ALLOW v1 <nonce>` on stdout ONLY for a
genuine OPEN decision, and nothing on stdout otherwise.

Stdout contract (RFC v1.1, mandatory): stdout is the sentinel line on OPEN and
empty otherwise; every diagnostic goes to stderr; stdin is opaque bytes parsed as
JSON DATA and never shell-interpolated, format-interpolated, eval'd, or reflected
to stdout; no code path prints the environment. The nonce is read from the
COSMIC_GATE_NONCE env var only — never from the payload — so no model-controlled
byte can forge or provoke a valid sentinel.

Greenfield module: it CALLS the frozen avionics (policy, rules, shell_guard); it
does not modify them. Checkpointing, PAUSE-token issuance, and the Helix witness
are later boxes; box 1 is envelope -> classify -> policy decision -> sentinel.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from cosmic_cli.gateway import ApprovalManager
from cosmic_cli.policy import ActionType, Disposition, evaluate_rules
from cosmic_cli.rules import load_rules_from_markdown
from cosmic_cli.secrets import is_sensitive_path
from cosmic_cli.shell_guard import check_shell

SENTINEL_VERSION = "v1"
_NONCE_RE = re.compile(r"[0-9a-f]{32,}")  # used with fullmatch: no trailing newline

# Tool-name classification. Claude names and their Grok aliases both map here
# (a matcher keeps its original name too), so one table covers both cockpits.
_SHELL = {"Bash", "run_terminal_command"}
_MUTATE = {"Write", "Edit", "MultiEdit", "Create", "search_replace", "write", "create"}
_READ = {"Read", "read_file", "Grep", "grep", "Glob", "ListDir", "list_dir"}
# Genuinely inert: no local filesystem read, so no sensitive-path exposure.
# NB: read-capable tools (Grep/Glob/ListDir) are NOT here — they READ file
# content/paths, so they route through _READ and the sensitive-path refusal.
# The allowlist used to enumerate NAMES; the property that matters is READS.
_INERT = {"WebSearch", "web_search"}


def _reason(msg: str) -> None:
    """Diagnostics to stderr ONLY — never stdout (would break the contract)."""
    print(msg, file=sys.stderr)


def _extract(tool_input: dict, *keys: str) -> str:
    for k in keys:
        v = tool_input.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _is_under_approval_store(path: str) -> bool:
    """The L2-only approval store (~/.cosmic-cli) is off-limits to L0 READ/WRITE
    tools regardless of which file inside it is named. A read-capable tool
    (Grep/Glob/ListDir) pointed at the store is the mutation-door shape: the
    allowlist enumerated NAMES, but the property that matters is whether it
    READS. Resolved-path containment, not a substring on a shell string, so it
    cannot be spelled around (that class is the kernel floor's job, not this)."""
    try:
        store = (Path.home() / ".cosmic-cli").resolve()
        target = Path(path).expanduser().resolve()
    except Exception:
        return False
    return target == store or store in target.parents


class _Deny(Exception):
    """Internal: a deny with a stderr reason. Never carries payload to stdout."""


def classify(tool_name: str, tool_input: dict) -> Tuple[str, ActionType | None, str]:
    """Return (kind, action_type, match_corpus). kind in {allow, gate, deny}.

    Deny-by-default: an MCP ``server__tool`` or any unrecognized name is denied by
    the classifier, before any rule engine runs.
    """
    if tool_name in _INERT:
        return ("allow", None, "")
    if tool_name in _SHELL:
        return ("gate", ActionType.SHELL, _extract(tool_input, "command", "cmd", "script"))
    if tool_name in _MUTATE:
        path = _extract(
            tool_input, "path", "file_path", "filePath", "target_file"
        )
        body = _extract(tool_input, "content", "new_string", "new_str", "text")
        return ("gate", ActionType.WRITE, f"{path}\n{body}")
    if tool_name in _READ:
        return (
            "gate",
            ActionType.READ,
            _extract(
                tool_input, "path", "file_path", "filePath", "target_file"
            ),
        )
    # MCP qualified name (server__tool) or anything unknown: deny-by-default.
    return ("deny", None, "")


def decide(envelope: dict, rules, exec_mode: str = "safe") -> Optional[str]:
    """Return the disposition path's OPEN reason, or raise _Deny. Never touches stdout.

    Returns None on OPEN (caller emits the sentinel); raises _Deny otherwise.
    """
    tool_name = envelope.get("toolName") or envelope.get("tool_name") or ""
    # A structurally malformed envelope denies; it is never repaired into a
    # permissive one (RFC v1.1: malformed input and malformed output both deny).
    # A truly absent toolInput is the legitimate no-arg case and becomes {};
    # a PRESENT toolInput that is not an object (string/list/number/null) denies.
    if "toolInput" in envelope:
        tool_input = envelope["toolInput"]
    elif "tool_input" in envelope:
        tool_input = envelope["tool_input"]
    else:
        tool_input = {}
    if not isinstance(tool_input, dict):
        raise _Deny("malformed envelope: toolInput is not an object")

    kind, action_type, corpus = classify(str(tool_name), tool_input)
    if kind == "allow":
        return None
    if kind == "deny":
        raise _Deny(f"deny-by-default: unclassified/MCP tool {str(tool_name)!r}")

    # Sensitive-path refusal for the read/write classes (unification table).
    if action_type in (ActionType.READ, ActionType.WRITE):
        path = _extract(
            tool_input, "path", "file_path", "filePath", "target_file"
        )
        if path and (
            is_sensitive_path(path)
            or is_sensitive_path(Path(path).name)
            or _is_under_approval_store(path)
        ):
            raise _Deny(f"sensitive-path {action_type.value} refused")

    decision = evaluate_rules(rules, action_type, corpus)
    if decision.disposition == Disposition.WITNESS:
        raise _Deny(f"WITNESS ({action_type.value}): blocked by compass rule")

    # Shell blocklist BEFORE any PAUSE-token handling, so a PAUSE token is never
    # minted or consumed for a command check_shell will reject anyway (the
    # token-burn lesson from the agent loop, applied at the seam).
    if action_type in (ActionType.SHELL, ActionType.CODE):
        blocked = check_shell(corpus, exec_mode=exec_mode)
        if blocked:
            raise _Deny(f"check_shell: {blocked}")

    if decision.disposition == Disposition.PAUSE:
        return _pause_at_seam(action_type, decision.evaluated_input_sha256)

    return None  # genuine OPEN


def _write_operator_pause_token(tok: str, channel: str, action_sha: str) -> None:
    """Write the PAUSE token to the operator-only 0600 file — same format the
    agent loop uses, so `cosmic-cli helix show-pause-token` reads it. The token
    NEVER touches stdout or a model-visible reason (confused-deputy lock)."""
    from datetime import datetime, timezone

    store = Path.home() / ".cosmic-cli" / "last_pause_token.json"
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(store.parent, 0o700)
        except OSError:
            pass
        payload = {
            "token": tok,
            "channel": f"gate-{channel}",
            "action_sha256": action_sha,
            "minted_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = store.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(store)
        try:
            os.chmod(store, 0o600)
        except OSError:
            pass
    except Exception:
        pass  # best-effort operator channel; the deny still fires


def _pause_at_seam(action_type: ActionType, action_sha: str) -> Optional[str]:
    """Bridge PAUSE (box 4). An operator-approved token (from COSMIC_APPROVAL_TOKEN,
    which only the operator sets — never the model) is consumed exactly-once via
    the frozen claim_once and yields OPEN. Otherwise a token is minted to the
    operator-only channel and the action is denied. The token is action-bound
    (evaluated_input_sha256), single-use, and never model-visible."""
    tok = os.environ.get("COSMIC_APPROVAL_TOKEN", "").strip()
    if tok:
        try:
            if ApprovalManager().claim_once(tok, action_sha):
                return None  # operator-approved retry -> OPEN
        except Exception:
            pass  # fail-closed: fall through to re-mint + deny
    try:
        minted = ApprovalManager().mint_token(action_sha)
        _write_operator_pause_token(minted, action_type.value, action_sha)
    except Exception:
        raise _Deny(f"PAUSE ({action_type.value}): cannot mint approval token")
    raise _Deny(
        f"PAUSE ({action_type.value}): human approval required — "
        "operator: cosmic-cli helix show-pause-token"
    )


def _load_rules(cwd: str):
    root = Path(cwd) if cwd else Path.cwd()
    cosmic_md = root / "COSMIC.md"
    if cosmic_md.is_file():
        try:
            return load_rules_from_markdown(cosmic_md)
        except Exception as e:  # malformed policy -> fail closed
            raise _Deny(f"COSMIC.md failed to load (fail-closed): {e}")
    return []


def run_gate(hook: str = "grok", verb_check: bool = False,
             stdin: Optional[str] = None, exec_mode: str = "safe") -> int:
    """Entry point. Returns process exit code. Emits the sentinel on OPEN only.

    stdout is written to EXACTLY once, and only the sentinel line, and only on a
    genuine OPEN. All other outcomes leave stdout empty (diagnostics to stderr).
    """
    if verb_check:
        return 0  # the wrapper's existence probe

    nonce = os.environ.get("COSMIC_GATE_NONCE", "")
    if not _NONCE_RE.fullmatch(nonce):
        _reason("gate: missing/malformed COSMIC_GATE_NONCE — cannot prove allow")
        return 2

    raw = stdin if stdin is not None else sys.stdin.read()
    try:
        envelope = json.loads(raw)
        if not isinstance(envelope, dict):
            raise ValueError("envelope is not a JSON object")
    except Exception as e:
        _reason(f"gate: unparseable envelope (fail-closed): {e}")
        return 2

    try:
        rules = _load_rules(str(envelope.get("cwd") or envelope.get("workspaceRoot") or ""))
        decide(envelope, rules, exec_mode=exec_mode)
    except _Deny as d:
        _reason(f"[BLOCKED] {d}")
        return 0  # deny is signaled by ABSENCE of the sentinel on stdout
    except Exception as e:  # anything unexpected -> fail closed, no sentinel
        _reason(f"gate: internal error (fail-closed): {type(e).__name__}: {e}")
        return 0

    # Genuine OPEN: the ONLY stdout write in the whole module.
    sys.stdout.write(f"COSMIC-ALLOW {SENTINEL_VERSION} {nonce}\n")
    return 0
