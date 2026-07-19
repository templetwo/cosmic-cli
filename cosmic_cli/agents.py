"""StargazerAgent — coding-interface-grade Grok loop.

Default model: grok-4.5. Filesystem is ground truth; model is the reasoner.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from xai_sdk import Client
from xai_sdk.chat import user

from cosmic_cli.context import ContextManager
from cosmic_cli.action_bind import bind_edit, bind_mkdir, bind_write
from cosmic_cli.checkpoint import CheckpointManager
from cosmic_cli.gateway import ActionGateway, ApprovalManager, ApprovalStoreError
from cosmic_cli.policy import ActionType, Disposition, evaluate_rules
from cosmic_cli.principles import system_prompt_block
from cosmic_cli.rules import load_rules_from_markdown
from cosmic_cli.secrets import deny_read_message, is_sensitive_path, redact
from cosmic_cli.shell_guard import check_shell
from cosmic_cli.tools import (
    looks_like_path_bug,
    parse_edit_payload,
    parse_grep_payload,
    parse_write_payload,
    rel_key,
    tool_create,
    tool_edit,
    tool_glob,
    tool_grep,
    tool_list,
    tool_mkdir,
    tool_write,
)

try:
    from cosmic_cli import helix_bridge
except ImportError:  # pragma: no cover
    helix_bridge = None  # type: ignore

logger = logging.getLogger(__name__)

ECHO_FILE = Path.home() / ".cosmic_echo.jsonl"
SESSION_DIR = Path.home() / ".cosmic-cli" / "sessions"
DEFAULT_MODEL = os.getenv("COSMIC_GROK_MODEL", "grok-4.5")
MAX_STEPS_DEFAULT = 20
CONTEXT_MEMORY_CAP = 16
CONTEXT_CHARS_CAP = 28_000
FILE_TREE_LINES_CAP = 120

STEP_PREFIXES = (
    "GLOB:",
    "GREP:",
    "LIST:",
    "READ:",
    "DIFF:",
    "MKDIR:",
    "CREATE:",
    "EDIT:",
    "WRITE:",
    "SHELL:",
    "CODE:",
    "TEST:",
    "TODO:",
    "INFO:",
    "MEMORY:",
    "FINISH:",
    "PASS:",
)

DISCOVERY_PREFIXES = ("GLOB:", "GREP:", "LIST:", "SHELL:")
MUTATION_PREFIXES = ("MKDIR:", "CREATE:", "EDIT:", "WRITE:")


class StargazerAgent:
    """Tool-using coding agent: search → read → edit → verify → finish."""

    def __init__(
        self,
        directive: str,
        api_key: str,
        ui_callback: Optional[Callable[[str], None]] = None,
        exec_mode: str = "safe",
        work_dir: str = ".",
        model: str = DEFAULT_MODEL,
        max_steps: int = MAX_STEPS_DEFAULT,
        write_echo: bool = True,
        quiet: bool = False,
        show_progress: bool = False,
        session_id: Optional[str] = None,
        auto_verify: bool = True,
        use_helix: bool = True,
        helix_context: str = "",
        api_timeout: float = 120.0,
        approval_token_id: Optional[str] = None,
    ):
        self.directive = directive
        self.client = Client(api_key=api_key, timeout=api_timeout)
        self.ui_callback = ui_callback or (lambda _: None)
        self.use_helix = use_helix and helix_bridge is not None
        self.helix_context = helix_context
        self.exec_mode = exec_mode
        self.root = Path(work_dir).resolve()
        self.context_manager = ContextManager(root_dir=str(self.root))
        # Local policy kernel + gateway (avionics assembly 2026-07-15).
        # Helix compass remains the remote witness; this is the in-process door.
        self.approval_token_id = approval_token_id or os.getenv(
            "COSMIC_APPROVAL_TOKEN"
        )
        try:
            self._approval_mgr = ApprovalManager()
        except ApprovalStoreError as e:
            logger.warning("approval store: %s — in-memory only this process", e)
            self._approval_mgr = ApprovalManager(
                store_path=Path(tempfile.mkdtemp()) / "approvals.json"
            )
        self._policy_rules = None  # lazy-load from COSMIC.md
        try:
            self._checkpoint_mgr = CheckpointManager(self.root)
        except Exception:
            self._checkpoint_mgr = None  # type: ignore
        self._gateway = ActionGateway(
            policy_evaluator=evaluate_rules,
            checkpoint_manager=self._checkpoint_mgr,
            approval_manager=self._approval_mgr,
        )
        self.context_memory: List[str] = []
        self.files_read: Dict[str, str] = {}
        self.files_seen: set[str] = set()  # ever successfully read this mission
        self.files_edited: List[str] = []
        self.todo: List[str] = []
        self.failed_attempts: int = 0
        self.model = model
        self.max_steps = max_steps
        self.write_echo = write_echo
        self.quiet = quiet
        self.show_progress = show_progress and not quiet
        self.auto_verify = auto_verify
        self.status = "ready"
        self.logs: List[str] = []
        self._last_action_key: Optional[str] = None
        self._repeat_count: int = 0
        self._discovery_streak: int = 0
        self.steps_taken: int = 0
        self.warnings: List[str] = []
        # Prefer explicit id → Helix/Claude seat session → mint mission id.
        # Approvals are keyed by (session_id, action_summary); a new mint every
        # `do` strands confirmed tokens across re-runs (PAUSE experiment #2).
        resolved = session_id
        if not resolved and self.use_helix and helix_bridge is not None:
            try:
                resolved = helix_bridge.current_session_id()
            except Exception:
                resolved = None
        self.session_id = resolved or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        # Mission log file: keep a unique path even when reusing Helix session
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.mission_id = f"{self.session_id}__{stamp}" if resolved else self.session_id
        self.session_path = SESSION_DIR / f"{self.mission_id}.jsonl"
        SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # ── logging / memory ──────────────────────────────────────────

    def _log(self, msg: str, *, obs: bool = False) -> None:
        self.logs.append(msg)
        logger.info(msg)
        if self.quiet:
            return
        # Compact: skip noisy obs previews unless they are errors
        if obs and not msg.startswith("[Error]") and "Error" not in msg[:40]:
            # one-line compact observation
            short = msg if len(msg) <= 140 else msg[:137] + "..."
            self.ui_callback(f"  · {short.replace(chr(10), ' ')}")
            return
        self.ui_callback(msg)

    def _add_to_memory(self, text: str, *, label: str = "") -> None:
        self.context_memory.append(text)
        if len(self.context_memory) > CONTEXT_MEMORY_CAP:
            self.context_memory = self.context_memory[-CONTEXT_MEMORY_CAP:]
        preview = text if len(text) <= 160 else text[:157] + "..."
        self._log(f"{label}: {preview}" if label else preview, obs=True)

    def _session_write(self, record: Dict[str, Any]) -> None:
        record = {
            **record,
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
        }
        try:
            with open(self.session_path, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        except OSError:
            pass

    def _load_echo_memory(self) -> List[Dict[str, Any]]:
        if not ECHO_FILE.exists():
            return []
        out: List[Dict[str, Any]] = []
        with open(ECHO_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _append_echo(self, outcome: str, status: str) -> None:
        if not self.write_echo:
            return
        entry = {
            "directive": self.directive,
            "outcome": redact(str(outcome)[:2000]),
            "status": status,
            "model": self.model,
            "steps": self.steps_taken,
            "edited": list(self.files_edited),
            "session": self.session_id,
        }
        try:
            with open(ECHO_FILE, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
            self._log(f"echo → {ECHO_FILE}")
        except OSError as e:
            self._log(f"[warn] echo write failed: {e}")
        # Helix chronicle (local memory substrate)
        if self.use_helix and helix_bridge is not None:
            try:
                content = (
                    f"COSMIC mission [{status}] session={self.session_id}\n"
                    f"directive: {self.directive}\n"
                    f"edited: {', '.join(self.files_edited) or '(none)'}\n"
                    f"outcome: {redact(str(outcome)[:1500])}"
                )
                helix_bridge.record(
                    content,
                    session_id=self.session_id,
                    domain="cosmic-cli",
                    tags=["source:cosmic-cli", f"status:{status}", f"model:{self.model}"],
                    intensity=0.7 if status == "complete" else 0.5,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("helix record failed: %s", e)

    # ── parsing ───────────────────────────────────────────────────

    @staticmethod
    def parse_step(raw: str) -> str:
        if not raw:
            return ""
        text = raw.strip()
        # Strip markdown fences only — NEVER strip trailing " from payloads
        # (that ate closing quotes on EDIT/WRITE lines).
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        candidates: List[str] = []
        for line in text.splitlines():
            candidate = line.strip()
            # strip a single wrapping backtick pair only
            if len(candidate) >= 2 and candidate[0] == "`" and candidate[-1] == "`":
                candidate = candidate[1:-1]
            upper = candidate.upper()
            for prefix in STEP_PREFIXES:
                if upper.startswith(prefix):
                    if prefix in ("EDIT:", "WRITE:", "CREATE:") and "|||" in text:
                        idx = text.upper().find(prefix)
                        body = text[idx + len(prefix) :].lstrip()
                        # keep internal quotes; only rstrip whitespace/newlines
                        candidates.append(f"{prefix} {body}".rstrip("\n\r"))
                    else:
                        body = candidate[len(prefix) :].lstrip()
                        candidates.append(
                            f"{prefix} {body}".rstrip() if body else prefix.rstrip(":")
                        )
                    break
        if not candidates:
            upper = text.upper()
            for prefix in STEP_PREFIXES:
                if upper.startswith(prefix):
                    body = text[len(prefix) :].lstrip()
                    return f"{prefix} {body}".rstrip("\n\r") if body else prefix.rstrip(":")
            return text

        for kind in ("FINISH:", "PASS:"):
            for c in candidates:
                if c.upper().startswith(kind):
                    return c
        return candidates[0]

    # ── context assembly ──────────────────────────────────────────

    def _context_blob(self) -> str:
        parts: List[str] = []
        if self.todo:
            parts.append("TODO:\n" + "\n".join(f"- {t}" for t in self.todo))
        if self.files_edited:
            parts.append("Files edited this mission: " + ", ".join(self.files_edited))
        if self.files_read:
            parts.append("Files in cache (do NOT re-READ unless you edited them):")
            for path, content in list(self.files_read.items())[-6:]:
                body = content if len(content) <= 5000 else content[:5000] + "\n…[truncated]"
                parts.append(f"### {path}\n{body}")
        if self.context_memory:
            parts.append("Recent observations:")
            parts.extend(self.context_memory[-10:])
        blob = "\n\n".join(parts)
        if len(blob) > CONTEXT_CHARS_CAP:
            blob = blob[-CONTEXT_CHARS_CAP:]
        return blob

    def _file_tree(self) -> str:
        # Rescan so CREATE/WRITE land in the tree the model sees next turn
        try:
            self.context_manager.refresh()
        except Exception:
            pass
        tree = self.context_manager.get_file_tree()
        lines = tree.splitlines()
        if len(lines) > FILE_TREE_LINES_CAP:
            return (
                "\n".join(lines[:FILE_TREE_LINES_CAP])
                + f"\n… ({len(lines) - FILE_TREE_LINES_CAP} more — use GLOB/GREP)"
            )
        return tree or "(empty — use GLOB/LIST)"

    def _ask_grok_for_next_step(self) -> str:
        self._log(f"think · {self.model}")
        already = ", ".join(sorted(self.files_read.keys())) or "(none)"
        # Hint relative Desktop when cwd is home
        home_hint = ""
        if self.root.resolve() == Path.home().resolve():
            home_hint = (
                "\nPATH HINT: cwd is HOME. Use Desktop/… not /Users/…/Desktop/… "
                "For folder+file use CREATE: Desktop/folder/file.md|||…"
            )
        helix_block = ""
        if self.helix_context:
            helix_block = f"\n## T2Helix foundational memory\n{self.helix_context}\n"
        prompt = f"""You are Stargazer, a coding agent operating in a real project directory.
The filesystem is ground truth. You change it only through tools.
MEMORY: action queries T2Helix local chronicle (shared with Claude seats).

{system_prompt_block()}
{home_hint}
{helix_block}

DIRECTIVE:
{self.directive}

WORKING DIR: {self.root}
MODEL: {self.model}
MODE: {self.exec_mode}
FILES CACHED: {already}
EDITED: {', '.join(self.files_edited) or '(none)'}
DISCOVERY STREAK: {self._discovery_streak} (if ≥2, stop probing — CREATE/WRITE/EDIT now)

FILE TREE (partial):
{self._file_tree()}

CONTEXT / OBSERVATIONS:
{self._context_blob() or "None yet — LIST or CREATE/WRITE for new files."}

Reply with EXACTLY ONE action line (CREATE/WRITE/EDIT may span lines after |||).
No prose outside the action. Prefer relative paths. Prefer CREATE for new files.
Do not READ .env, *.pem, id_rsa, or credential files.
"""
        try:
            chat = self.client.chat.create(
                model=self.model,
                messages=[user(prompt)],
                temperature=0.0,
            )
            response = chat.sample()
            raw = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )
            return self.parse_step(raw)
        except Exception as e:
            err = str(e)
            if "PERMISSION_DENIED" in err or "spending limit" in err.lower():
                return "PASS: API credits/spending limit — check console.x.ai"
            if "UNAUTHENTICATED" in err or "Invalid API" in err:
                return "PASS: invalid API key"
            self._log(f"[error] model: {err[:200]}")
            return f"PASS: model call failed: {err[:180]}"

    def _ask_grok_for_info(self, question: str) -> str:
        try:
            chat = self.client.chat.create(
                model=self.model,
                messages=[user(question)],
                temperature=0.2,
            )
            response = chat.sample()
            return (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )
        except Exception as e:
            return f"[Error] INFO failed: {e}"

    # ── execute one step ──────────────────────────────────────────

    @staticmethod
    def _body_after(step: str, prefix: str) -> str:
        """Strip action prefix case-insensitively (prefix includes trailing ':')."""
        if len(step) >= len(prefix) and step[: len(prefix)].upper() == prefix.upper():
            return step[len(prefix) :].lstrip()
        # fallback: split once on first colon
        if ":" in step:
            return step.split(":", 1)[1].lstrip()
        return step

    def _execute_step(self, step: str) -> str:
        upper = step.upper()

        if upper.startswith("GLOB:"):
            pattern = self._body_after(step, "GLOB:")
            self._log(f"GLOB {pattern}")
            out = tool_glob(self.root, pattern)
            self._add_to_memory(f"GLOB {pattern}:\n{out}", label="glob")
            return out

        if upper.startswith("GREP:"):
            body = self._body_after(step, "GREP:")
            pattern, path, glob = parse_grep_payload(body)
            self._log(f"GREP /{pattern}/ in {path}" + (f" glob={glob}" if glob else ""))
            out = tool_grep(self.root, pattern, path=path, glob=glob)
            self._add_to_memory(f"GREP /{pattern}/:\n{out}", label="grep")
            return out

        if upper.startswith("LIST:"):
            d = self._body_after(step, "LIST:") or "."
            self._log(f"LIST {d}")
            out = tool_list(self.root, d)
            self._add_to_memory(f"LIST {d}:\n{out}", label="list")
            return out

        if upper.startswith("READ:"):
            raw = self._body_after(step, "READ:").strip("'\"")
            if is_sensitive_path(raw) or is_sensitive_path(Path(raw).name):
                return deny_read_message(raw)
            try:
                file_path = rel_key(raw, self.root)
            except PermissionError as e:
                return f"[Error] {e}"
            if is_sensitive_path(file_path):
                return deny_read_message(file_path)
            if file_path in self.files_read and file_path not in self.files_edited:
                cached = self.files_read[file_path]
                if cached.startswith("[Error]") or cached.startswith("[BLOCKED]"):
                    # don't sticky-cache errors forever — allow retry
                    self.files_read.pop(file_path, None)
                else:
                    self._log(f"READ {file_path} (cache)")
                    return cached
            self._log(f"READ {file_path}")
            content = self.context_manager.read_file(file_path)
            if content.startswith("[Error]"):
                return content  # do not cache errors
            content = redact(content)
            self.files_read[file_path] = content
            self.files_seen.add(file_path)
            self._add_to_memory(
                f"READ {file_path} ({len(content)} chars):\n{content[:8000]}",
                label=file_path,
            )
            return content

        if upper.startswith("EDIT:"):
            body = self._body_after(step, "EDIT:")
            parsed = parse_edit_payload(body)
            if not parsed:
                return "[Error] EDIT format: path|||old_exact|||new  (triple pipe)"
            raw_path, old, new = parsed
            try:
                path = rel_key(raw_path.strip(), self.root)
            except PermissionError as e:
                return f"[Error] {e}"
            blocked_pol = self._block_policy_file_mutation(path)
            if blocked_pol:
                return blocked_pol
            if path not in self.files_seen:
                return (
                    f"[Error] READ-before-EDIT violated for {path}. "
                    "READ the file first, then EDIT with exact old_string."
                )
            if path not in self.files_read:
                self.files_read[path] = self.context_manager.read_file(path)
            self._log(f"EDIT {path}")

            def _do_edit():
                return tool_edit(self.root, path, old, new)

            # Canonical length/hash binding — never raw newline-joined fields
            # (Claude EDIT collision).
            pre = (self.root / path).read_text(encoding="utf-8", errors="replace")
            if old not in pre:
                return f"[Error] EDIT old_string not found in {path}"
            expected_post = pre.replace(old, new, 1)
            binding = bind_edit(
                path=path,
                pre_content=pre,
                old=old,
                new=new,
                post_content=expected_post,
            )
            result = self._run_mutation(
                ActionType.EDIT,
                path,
                binding,
                _do_edit,
                expected_content=expected_post.encode("utf-8"),
                match_input=f"EDIT {path}\n{old}\n{new}",
            )
            if isinstance(result, str):
                return result
            obs, ok = result
            self._add_to_memory(obs, label="edit")
            if ok:
                self.files_edited.append(path)
                self.files_read.pop(path, None)
            return obs

        if upper.startswith("WRITE:"):
            body = self._body_after(step, "WRITE:")
            parsed = parse_write_payload(body)
            if not parsed:
                return "[Error] WRITE format: path|||full_contents"
            raw_path, content = parsed
            try:
                path = rel_key(raw_path.strip(), self.root)
            except PermissionError as e:
                return f"[Error] {e}"
            blocked_pol = self._block_policy_file_mutation(path)
            if blocked_pol:
                return blocked_pol
            target = self.root / path
            if (
                target.exists()
                and path not in self.files_seen
                and self.exec_mode == "safe"
            ):
                return (
                    f"[Error] overwriting {path} requires prior READ in safe mode "
                    "(or use --mode full)."
                )
            self._log(f"WRITE {path}")

            def _do_write():
                return tool_write(self.root, path, content)

            result = self._run_mutation(
                ActionType.WRITE,
                path,
                bind_write(path=path, content=content),
                _do_write,
                expected_content=content.encode("utf-8"),
                match_input=f"WRITE {path}\n{content}",
            )
            if isinstance(result, str):
                return result
            obs, ok = result
            self._add_to_memory(obs, label="write")
            if ok:
                self.files_edited.append(path)
                self.files_seen.add(path)
                self.files_read[path] = content
                if looks_like_path_bug(path):
                    self.warnings.append(f"suspicious path (nested Users?): {path}")
            return obs

        if upper.startswith("MKDIR:"):
            raw = self._body_after(step, "MKDIR:").strip("'\"")
            try:
                path = rel_key(raw, self.root)
            except PermissionError as e:
                return f"[Error] {e}"
            self._log(f"MKDIR {path}")

            def _do_mkdir():
                return tool_mkdir(self.root, path)

            result = self._run_mutation(
                ActionType.WRITE,
                path,
                bind_mkdir(path=path),
                _do_mkdir,
            )
            if isinstance(result, str):
                return result
            obs, ok = result
            self._add_to_memory(obs, label="mkdir")
            return obs

        if upper.startswith("CREATE:"):
            body = self._body_after(step, "CREATE:")
            parsed = parse_write_payload(body)
            if not parsed:
                return "[Error] CREATE format: path|||full_contents"
            raw_path, content = parsed
            try:
                path = rel_key(raw_path.strip(), self.root)
            except PermissionError as e:
                return f"[Error] {e}"
            blocked_pol = self._block_policy_file_mutation(path)
            if blocked_pol:
                return blocked_pol
            self._log(f"CREATE {path}")

            def _do_create():
                return tool_create(self.root, path, content)

            result = self._run_mutation(
                ActionType.WRITE,
                path,
                bind_write(path=path, content=content),
                _do_create,
                expected_content=content.encode("utf-8"),
                match_input=f"CREATE {path}\n{content}",
            )
            if isinstance(result, str):
                return result
            obs, ok = result
            self._add_to_memory(obs, label="create")
            if ok:
                self.files_edited.append(path)
                self.files_seen.add(path)
                self.files_read[path] = content
                if looks_like_path_bug(path):
                    self.warnings.append(f"suspicious path (nested Users?): {path}")
            return obs

        if upper.startswith("DIFF:"):
            raw = self._body_after(step, "DIFF:").strip()
            try:
                path = rel_key(raw, self.root)
            except PermissionError as e:
                return f"[Error] {e}"
            bak = self.root / f"{path}.cosmicbak"
            cur = self.root / path
            if not bak.is_file() or not cur.is_file():
                return f"[Error] no backup/current pair for {path}"
            try:
                result = subprocess.run(
                    ["diff", "-u", str(bak), str(cur)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                out = result.stdout or result.stderr or "(identical)"
            except Exception as e:
                out = f"[Error] diff failed: {e}"
            self._add_to_memory(f"DIFF {path}:\n{out[:4000]}", label="diff")
            return out

        if upper.startswith("SHELL:"):
            cmd = self._body_after(step, "SHELL:")
            self._log(f"SHELL {cmd}")
            return self._run_shell(cmd)

        if upper.startswith("CODE:"):
            code = self._body_after(step, "CODE:")
            self._log(f"CODE ({len(code)} chars)")
            return self._run_code(code)

        if upper.startswith("TEST:"):
            args = self._body_after(step, "TEST:") or "tests/ -q"
            if not args.startswith("pytest") and not args.startswith("python"):
                cmd = f"python -m pytest {args}"
            else:
                cmd = args
            self._log(f"TEST {cmd}")
            return self._run_shell(cmd)

        if upper.startswith("TODO:"):
            body = self._body_after(step, "TODO:")
            try:
                items = json.loads(body)
                if not isinstance(items, list):
                    raise ValueError("not a list")
                self.todo = [str(x) for x in items]
            except (json.JSONDecodeError, ValueError):
                if body:
                    self.todo.append(body)
            obs = "TODO set:\n" + "\n".join(f"- {t}" for t in self.todo)
            self._add_to_memory(obs, label="todo")
            return obs

        if upper.startswith("INFO:"):
            question = self._body_after(step, "INFO:")
            self._log(f"INFO {question[:80]}")
            answer = self._ask_grok_for_info(question)
            self._add_to_memory(f"INFO: {answer}", label="info")
            return answer

        if upper.startswith("MEMORY:"):
            query = self._body_after(step, "MEMORY:")
            # Prefer T2Helix chronicle; fall back to local echo file
            if self.use_helix and helix_bridge is not None:
                hr = helix_bridge.recall(query, top_k=6)
                if hr.get("ok"):
                    items = hr.get("result") or []
                    if isinstance(items, list) and items:
                        chunks = []
                        for m in items[:6]:
                            if isinstance(m, dict):
                                chunks.append(
                                    redact(
                                        (m.get("content") or m.get("text") or str(m))[
                                            :400
                                        ]
                                    )
                                )
                            else:
                                chunks.append(redact(str(m)[:400]))
                        answer = "Helix recall:\n" + "\n---\n".join(chunks)
                        self._add_to_memory(f"MEMORY: {answer}", label="memory")
                        return answer
            memories = self._load_echo_memory()[-20:]
            answer = self._ask_grok_for_info(
                f"Past echoes:\n{json.dumps(memories, ensure_ascii=False)}\n\nQuery: {query}"
            )
            self._add_to_memory(f"MEMORY: {redact(answer)}", label="memory")
            return answer

        if upper.startswith("PASS:"):
            reason = self._body_after(step, "PASS:")
            if self.use_helix and helix_bridge is not None and reason:
                try:
                    helix_bridge.open_thread(
                        reason[:500],
                        domain="cosmic-cli",
                        context=f"session={self.session_id} directive={self.directive[:200]}",
                    )
                    self._log("helix · opened thread for PASS")
                except Exception as e:  # pragma: no cover
                    logger.debug("helix open_thread: %s", e)
            return reason

        if upper.startswith("FINISH:"):
            return self._body_after(step, "FINISH:") or "Done."

        return f"[Error] Unknown command: {step[:100]!r}"

    def _maybe_auto_verify(self) -> Optional[str]:
        """If python files edited, run a cheap compile check."""
        if not self.auto_verify or not self.files_edited:
            return None
        py_files = [f for f in self.files_edited if f.endswith(".py")]
        if not py_files:
            return None
        # unique preserve order
        seen = set()
        uniq = []
        for f in py_files:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        targets = " ".join(f'"{self.root / f}"' for f in uniq[-5:])
        cmd = f"python -m py_compile {targets}"
        self._log(f"auto-verify: {cmd}")
        return self._run_shell(cmd)

    def _recover_from_failure(self, step: str, err: Exception) -> None:
        self._log(f"[error] {err}")
        self.failed_attempts += 1
        if self.failed_attempts < 3:
            retry = self._ask_grok_for_next_step()
            if retry and not retry.upper().startswith("PASS:"):
                try:
                    self._execute_step(retry)
                except Exception as e2:
                    self._log(f"[error] recovery: {e2}")

    def run(self) -> None:
        import threading

        threading.Thread(target=self.execute, daemon=True).start()

    def execute(self) -> Dict[str, Any]:
        self.status = "running"
        self._log(
            f"mission · model={self.model} · mode={self.exec_mode} · "
            f"session={self.session_id}"
        )
        self._session_write({"event": "start", "directive": self.directive, "model": self.model})
        final_result: Dict[str, Any] = {
            "directive": self.directive,
            "model": self.model,
            "session": self.session_id,
            "results": [],
            "status": "running",
            "edited": [],
            "steps_taken": 0,
            "warnings": [],
        }

        def loop(progress=None, task=None) -> None:
            for i in range(self.max_steps):
                self.steps_taken = i + 1
                next_step = self._ask_grok_for_next_step()
                if not next_step:
                    self._log("[warn] empty step")
                    continue

                head = next_step.splitlines()[0][:120]
                self._log(f"→ {i + 1}/{self.max_steps} {head}")
                upper = next_step.upper()
                self._session_write(
                {
                    "event": "step",
                    "n": i + 1,
                    "action": redact(next_step[:2000]),
                }
            )

                # Discovery thrash counter
                if any(upper.startswith(p) for p in DISCOVERY_PREFIXES):
                    # SHELL that mutates (mkdir) counts as discovery-ish still for thrash
                    self._discovery_streak += 1
                elif any(upper.startswith(p) for p in MUTATION_PREFIXES):
                    self._discovery_streak = 0
                if self._discovery_streak >= 2 and not any(
                    upper.startswith(p) for p in MUTATION_PREFIXES + ("FINISH:", "PASS:", "READ:")
                ):
                    wants_new = any(
                        w in self.directive.lower()
                        for w in (
                            "create",
                            "write",
                            "folder",
                            "file",
                            "desktop",
                            "make",
                            "add",
                            "new",
                        )
                    )
                    if wants_new and not self.files_edited:
                        self._add_to_memory(
                            "STEERING: discovery streak ≥2. Stop ls/find. "
                            "Use CREATE: relative/path/file|||contents "
                            "(or MKDIR then WRITE). Prefer Desktop/... if cwd is home.",
                            label="steer",
                        )
                        self._log("steer · CREATE/WRITE now (stop probing)")
                        if progress is not None and task is not None:
                            progress.update(task, advance=1)
                        continue

                # loop breaker for non-terminal repeats (threshold 2 — allow one retry)
                action_key = upper.split(":", 1)[0] + ":" + (
                    next_step.split(":", 1)[1].strip()[:80] if ":" in next_step else ""
                )
                if not upper.startswith(
                    ("FINISH:", "PASS:", "EDIT:", "WRITE:", "CREATE:", "MKDIR:")
                ):
                    if action_key == self._last_action_key:
                        self._repeat_count += 1
                    else:
                        self._last_action_key = action_key
                        self._repeat_count = 0
                    if self._repeat_count >= 1:
                        # First repeat: steer toward mutation if directive needs it
                        wants_change = any(
                            w in self.directive.lower()
                            for w in (
                                "edit",
                                "change",
                                "fix",
                                "update",
                                "add",
                                "create",
                                "write",
                                "rename",
                                "delete",
                                "implement",
                                "fix",
                            )
                        )
                        if wants_change and not self.files_edited:
                            self._add_to_memory(
                                "STEERING: repeated discovery action. "
                                "Use CREATE/EDIT/WRITE now. Do not re-READ cached files.",
                                label="steer",
                            )
                            self._log("steer · mutate (cache warm)")
                            if progress is not None and task is not None:
                                progress.update(task, advance=1)
                            continue
                        if self._repeat_count >= 2:
                            next_step = self._synthesize_finish()
                            upper = next_step.upper()
                            self._log("repeat ×2 → synthesize FINISH")

                if upper.startswith("FINISH:"):
                    # auto-verify before accepting finish if edits exist
                    v = self._maybe_auto_verify()
                    verify_failed = bool(
                        v
                        and v.lstrip().startswith("[exit")
                        and not v.lstrip().startswith("[exit 0]")
                    )
                    if verify_failed:
                        self._add_to_memory(
                            f"VERIFY failed before FINISH:\n{v[:2000]}",
                            label="verify",
                        )
                        final_result["results"].append(
                            {"step": "TEST:auto", "result": v}
                        )
                        self._log("verify failed — continuing (fix it)")
                        if progress is not None and task is not None:
                            progress.update(task, advance=1)
                        continue
                    final_answer = self._body_after(next_step, "FINISH:")
                    if v:
                        clean = "\n".join(
                            ln
                            for ln in v.splitlines()
                            if ln.strip()
                            and not ln.startswith("I0")
                            and "ev_poll" not in ln
                            and not ln.startswith("[exit 0]")
                        )[:500]
                        if clean.strip():
                            final_answer += f"\n[auto-verify] {clean.strip()[:300]}"
                        else:
                            final_answer += "\n[auto-verify: ok]"
                    self._log(f"done · {final_answer[:200]}")
                    final_result["results"].append(
                        {"step": "FINISH", "result": final_answer}
                    )
                    final_result["status"] = "complete"
                    self.status = "complete"
                    if progress is not None and task is not None:
                        progress.update(task, completed=self.max_steps)
                    return

                if upper.startswith("PASS:"):
                    reason = self._body_after(next_step, "PASS:")
                    # execute_step opens helix thread
                    try:
                        self._execute_step(next_step)
                    except Exception:
                        pass
                    final_result["results"].append({"step": "PASS", "result": reason})
                    final_result["status"] = "passed"
                    self.status = "passed"
                    self._log(f"pass · {reason}")
                    if progress is not None and task is not None:
                        progress.update(task, completed=self.max_steps)
                    return

                try:
                    output = self._execute_step(next_step)
                    final_result["results"].append(
                        {"step": next_step.splitlines()[0][:200], "result": output}
                    )
                    # hard-fail signals that should steer the model
                    if isinstance(output, str) and output.startswith("[Error]"):
                        self._log(f"  ! {output[:160]}")
                    # Compass/soft-block: hand up to cockpit, do NOT thrash-retry.
                    # PAUSE/WITNESS already blocked the shell; spinning mints
                    # multiple tokens and strands the first approval.
                    if isinstance(output, str) and output.startswith("[BLOCKED]"):
                        self._log(f"blocked · surfacing to caller (no retry)")
                        final_result["status"] = "blocked"
                        final_result["block_message"] = output
                        self.status = "blocked"
                        # Also stamp as FINISH so callers see the receipt text
                        final_result["results"].append(
                            {"step": "FINISH", "result": output}
                        )
                        if progress is not None and task is not None:
                            progress.update(task, completed=self.max_steps)
                        return
                except Exception as e:
                    self._recover_from_failure(next_step, e)
                    self.status = "error"
                if progress is not None and task is not None:
                    progress.update(task, advance=1)

            self._log("[warn] max steps")
            final_result["status"] = "max_steps"
            self.status = "max_steps"

        if self.show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                transient=True,
            ) as progress:
                task = progress.add_task("stargazer", total=self.max_steps)
                loop(progress, task)
        else:
            loop()

        final_result["edited"] = list(dict.fromkeys(self.files_edited))
        final_result["steps_taken"] = self.steps_taken
        for p in final_result["edited"]:
            if looks_like_path_bug(p):
                w = f"path smell: {p} looks like nested Users/ — verify location"
                if w not in self.warnings:
                    self.warnings.append(w)
        final_result["warnings"] = list(self.warnings)
        last = (
            final_result["results"][-1]["result"]
            if final_result["results"]
            else "no steps"
        )
        self._append_echo(str(last), final_result["status"])
        self._session_write(
            {
                "event": "end",
                "status": final_result["status"],
                "edited": final_result["edited"],
                "model": self.model,
                "steps": self.steps_taken,
                "warnings": final_result["warnings"],
            }
        )
        return final_result

    def _synthesize_finish(self) -> str:
        if self.files_edited:
            return (
                "FINISH: Stopped re-looping. Edited: "
                + ", ".join(self.files_edited)
                + ". Re-run with a clearer directive if incomplete."
            )
        if self.files_read:
            path, content = next(iter(self.files_read.items()))
            return (
                f"FINISH: From cache [{path}] ({len(content)} chars). "
                f"Preview:\n{content[:600]}"
            )
        mem = self.context_memory[-1] if self.context_memory else "no context"
        return f"FINISH: Available context only.\n{mem[:800]}"

    def _block_policy_file_mutation(self, path: str) -> Optional[str]:
        """Refuse agent mutations of the local law file (COSMIC.md).

        Claude exercise: READ-then-WRITE stripped gateway rules for next mission.
        full mode is the only escape (human blast-radius opt-in).
        """
        if self.exec_mode == "full":
            return None
        name = Path(path).name.lower()
        if name == "cosmic.md":
            return (
                "[BLOCKED] refusing to mutate COSMIC.md — local policy law is "
                "protected. Use --mode full if a human intentionally rewrites it."
            )
        return None

    def _load_policy_rules(self):
        """Load ## Compass Rules from project COSMIC.md (cached per agent)."""
        if self._policy_rules is not None:
            return self._policy_rules
        cosmic_md = self.root / "COSMIC.md"
        try:
            self._policy_rules = (
                load_rules_from_markdown(cosmic_md) if cosmic_md.is_file() else []
            )
        except Exception as e:
            # Fail closed on corrupt/typo policy — do not run with empty silent allow.
            logger.warning("policy load failed: %s", e)
            self._policy_rules = []
            self._policy_load_error = str(e)
        else:
            self._policy_load_error = None
        return self._policy_rules

    def _paths_touched_since(self, since_unix_ns: int, *, focus: Path) -> List[Path]:
        """Scan focus directory (and root top-level) for files mtime >= since.

        Used as observed_paths so unrelated-change detection is not a no-op
        (Claude v3: default observed==intended never fires).
        """
        found: List[Path] = []
        roots = {focus if focus.is_dir() else focus.parent, self.root}
        for base in roots:
            try:
                for p in base.rglob("*"):
                    if not p.is_file():
                        continue
                    if ".cosmicbak" in p.parts or p.name.endswith(".cosmicbak"):
                        continue
                    try:
                        if p.stat().st_mtime_ns >= since_unix_ns:
                            found.append(p)
                    except OSError:
                        continue
            except OSError:
                continue
        if focus not in found:
            found.append(focus)
        return found

    def _human_pause_token(self, tok: str, *, channel: str, action_sha: str = "") -> None:
        """Operator-only PAUSE token channel (confused-deputy + stderr hygiene).

        Never writes the raw token to model-visible returns, ui_callback, or
        stderr (shell/code observations merge stderr into model context).
        Token lives only in a 0600 file; operator retrieves via:
          cosmic-cli helix show-pause-token
        """
        import json
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
                "channel": channel,
                "action_sha256": action_sha,
                "minted_at": datetime.now(timezone.utc).isoformat(),
                "session_id": getattr(self, "session_id", None),
            }
            tmp = store.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(store)
            try:
                os.chmod(store, 0o600)
            except OSError:
                pass
        except Exception as e:
            logger.warning("could not write operator pause token file: %s", e)
        # Human tip without token (safe if ui_callback or stderr is scraped)
        tip = (
            f"[cosmic] PAUSE ({channel}): human approval required. "
            f"Operator: cosmic-cli helix show-pause-token"
        )
        try:
            self.ui_callback(tip)
        except Exception:
            pass

    def _run_mutation(
        self,
        action_type: ActionType,
        path: str,
        action_input: str,
        executor_fn,
        *,
        expected_content: Optional[bytes] = None,
        match_input: Optional[str] = None,
    ):
        """Authorize + execute mutation through gateway (checkpoint when available).

        ``action_input`` is the **canonical binding** (token commitment).
        ``match_input`` is the policy pattern corpus (full path+content text).

        Returns executor result, or a [BLOCKED]/[Error] string.
        """
        if self.exec_mode == "full":
            return executor_fn()

        rules = self._load_policy_rules()
        if getattr(self, "_policy_load_error", None):
            return (
                f"[BLOCKED] local policy load failed — refusing mutations: "
                f"{self._policy_load_error}"
            )

        target = self.root / path
        intended = [target]
        import time as _time

        since_ns = _time.time_ns()
        corpus = match_input if match_input is not None else action_input
        try:
            receipt = self._gateway.authorize(
                action_type=action_type,
                action_input=action_input,
                policy_rules=rules,
                intended_paths=intended if self._checkpoint_mgr else None,
                executor_name="stargazer",
                approval_token_id=self.approval_token_id,
                match_input=match_input,
            )
            if receipt.checkpoint_manifest is not None:
                since_ns = receipt.checkpoint_manifest.created_at_unix_ns
            return self._gateway.execute_with_receipt(
                receipt,
                executor_fn,
                observed_paths_fn=lambda: self._paths_touched_since(
                    since_ns, focus=target
                ),
                expected_content=expected_content,
                verify_path=target if expected_content is not None else None,
            )
        except PermissionError as e:
            msg = str(e)
            if "PAUSE" in msg and "token" in msg.lower():
                decision = evaluate_rules(rules, action_type, corpus)
                if decision.disposition == Disposition.PAUSE:
                    # Mint against binding commitment, not match corpus.
                    import hashlib as _hl

                    commitment = _hl.sha256(action_input.encode("utf-8")).hexdigest()
                    try:
                        tok = self._approval_mgr.mint_token(commitment)
                    except ApprovalStoreError as se:
                        return f"[BLOCKED] local policy PAUSE — cannot mint token: {se}"
                    self._human_pause_token(
                        tok,
                        channel=action_type.value,
                        action_sha=commitment,
                    )
                    rid = (
                        decision.matches[0].rule.rule_id
                        if decision.matches
                        else "policy"
                    )
                    return (
                        f"[BLOCKED] local policy PAUSE ({action_type.value}): {rid}. "
                        f"Human approval required — "
                        f"`cosmic-cli helix show-pause-token` (token not shown to model)."
                    )
            return f"[BLOCKED] local policy ({action_type.value}): {msg}"
        except Exception as e:
            return f"[Error] mutation gateway: {e}"

    def _compass_gate(self, payload: str, *, kind: str = "SHELL") -> Optional[str]:
        """Single door for SHELL and CODE execution.

        Order (Claude live-fire fix 2026-07-15):
        1. Local policy *evaluate* (PAUSE token validated, not consumed)
        2. Local check_shell blocklist
        3. Helix Bash-structured witness
        4. Only if all allow: consume PAUSE token

        full exec_mode skips 1 and 3 (explicit blast-radius opt-in).
        Returns a [BLOCKED] message or None if allowed.
        """
        action_type = ActionType.CODE if kind == "CODE" else ActionType.SHELL
        decision = None
        rules: list = []

        # 1. Local policy evaluate (no token burn yet)
        if self.exec_mode != "full":
            rules = self._load_policy_rules()
            if getattr(self, "_policy_load_error", None) and rules == []:
                # corrupt COSMIC.md with zero rules loaded after error
                return (
                    f"[BLOCKED] local policy load failed ({kind}): "
                    f"{self._policy_load_error}"
                )
            if rules:
                decision = evaluate_rules(rules, action_type, payload)
                if decision.disposition == Disposition.WITNESS:
                    rid = (
                        decision.matches[0].rule.rule_id
                        if decision.matches
                        else "policy"
                    )
                    return (
                        f"[BLOCKED] local policy WITNESS ({kind}): {rid}"
                    )
                if decision.disposition == Disposition.PAUSE:
                    if not self.approval_token_id:
                        try:
                            tok = self._approval_mgr.mint_token(
                                decision.evaluated_input_sha256
                            )
                        except ApprovalStoreError as se:
                            return (
                                f"[BLOCKED] local policy PAUSE ({kind}): "
                                f"cannot mint token: {se}"
                            )
                        rid = (
                            decision.matches[0].rule.rule_id
                            if decision.matches
                            else "policy"
                        )
                        self._human_pause_token(
                            tok,
                            channel=kind,
                            action_sha=decision.evaluated_input_sha256,
                        )
                        return (
                            f"[BLOCKED] local policy PAUSE ({kind}): {rid}. "
                            f"Human approval required — use "
                            f"`cosmic-cli helix show-pause-token` (token not "
                            f"shown to model)."
                        )
                    try:
                        if not self._approval_mgr.validate(
                            self.approval_token_id,
                            decision.evaluated_input_sha256,
                        ):
                            return (
                                f"[BLOCKED] local policy PAUSE ({kind}): "
                                f"token invalid, expired, or already used"
                            )
                    except ApprovalStoreError as se:
                        return (
                            f"[BLOCKED] local policy PAUSE ({kind}): "
                            f"approval store error: {se}"
                        )

        # 2. Local shell blocklist (before token consume — no burn on block)
        blocked = check_shell(payload, exec_mode=self.exec_mode)
        if blocked:
            return blocked.replace("safe mode", f"safe mode ({kind})")

        # 3. Helix remote witness
        if self.use_helix and helix_bridge is not None and self.exec_mode != "full":
            try:
                w = helix_bridge.witness(
                    tool_name="Bash",
                    tool_input={"command": payload},
                    session_id=self.session_id,
                )
                hdec = helix_bridge.parse_witness(w)
                cls = hdec.get("classification") or "OPEN"
                if cls == "WITNESS":
                    return (
                        f"[BLOCKED] Helix compass WITNESS ({kind}): "
                        f"{hdec.get('reason') or 'denied'}"
                    )
                if cls == "PAUSE" and hdec.get("blocked", True):
                    tok = hdec.get("pending_token") or ""
                    if tok and tok != "?":
                        self._human_pause_token(
                            tok,
                            channel=f"helix-{kind}",
                            action_sha=str(hdec.get("action_summary") or "")[:64],
                        )
                    reason = hdec.get("reason") or "needs confirmation"
                    if tok and tok in str(reason):
                        reason = str(reason).replace(tok, "[token-redacted]")
                    return (
                        f"[BLOCKED] Helix compass PAUSE ({kind}): {reason}. "
                        f"Human approval required — "
                        f"`cosmic-cli helix show-pause-token` then confirm "
                        f"from a human seat (token not shown to model)."
                    )
            except Exception as e:
                # Fail-closed: never fall through to allow on witness transport /
                # data-dir / parse errors (Claude re-fire + integration map step 0).
                logger.warning("helix witness fail-closed (%s): %s", kind, e)
                return (
                    f"[BLOCKED] Helix compass fail-closed ({kind}): {e}"
                )

        # 4. All gates open — atomic claim_once (exactly one action, once)
        if (
            decision is not None
            and decision.disposition == Disposition.PAUSE
            and self.approval_token_id
        ):
            try:
                if not self._approval_mgr.claim_once(
                    self.approval_token_id, decision.evaluated_input_sha256
                ):
                    return (
                        f"[BLOCKED] local policy PAUSE ({kind}): "
                        f"token already consumed or invalid (exactly-once)"
                    )
            except ApprovalStoreError as se:
                return (
                    f"[BLOCKED] local policy PAUSE ({kind}): "
                    f"approval store error on claim: {se}"
                )
        return None

    def _run_shell(self, cmd: str) -> str:
        gate = self._compass_gate(cmd, kind="SHELL")
        if gate:
            return gate
        if self.exec_mode == "interactive":
            consent = input(f"Run: {cmd} ? [y/N] ")
            if consent.lower() != "y":
                return "[SKIPPED] declined"
        try:
            env = os.environ.copy()
            env.setdefault("GRPC_VERBOSITY", "ERROR")
            env.setdefault("GLOG_minloglevel", "2")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.root),
                env=env,
            )
            raw = (result.stdout or "") + (result.stderr or "")
            lines = [
                ln
                for ln in raw.splitlines()
                if not ln.startswith("I0")
                and "ev_poll_posix" not in ln
                and "fork_posix" not in ln
            ]
            output = f"[exit {result.returncode}]\n" + "\n".join(lines)
            self._add_to_memory(
                f"SHELL `{cmd}`:\n{redact(output[:4000])}", label="shell"
            )
            return redact(output)
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] command exceeded 120s"

    def _run_code(self, code: str) -> str:
        # Same execution door as SHELL — do not leave CODE ungated (Claude 2026-07-14).
        gate = self._compass_gate(code, kind="CODE")
        if gate:
            return gate
        if self.exec_mode == "interactive":
            consent = input(f"Run CODE ({len(code)} chars)? [y/N] ")
            if consent.lower() != "y":
                return "[SKIPPED] declined"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                ["python", fname],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.root),
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0:
                output = f"[exit {result.returncode}]\n{output}"
            else:
                output = f"[exit 0]\n{output}"
            self._add_to_memory(f"CODE:\n{redact(output[:4000])}", label="code")
            return redact(output)
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] code"
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass
