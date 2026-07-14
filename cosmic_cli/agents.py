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
from cosmic_cli.principles import system_prompt_block
from cosmic_cli.tools import (
    parse_edit_payload,
    parse_grep_payload,
    parse_write_payload,
    tool_edit,
    tool_glob,
    tool_grep,
    tool_list,
    tool_write,
)

logger = logging.getLogger(__name__)

ECHO_FILE = Path.home() / ".cosmic_echo.jsonl"
SESSION_DIR = Path.home() / ".cosmic-cli" / "sessions"
DEFAULT_MODEL = os.getenv("COSMIC_GROK_MODEL", "grok-4.5")
MAX_STEPS_DEFAULT = 20
CONTEXT_MEMORY_CAP = 16
CONTEXT_CHARS_CAP = 28_000
FILE_TREE_LINES_CAP = 120

DANGEROUS_SHELL = (
    "rm -rf",
    "rm -r ",
    "sudo ",
    " mkfs",
    "dd if=",
    "> /dev",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "shutdown",
    "reboot",
    "diskutil erase",
    "mkfs.",
    "git push --force",
    "git reset --hard",
)

STEP_PREFIXES = (
    "GLOB:",
    "GREP:",
    "LIST:",
    "READ:",
    "DIFF:",
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
        show_progress: bool = True,
        session_id: Optional[str] = None,
        auto_verify: bool = True,
    ):
        self.directive = directive
        self.client = Client(api_key=api_key)
        self.ui_callback = ui_callback or (lambda _: None)
        self.exec_mode = exec_mode
        self.root = Path(work_dir).resolve()
        self.context_manager = ContextManager(root_dir=str(self.root))
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
        self.steps_taken: int = 0
        self.session_id = session_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self.session_path = SESSION_DIR / f"{self.session_id}.jsonl"
        SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # ── logging / memory ──────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self.logs.append(msg)
        logger.info(msg)
        if not self.quiet:
            self.ui_callback(msg)

    def _add_to_memory(self, text: str, *, label: str = "") -> None:
        self.context_memory.append(text)
        if len(self.context_memory) > CONTEXT_MEMORY_CAP:
            self.context_memory = self.context_memory[-CONTEXT_MEMORY_CAP:]
        preview = text if len(text) <= 110 else text[:107] + "..."
        self._log(f"  obs{(' · ' + label) if label else ''}: {preview!r}")

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
            "outcome": str(outcome)[:2000],
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
                    if prefix in ("EDIT:", "WRITE:") and "|||" in text:
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
        prompt = f"""You are Stargazer, a coding agent operating in a real project directory.
The filesystem is ground truth. You change it only through tools.

{system_prompt_block()}

DIRECTIVE:
{self.directive}

WORKING DIR: {self.root}
MODEL: {self.model}
MODE: {self.exec_mode}
FILES CACHED: {already}
EDITED: {', '.join(self.files_edited) or '(none)'}

FILE TREE (partial):
{self._file_tree()}

CONTEXT / OBSERVATIONS:
{self._context_blob() or "None yet — start with GLOB/GREP/LIST or READ."}

Reply with EXACTLY ONE action line (EDIT/WRITE may span multiple lines after |||).
No prose outside the action.
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
            file_path = self._body_after(step, "READ:").strip("'\"").lstrip("./")
            if file_path in self.files_read and file_path not in self.files_edited:
                self._log(f"READ {file_path} (cache)")
                return self.files_read[file_path]
            self._log(f"READ {file_path}")
            content = self.context_manager.read_file(file_path)
            self.files_read[file_path] = content
            if not content.startswith("[Error]"):
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
            path, old, new = parsed
            path = path.strip().lstrip("./")
            if path not in self.files_seen:
                return (
                    f"[Error] READ-before-EDIT violated for {path}. "
                    "READ the file first, then EDIT with exact old_string."
                )
            # refresh cache from disk if invalidated after prior edit
            if path not in self.files_read:
                self.files_read[path] = self.context_manager.read_file(path)
            self._log(f"EDIT {path}")
            obs, ok = tool_edit(self.root, path, old, new)
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
            path, content = parsed
            path = path.strip().lstrip("./")
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
            obs, ok = tool_write(self.root, path, content)
            self._add_to_memory(obs, label="write")
            if ok:
                self.files_edited.append(path)
                self.files_seen.add(path)
                self.files_read[path] = content
            return obs

        if upper.startswith("DIFF:"):
            path = self._body_after(step, "DIFF:").strip().lstrip("./")
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
            memories = self._load_echo_memory()[-20:]
            answer = self._ask_grok_for_info(
                f"Past echoes:\n{json.dumps(memories, ensure_ascii=False)}\n\nQuery: {query}"
            )
            self._add_to_memory(f"MEMORY: {answer}", label="memory")
            return answer

        if upper.startswith("PASS:"):
            return self._body_after(step, "PASS:")

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
        }

        def loop(progress=None, task=None) -> None:
            for i in range(self.max_steps):
                self.steps_taken = i + 1
                next_step = self._ask_grok_for_next_step()
                if not next_step:
                    self._log("[warn] empty step")
                    continue

                self._log(f"→ {i + 1}/{self.max_steps} {next_step.splitlines()[0][:160]}")
                upper = next_step.upper()
                self._session_write({"event": "step", "n": i + 1, "action": next_step[:2000]})

                # loop breaker for non-terminal repeats (threshold 2 — allow one retry)
                action_key = upper.split(":", 1)[0] + ":" + (
                    next_step.split(":", 1)[1].strip()[:80] if ":" in next_step else ""
                )
                if not upper.startswith(("FINISH:", "PASS:", "EDIT:", "WRITE:")):
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
                                "File cache is enough — use EDIT/WRITE now. "
                                "Do not re-READ cached files.",
                                label="steer",
                            )
                            self._log("steer · use EDIT (cache warm)")
                            # skip executing the duplicate discovery step
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
                    reason = next_step[5:].strip()
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

    def _run_shell(self, cmd: str) -> str:
        if self.exec_mode == "safe":
            if any(d in cmd for d in DANGEROUS_SHELL):
                return "[BLOCKED] dangerous command in safe mode."
            if re.search(r"\brm\b.*(-[a-zA-Z]*[rf]|--recursive)", cmd):
                return "[BLOCKED] dangerous command in safe mode."
        if self.exec_mode == "interactive":
            consent = input(f"Run: {cmd} ? [y/N] ")
            if consent.lower() != "y":
                return "[SKIPPED] declined"
        try:
            env = os.environ.copy()
            # quiet noisy native logging from xai/grpc in child processes
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
            # drop absl/grpc fork spam that confuses verify
            lines = [
                ln
                for ln in raw.splitlines()
                if not ln.startswith("I0")
                and "ev_poll_posix" not in ln
                and "fork_posix" not in ln
            ]
            output = "\n".join(lines)
            output = f"[exit {result.returncode}]\n{output}"
            self._add_to_memory(f"SHELL `{cmd}`:\n{output[:4000]}", label="shell")
            return output
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] command exceeded 120s"

    def _run_code(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                ["python", fname], capture_output=True, text=True, timeout=30
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0:
                output = f"[exit {result.returncode}]\n{output}"
            self._add_to_memory(f"CODE:\n{output[:4000]}", label="code")
            return output
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] code"
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass
