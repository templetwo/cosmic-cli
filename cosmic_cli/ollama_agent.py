"""Ollama Stargazer — same tools + shell guard as Grok path (audit C1/C2)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from cosmic_cli.agents import StargazerAgent
from cosmic_cli.context import ContextManager
from cosmic_cli.secrets import deny_read_message, is_sensitive_path, redact
from cosmic_cli.shell_guard import check_shell
from cosmic_cli.tools import (
    parse_edit_payload,
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

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    action_type: str
    content: str
    success: bool
    output: str
    timestamp: datetime = field(default_factory=datetime.now)


class OllamaStargazerAgent:
    """Local-model agent reusing tools.py + shell_guard (no drifted safety)."""

    def __init__(
        self,
        directive: str,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma4:e2b",
        ui_callback: Optional[Callable[[str], None]] = None,
        exec_mode: str = "safe",
        work_dir: str = ".",
        enable_consciousness: bool = False,
    ):
        self.directive = directive
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.ui_callback = ui_callback or (lambda _: None)
        self.exec_mode = exec_mode
        self.root = Path(work_dir).resolve()
        self.context_manager = ContextManager(root_dir=str(self.root))
        self.context_memory: List[str] = []
        self.files_read: Dict[str, str] = {}
        self.files_seen: set[str] = set()
        self.enable_consciousness = enable_consciousness
        self.status = "ready"
        self.logs: List[str] = []
        self.action_history: List[AgentAction] = []
        self._last_key: Optional[str] = None
        self._repeat = 0
        # optional legacy consciousness (off by default; not required)
        self.consciousness_metrics = None
        self.consciousness_protocol = None
        if enable_consciousness:
            try:
                from cosmic_cli.consciousness_assessment import (
                    AssessmentProtocol,
                    ConsciousnessMetrics,
                )

                self.consciousness_metrics = ConsciousnessMetrics()
                self.consciousness_protocol = AssessmentProtocol()
            except Exception as e:  # pragma: no cover
                logger.warning("consciousness unavailable: %s", e)
                self.enable_consciousness = False

    def _log(self, msg: str) -> None:
        self.logs.append(msg)
        logger.info(msg)
        self.ui_callback(msg)

    def _add_to_memory(self, text: str) -> None:
        self.context_memory.append(redact(text)[:4000])
        self._log(f"  · {text[:120].replace(chr(10), ' ')}…")

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 2048},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def execute(self, max_steps: int = 12) -> bool:
        self._log(f"ollama mission · {self.model} @ {self.ollama_url} · mode={self.exec_mode}")
        for step in range(max_steps):
            self._log(f"→ {step + 1}/{max_steps}")
            prompt = self._build_prompt()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Stargazer on Ollama. One action line only. "
                        "Use CREATE/WRITE/READ/LIST/GREP/SHELL/FINISH. "
                        "Relative paths. Safe mode is enforced server-side."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            try:
                response = self._call_ollama(messages)
            except Exception as e:
                self._log(f"[Error] ollama: {e}")
                return False
            self._log(f"  model: {response[:160].replace(chr(10), ' ')}")
            step_line = StargazerAgent.parse_step(response)
            if not step_line:
                continue
            key = step_line[:80]
            if key == self._last_key:
                self._repeat += 1
            else:
                self._last_key = key
                self._repeat = 0
            if self._repeat >= 2 and not step_line.upper().startswith("FINISH"):
                step_line = "FINISH: stopped after repeated action"
            result = self._execute_step(step_line)
            self._update_consciousness(step_line.split(":", 1)[0], not result.startswith("["))
            if step_line.upper().startswith("FINISH:"):
                self._log("done")
                return True
            if step_line.upper().startswith("PASS:"):
                self._log(f"pass · {result}")
                return False
        self._log("max steps")
        return False

    def _build_prompt(self) -> str:
        mem = "\n".join(self.context_memory[-6:]) or "(none)"
        return (
            f"Directive: {self.directive}\n"
            f"Work dir: {self.root}\n"
            f"Mode: {self.exec_mode}\n"
            f"Memory:\n{mem}\n\n"
            "One action: LIST/GREP/READ/CREATE/WRITE/EDIT/SHELL/FINISH/PASS\n"
            "CREATE: path|||contents for new files.\n"
        )

    def _execute_step(self, step: str) -> str:
        upper = step.upper()
        try:
            if upper.startswith("LIST:"):
                d = step.split(":", 1)[1].strip() or "."
                out = tool_list(self.root, d)
            elif upper.startswith("GREP:"):
                from cosmic_cli.tools import parse_grep_payload

                pat, path, glob = parse_grep_payload(step.split(":", 1)[1].strip())
                out = tool_grep(self.root, pat, path=path, glob=glob)
            elif upper.startswith("GLOB:"):
                out = tool_glob(self.root, step.split(":", 1)[1].strip())
            elif upper.startswith("READ:"):
                raw = step.split(":", 1)[1].strip().strip("'\"")
                if is_sensitive_path(raw):
                    out = deny_read_message(raw)
                else:
                    path = rel_key(raw, self.root)
                    if is_sensitive_path(path):
                        out = deny_read_message(path)
                    else:
                        out = redact(self.context_manager.read_file(path))
                        if not out.startswith("[Error]"):
                            self.files_seen.add(path)
                            self.files_read[path] = out
            elif upper.startswith("CREATE:"):
                parsed = parse_write_payload(step.split(":", 1)[1].lstrip())
                if not parsed:
                    out = "[Error] CREATE format path|||content"
                else:
                    path = rel_key(parsed[0], self.root)
                    obs, ok = tool_create(self.root, path, parsed[1])
                    out = obs
                    if ok:
                        self.files_seen.add(path)
            elif upper.startswith("WRITE:"):
                parsed = parse_write_payload(step.split(":", 1)[1].lstrip())
                if not parsed:
                    out = "[Error] WRITE format"
                else:
                    path = rel_key(parsed[0], self.root)
                    obs, ok = tool_write(self.root, path, parsed[1])
                    out = obs
                    if ok:
                        self.files_seen.add(path)
            elif upper.startswith("EDIT:"):
                parsed = parse_edit_payload(step.split(":", 1)[1].lstrip())
                if not parsed:
                    out = "[Error] EDIT format"
                else:
                    path = rel_key(parsed[0], self.root)
                    if path not in self.files_seen:
                        out = f"[Error] READ-before-EDIT for {path}"
                    else:
                        obs, ok = tool_edit(self.root, path, parsed[1], parsed[2])
                        out = obs
            elif upper.startswith("MKDIR:"):
                path = rel_key(step.split(":", 1)[1].strip(), self.root)
                obs, ok = tool_mkdir(self.root, path)
                out = obs
            elif upper.startswith("SHELL:"):
                cmd = step.split(":", 1)[1].strip()
                block = check_shell(cmd, exec_mode=self.exec_mode)
                if block:
                    out = block
                else:
                    r = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=str(self.root),
                    )
                    out = f"[exit {r.returncode}]\n{(r.stdout or '') + (r.stderr or '')}"
            elif upper.startswith("CODE:"):
                code = step.split(":", 1)[1].strip()
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, encoding="utf-8"
                ) as f:
                    f.write(code)
                    fname = f.name
                try:
                    r = subprocess.run(
                        ["python", fname], capture_output=True, text=True, timeout=30
                    )
                    out = (r.stdout or "") + (r.stderr or "")
                finally:
                    try:
                        os.unlink(fname)
                    except OSError:
                        pass
            elif upper.startswith("FINISH:"):
                out = step.split(":", 1)[1].strip()
            elif upper.startswith("PASS:"):
                out = step.split(":", 1)[1].strip()
            else:
                out = f"[Error] unknown: {step[:80]}"
        except Exception as e:
            out = f"[Error] {e}"

        self._add_to_memory(f"{step.split(':',1)[0]}: {out[:500]}")
        self.action_history.append(
            AgentAction(
                action_type=step.split(":", 1)[0],
                content=step[:200],
                success=not out.startswith("[Error]") and not out.startswith("[BLOCKED]"),
                output=out[:2000],
            )
        )
        return out

    def _update_consciousness(self, action_type: str, success: bool) -> None:
        if not self.enable_consciousness or not self.consciousness_metrics:
            return
        try:
            self.consciousness_metrics.update_metrics(
                {
                    "coherence": 0.8 if success else 0.4,
                    "self_reflection": min(1.0, len(self.context_memory) / 20.0),
                    "contextual_understanding": min(1.0, len(self.action_history) / 10.0),
                    "adaptive_reasoning": 0.9 if success else 0.5,
                }
            )
        except Exception:
            pass

    def get_consciousness_report(self) -> Dict[str, Any]:
        if not self.enable_consciousness or not self.consciousness_metrics:
            return {"enabled": False}
        level = self.consciousness_protocol.evaluate(self.consciousness_metrics)
        return {
            "enabled": True,
            "level": level.value,
            "score": self.consciousness_metrics.get_overall_score(),
        }
