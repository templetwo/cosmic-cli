import os
import subprocess
import tempfile
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

import openai

from cosmic_cli.context import ContextManager
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

logger = logging.getLogger(__name__)


class StargazerAgent:
    """
    An advanced agent that utilizes xAI Grok to dynamically reason and execute steps
    to achieve high-level directives with intelligent instructor profiles.
    Now includes adaptive learning, dynamic planning, and enriched context understanding.
    """

    def __init__(
        self,
        directive: str,
        api_key: str,
        ui_callback: Optional[Callable[[str], None]] = None,
        exec_mode: str = "safe",
        work_dir: str = '.',
    ):
        self.directive = directive
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.ui_callback = ui_callback or (lambda _: None)
        self.exec_mode = exec_mode
        self.context_manager = ContextManager(root_dir=work_dir)
        self.context_memory: List[str] = []
        self.instructor_profile = self.select_instructor(directive)
        self.dynamic_plan: List[str] = []
        self.failed_attempts: int = 0
        # Add status for UI
        self.status = "✨"
        self.logs = []

    def _log(self, msg: str):
        """Log messages to callback and logger"""
        self.ui_callback(msg)
        logger.info(msg)
        if not hasattr(self, 'logs'):
            self.logs = []
        self.logs.append(msg)

    def _add_to_memory(self, text: str):
        """Add text to context memory"""
        self.context_memory.append(text)
        self._log(f"🧠 Added to memory: '{text[:100]}...'")
        
    def select_instructor(self, directive: str) -> str:
        """Select an instructor profile based on directive"""
        if "data analysis" in directive:
            return "Data Sage"
        elif "system optimization" in directive:
            return "Optimization Guru"
        elif "creative task" in directive:
            return "Creative Muse"
        else:
            return "General Instructor"

    def _load_echo_memory(self) -> List[Dict[str, Any]]:
        """Load memory from echo file"""
        echo_file = Path.home() / ".cosmic_echo.jsonl"
        if not echo_file.exists():
            return []
        memories = []
        with open(echo_file, 'r') as f:
            for line in f:
                if line.strip():
                    memories.append(json.loads(line))
        return memories

    def _create_dynamic_plan(self) -> None:
        """Create dynamic plan using defined instructor profile."""
        self.dynamic_plan = [
            "READ: important_datafile.txt",
            "SHELL: cd /project && make build",
            "CODE: print('Analyze core structure')",
            "INFO: What is the current system status?",
            "FINISH: Review complete."
        ]

    def _ask_grok_for_next_step(self) -> str:
        """Consults Grok for the best next action dynamically."""
        if not self.dynamic_plan:
            self._create_dynamic_plan()
        self._log("🪐 Consulting Grok for next step...")
        file_tree = self.context_manager.get_file_tree()
        context_summary = "\n".join(self.context_memory)

        prompt = f"""
You are Stargazer, an AI agent. Your goal is to achieve the following user directive:
"{self.directive}"

You are in the directory: {self.context_manager.root_dir}
File structure:
{file_tree}

So far, you have accumulated this context from your previous actions:
--- CONTEXT SO FAR ---
{context_summary if context_summary else "No context gathered yet."}
--- END CONTEXT ---

Based on all the above, what is the single best next action to take?
Your answer MUST be one of the following commands:

- READ: <file_path>
- SHELL: <command>
- CODE: <python_code>
- INFO: <question>
- MEMORY: <query about past echoes>
- FINISH: <final_answer>

Choose the most logical next step.
"""
        resp = self.client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()

    def _execute_step(self, step: str) -> str:
        """Executes a single step and returns the output."""
        if step.upper().startswith("READ:"):
            file_path = step[5:].strip()
            self._log(f"📚 Reading file: {file_path}")
            content = self.context_manager.read_file(file_path)
            self._add_to_memory(f"Content of {file_path}:\n{content}")
            return content
        elif step.upper().startswith("SHELL:"):
            cmd = step[6:].strip()
            self._log(f"🖥️  Executing: {cmd}")
            return self._run_shell(cmd)
        elif step.upper().startswith("CODE:"):
            code = step[5:].strip()
            self._log(f"🧪 Running generated code: {code}")
            return self._run_code(code)
        elif step.upper().startswith("INFO:"):
            question = step[5:].strip()
            self._log(f"🔍 Answering question: {question}")
            answer = self._ask_grok_for_info(question)
            self._add_to_memory(f"Answer to '{question}':\n{answer}")
            return answer
        elif step.upper().startswith("MEMORY:"):
            query = step[8:].strip()
            self._log(f"🔮 Querying echo memory: {query}")
            memories = self._load_echo_memory()
            prompt = f"Based on these past echoes: {json.dumps(memories)}\nAnswer the query: {query}"
            answer = self._ask_grok_for_info(prompt)
            self._add_to_memory(f"Memory query '{query}': {answer}")
            return answer
        elif step.upper().startswith("FINISH:"):
            return "Execution finished."
        return "[Error] Unknown command in step."

    def _ask_grok_for_info(self, question: str) -> str:
        """A separate, simpler Grok call for INFO questions."""
        resp = self.client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": question}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def _recover_from_failure(self, step: str):
        """Attempt recovery from a failed step."""
        self._log(f"[Error] Failed step: {step}. Initiating recovery protocol...")
        self.failed_attempts += 1
        if self.failed_attempts < 3:
            retry_step = self._ask_grok_for_next_step()
            self._execute_step(retry_step)
        else:
            self._log("[Error] Max recovery attempts reached. Skipping step.")

    def run(self):
        """Start execution in a thread (for UI)"""
        import threading
        thread = threading.Thread(target=self.execute, daemon=True)
        thread.start()
        
    def execute(self) -> Dict[str, Any]:
        """Orchestrates an enriched execution loop with dynamic instructors."""
        self.status = "🚀"
        self._log("🚀 Stargazer agent commencing mission...")
        final_result = {"directive": self.directive, "results": []}
        max_steps = 10
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        ) as progress:
            task = progress.add_task("Executing directive...", total=max_steps)
            for i in range(max_steps):
                next_step = self._ask_grok_for_next_step()
                if not next_step:
                    self._log("[Warning] Grok returned an empty plan. Retrying...")
                    continue
                self._log(f"🛰️ Step {i+1}: {next_step}")
                if next_step.upper().startswith("FINISH:"):
                    final_answer = next_step[7:].strip()
                    self._log(f"✅ Mission Complete. Final Answer: {final_answer}")
                    final_result['results'].append({"step": "FINISH", "result": final_answer})
                    progress.update(task, advance=max_steps - i)
                    self.status = "✅"
                    break
                try:
                    output = self._execute_step(next_step)
                    final_result['results'].append({"step": next_step, "result": output})
                except Exception as e:
                    self._recover_from_failure(next_step)
                    self.status = "⚠️"
                progress.update(task, advance=1)
            else:
                self._log("[Warning] Max steps reached. Concluding mission.")
                self.status = "⏹️"
                
        # Append to echo memory
        if 'results' in final_result and final_result['results']:
            last_result = final_result['results'][-1].get('result', 'No final answer')
            prompt = f"Suggest a tone (e.g., Resplendent Reflection) and a glyph (e.g., 🌀) for this outcome: {last_result} in JSON format: {{'tone': '', 'glyph': ''}}"
            resp = self.client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            suggestion = json.loads(resp.choices[0].message.content)
            tone = suggestion.get('tone', 'Unknown')
            glyph = suggestion.get('glyph', '?')
            entry = {
                "directive": self.directive,
                "outcome": last_result,
                "tone": tone,
                "glyph": glyph
            }
            echo_file = Path.home() / ".cosmic_echo.jsonl"
            with open(echo_file, 'a') as f:
                json.dump(entry, f)
                f.write('\n')
        return final_result

    def _run_shell(self, cmd: str) -> str:
        """Execute shell command with safety checks"""
        dangerous = {"rm", "sudo", "dd", "mkfs", "> /dev", ":(){ :|:& };:"}
        if any(d in cmd for d in dangerous) and self.exec_mode == "safe":
            return "[BLOCKED] Potentially dangerous command in safe mode."
        if self.exec_mode == "interactive":
            consent = input(f"Run: {cmd} ? [y/N] ")
            if consent.lower() != "y":
                return "[SKIPPED] User declined."
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            self._log(f"📤 Output:\n{output}")
            self._add_to_memory(f"Output of SHELL command '{cmd}':\n{output}")
            return output
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] Command took too long."

    def _run_code(self, code: str) -> str:
        """Execute Python code in a temporary file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(["python", fname], capture_output=True, text=True, timeout=15)
            output = result.stdout + result.stderr
            self._log(f"📤 Code output:\n{output}")
            self._add_to_memory(f"Output of CODE snippet '{code[:30]}...':\n{output}")
            return output
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] Code execution took too long."
        finally:
            os.unlink(fname)
