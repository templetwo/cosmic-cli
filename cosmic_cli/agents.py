import os
import subprocess
import tempfile
import json
import logging
from typing import List, Dict, Any, Optional, Callable

import openai

from cosmic_cli.context import ContextManager

logger = logging.getLogger(__name__)


class StargazerAgent:
    """
    An agentic layer that uses xAI Grok to reason and execute steps
    to fulfill a high-level directive, with a feedback loop for context.
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

    def _log(self, msg: str):
        self.ui_callback(msg)
        logger.info(msg)

    def _add_to_memory(self, text: str):
        self.context_memory.append(text)
        self._log(f"🧠 Added to memory: '{text[:100]}...'")

    def _ask_grok_for_next_step(self) -> str:
        """Consults Grok for the best next action based on current context."""
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

    def execute(self) -> Dict[str, Any]:
        """Orchestrates the new step-by-step execution loop."""
        self._log("🚀 Stargazer agent commencing mission...")
        final_result = {"directive": self.directive, "results": []}
        for i in range(10):
            next_step = self._ask_grok_for_next_step()
            if not next_step:
                self._log("[Warning] Grok returned an empty plan. Retrying...")
                continue
            self._log(f"🛰️ Step {i+1}: {next_step}")
            if next_step.upper().startswith("FINISH:"):
                final_answer = next_step[7:].strip()
                self._log(f"✅ Mission Complete. Final Answer: {final_answer}")
                final_result['results'].append({"step": "FINISH", "result": final_answer})
                break
            output = self._execute_step(next_step)
            final_result['results'].append({"step": next_step, "result": output})
        else:
            self._log("[Warning] Max steps reached. Concluding mission.")
        return final_result

    def _run_shell(self, cmd: str) -> str:
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