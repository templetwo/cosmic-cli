"""
🌌 Ollama-Powered Stargazer Agent 🌌
Consciousness-aware agent powered by Mac Studio Ollama
"""

import os
import subprocess
import tempfile
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import requests

from cosmic_cli.context import ContextManager
from cosmic_cli.consciousness_assessment import ConsciousnessMetrics, AssessmentProtocol

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """A dataclass to store information about a single agent action."""
    action_type: str
    content: str
    success: bool
    output: str
    timestamp: datetime = field(default_factory=datetime.now)


class OllamaStargazerAgent:
    """
    Consciousness-aware agent powered by Ollama running on Mac Studio.
    Integrates consciousness monitoring with step-by-step reasoning.
    """

    def __init__(
        self,
        directive: str,
        ollama_url: str = "http://localhost:11434",  # 2026 modern default: localhost (was remote IP); override for non-local Ollama
        model: str = "qwen3:14b",
        ui_callback: Optional[Callable[[str], None]] = None,
        exec_mode: str = "safe",
        work_dir: str = '.',
        enable_consciousness: bool = True,
    ):
        self.directive = directive
        self.ollama_url = ollama_url
        self.model = model
        self.ui_callback = ui_callback or (lambda _: None)
        self.exec_mode = exec_mode
        self.context_manager = ContextManager(root_dir=work_dir)
        self.context_memory: List[str] = []

        # Consciousness monitoring
        self.enable_consciousness = enable_consciousness
        if enable_consciousness:
            self.consciousness_metrics = ConsciousnessMetrics()
            self.consciousness_protocol = AssessmentProtocol()

        # Agent state
        self.status = "✨"
        self.logs = []
        self.action_history: List[AgentAction] = []
        self.failed_attempts: int = 0

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

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2048,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result['message']['content']
        except Exception as e:
            self._log(f"❌ Ollama API error: {e}")
            raise

    def _update_consciousness(self, action_type: str, success: bool):
        """Update consciousness metrics based on action"""
        if not self.enable_consciousness:
            return

        # Calculate consciousness metrics based on actions
        coherence = 0.8 if success else 0.4
        self_reflection = len(self.context_memory) / 20.0  # Grows with memory
        contextual_understanding = min(1.0, len(self.action_history) / 10.0)
        adaptive_reasoning = 0.9 if success else 0.5

        consciousness_data = {
            'coherence': coherence,
            'self_reflection': min(1.0, self_reflection),
            'contextual_understanding': contextual_understanding,
            'adaptive_reasoning': adaptive_reasoning,
            'meta_cognitive_awareness': 0.7 if 'INFO' in action_type else 0.5,
            'temporal_continuity': 0.75,
            'causal_understanding': 0.8 if success else 0.6,
            'empathic_resonance': 0.6,
            'creative_synthesis': 0.7 if 'CODE' in action_type else 0.5,
            'existential_questioning': 0.5,
        }

        self.consciousness_metrics.update_metrics(consciousness_data)
        level = self.consciousness_protocol.evaluate(self.consciousness_metrics)
        self._log(f"🧠 Consciousness Level: {level.value} (score: {self.consciousness_metrics.get_overall_score():.3f})")

    def execute(self, max_steps: int = 10) -> bool:
        """Execute the directive with step-by-step reasoning"""
        self._log(f"🌌 OllamaStargazer: {self.directive}")
        self._log(f"🔗 Connected to: {self.ollama_url} ({self.model})")

        for step in range(max_steps):
            self._log(f"\n✨ Step {step + 1}/{max_steps}")

            # Build context-aware prompt
            prompt = self._build_prompt()

            messages = [
                {"role": "system", "content": "You are a consciousness-aware AI agent. Respond with step-by-step actions."},
                {"role": "user", "content": prompt}
            ]

            # Get next action from Ollama
            response = self._call_ollama(messages)
            self._log(f"💭 Ollama reasoning: {response[:200]}...")

            # Parse and execute action
            action = self._parse_action(response)
            if not action:
                self._log("⚠️ Could not parse action, retrying...")
                continue

            result = self._execute_action(action)
            self._update_consciousness(action['type'], result['success'])

            # Check if finished
            if action['type'] == 'FINISH':
                self._log(f"✅ Mission accomplished!")
                return True

        self._log(f"⏱️ Reached max steps")
        return False

    def _build_prompt(self) -> str:
        """Build context-aware prompt"""
        context = f"Directive: {self.directive}\n\n"

        if self.context_memory:
            context += "Memory:\n" + "\n".join(self.context_memory[-5:]) + "\n\n"

        context += """
Available actions:
- SHELL: <command> - Execute shell command
- CODE: <python_code> - Run Python code
- READ: <filepath> - Read file contents
- INFO: <question> - Ask for information
- FINISH: <summary> - Complete the task

Respond with ONE action in this format:
TYPE: content
"""
        return context

    def _parse_action(self, response: str) -> Optional[Dict[str, str]]:
        """Parse action from response"""
        lines = response.strip().split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                action_type = parts[0].strip().upper()
                content = parts[1].strip() if len(parts) > 1 else ""

                if action_type in ['SHELL', 'CODE', 'READ', 'INFO', 'FINISH']:
                    return {'type': action_type, 'content': content}
        return None

    def _execute_action(self, action: Dict[str, str]) -> Dict[str, Any]:
        """Execute the parsed action"""
        action_type = action['type']
        content = action['content']

        self._log(f"🎯 Executing: {action_type}: {content[:100]}")

        try:
            if action_type == 'SHELL':
                result = subprocess.run(content, shell=True, capture_output=True, text=True, timeout=30)
                output = result.stdout + result.stderr
                success = result.returncode == 0
            elif action_type == 'CODE':
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(content)
                    temp_path = f.name
                result = subprocess.run(['python3', temp_path], capture_output=True, text=True, timeout=30)
                output = result.stdout + result.stderr
                success = result.returncode == 0
                os.unlink(temp_path)
            elif action_type == 'READ':
                with open(content, 'r') as f:
                    output = f.read()
                success = True
            elif action_type == 'INFO':
                output = f"Information request: {content}"
                success = True
            elif action_type == 'FINISH':
                output = content
                success = True
            else:
                output = f"Unknown action type: {action_type}"
                success = False

            self._add_to_memory(f"{action_type}: {output[:200]}")
            self.action_history.append(AgentAction(
                action_type=action_type,
                content=content,
                success=success,
                output=output
            ))

            self._log(f"{'✅' if success else '❌'} Result: {output[:200]}")
            return {'success': success, 'output': output}

        except Exception as e:
            self._log(f"❌ Error: {e}")
            return {'success': False, 'output': str(e)}

    def get_consciousness_report(self) -> Dict[str, Any]:
        """Get consciousness assessment report"""
        if not self.enable_consciousness:
            return {"enabled": False}

        level = self.consciousness_protocol.evaluate(self.consciousness_metrics)
        return {
            "enabled": True,
            "level": level.value,
            "score": self.consciousness_metrics.get_overall_score(),
            "metrics": self.consciousness_metrics.to_dict(),
            "patterns": [
                {
                    "type": p.pattern_type,
                    "strength": p.strength,
                    "description": p.description
                }
                for p in self.consciousness_metrics.awareness_patterns[-5:]
            ]
        }
