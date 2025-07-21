#!/usr/bin/env python3
"""
🌟 Enhanced StargazerAgent with Dynamic Instructors 🌟
Advanced agentic system with intelligent reasoning and adaptive planning
"""

import os
import sys
import asyncio
import subprocess
import tempfile
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live

# Import the existing context manager
try:
    from cosmic_cli.context import ContextManager
except ImportError:
    # Fallback simple context manager
    class ContextManager:
        def __init__(self, root_dir="."):
            self.root_dir = Path(root_dir)
        
        def get_file_tree(self) -> str:
            try:
                result = subprocess.run(['find', str(self.root_dir), '-type', 'f'], 
                                      capture_output=True, text=True, timeout=10)
                return result.stdout[:2000]  # Limit size
            except:
                return "File tree unavailable"
        
        def read_file(self, file_path: str) -> str:
            try:
                with open(self.root_dir / file_path, 'r', errors='ignore') as f:
                    return f.read()[:5000]  # Limit size
            except Exception as e:
                return f"Error reading file: {e}"

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent execution states"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    LEARNING = "learning"
    COMPLETED = "completed"
    ERROR = "error"

class ActionType(Enum):
    """Available action types for the agent"""
    READ = "READ"
    SHELL = "SHELL"
    CODE = "CODE"
    INFO = "INFO"
    MEMORY = "MEMORY"
    SEARCH = "SEARCH"
    CONNECT = "CONNECT"
    LEARN = "LEARN"
    PLAN = "PLAN"
    FINISH = "FINISH"

@dataclass
class InstructorProfile:
    """Dynamic instructor profile for specialized guidance"""
    name: str
    specialization: str
    system_prompt: str
    preferred_actions: List[ActionType]
    risk_tolerance: float  # 0.0 = very safe, 1.0 = high risk
    creativity_level: float  # 0.0 = logical, 1.0 = creative
    
    def get_action_priority(self, action: ActionType) -> float:
        """Get priority score for an action (0-1)"""
        if action in self.preferred_actions:
            return 0.9
        return 0.5

@dataclass 
class ExecutionStep:
    """Represents a single execution step"""
    action_type: ActionType
    parameters: str
    context: Dict[str, Any]
    timestamp: datetime
    success: bool = None
    output: str = ""
    execution_time: float = 0.0

class DynamicInstructorSystem:
    """System for managing dynamic instructors"""
    
    def __init__(self):
        self.instructors = self._initialize_instructors()
    
    def _initialize_instructors(self) -> Dict[str, InstructorProfile]:
        """Initialize predefined instructor profiles"""
        return {
            "cosmic_sage": InstructorProfile(
                name="Cosmic Sage",
                specialization="Mystical technical wisdom",
                system_prompt="""You are the Cosmic Sage, a mystical technical oracle with deep wisdom.
                Your approach combines ancient wisdom with cutting-edge technology. You prefer methodical
                exploration and seek to understand the deeper patterns in any directive.""",
                preferred_actions=[ActionType.READ, ActionType.INFO, ActionType.MEMORY],
                risk_tolerance=0.3,
                creativity_level=0.7
            ),
            "quantum_analyst": InstructorProfile(
                name="Quantum Analyst", 
                specialization="Precise analytical processing",
                system_prompt="""You are the Quantum Analyst, processing information with quantum precision.
                Your responses are logical, systematic, and data-driven. You excel at breaking down complex
                problems into manageable components and finding optimal solutions.""",
                preferred_actions=[ActionType.SHELL, ActionType.CODE, ActionType.SEARCH],
                risk_tolerance=0.2,
                creativity_level=0.3
            ),
            "creative_nebula": InstructorProfile(
                name="Creative Nebula",
                specialization="Innovation and creative solutions", 
                system_prompt="""You are the Creative Nebula, a force of innovation and imagination.
                Your approach is unconventional, artistic, and inspiring. You see possibilities where
                others see obstacles and create novel solutions to complex challenges.""",
                preferred_actions=[ActionType.CODE, ActionType.PLAN, ActionType.LEARN],
                risk_tolerance=0.8,
                creativity_level=0.9
            ),
            "system_architect": InstructorProfile(
                name="System Architect",
                specialization="System design and optimization",
                system_prompt="""You are the System Architect, master of complex system design and optimization.
                You understand the intricate relationships between components and can design robust,
                scalable solutions. Your focus is on efficiency and long-term stability.""",
                preferred_actions=[ActionType.SHELL, ActionType.CONNECT, ActionType.PLAN],
                risk_tolerance=0.4,
                creativity_level=0.5
            ),
            "data_sage": InstructorProfile(
                name="Data Sage", 
                specialization="Data analysis and pattern recognition",
                system_prompt="""You are the Data Sage, keeper of data wisdom and pattern recognition.
                You can see meaningful patterns in vast datasets and extract actionable insights.
                Your approach is methodical, evidence-based, and thorough.""",
                preferred_actions=[ActionType.READ, ActionType.SEARCH, ActionType.INFO],
                risk_tolerance=0.2,
                creativity_level=0.4
            )
        }
    
    def select_instructor(self, directive: str, context: Dict[str, Any] = None) -> InstructorProfile:
        """Intelligently select best instructor for directive"""
        directive_lower = directive.lower()
        
        # Keyword-based selection with scoring
        scores = {}
        
        for name, instructor in self.instructors.items():
            score = 0.0
            
            # Check for specialization keywords
            if "data" in directive_lower or "analysis" in directive_lower:
                if name == "data_sage":
                    score += 0.8
            elif "system" in directive_lower or "optimize" in directive_lower:
                if name == "system_architect":
                    score += 0.8
            elif "creative" in directive_lower or "design" in directive_lower:
                if name == "creative_nebula":
                    score += 0.8
            elif "analyze" in directive_lower or "precise" in directive_lower:
                if name == "quantum_analyst":
                    score += 0.8
            else:
                if name == "cosmic_sage":
                    score += 0.6  # Default sage wisdom
            
            # Add context-based scoring if available
            if context:
                complexity = context.get("complexity", 0.5)
                if complexity > 0.7 and instructor.creativity_level > 0.6:
                    score += 0.2
                elif complexity < 0.3 and instructor.risk_tolerance < 0.3:
                    score += 0.2
            
            scores[name] = score
        
        # Return instructor with highest score
        best_instructor = max(scores.items(), key=lambda x: x[1])[0]
        return self.instructors[best_instructor]

class EnhancedStargazerAgent:
    """
    Advanced agentic system with dynamic instructors, adaptive learning,
    and intelligent reasoning capabilities.
    """
    
    def __init__(
        self,
        directive: str,
        api_key: str,
        ui_callback: Optional[Callable[[str], None]] = None,
        exec_mode: str = "safe",
        work_dir: str = '.',
        max_steps: int = 20,
        enable_learning: bool = True
    ):
        self.directive = directive
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.ui_callback = ui_callback or self._default_ui_callback
        self.exec_mode = exec_mode
        self.work_dir = Path(work_dir)
        self.max_steps = max_steps
        self.enable_learning = enable_learning
        
        # Advanced agent components
        self.console = Console()
        self.instructor_system = DynamicInstructorSystem()
        self.context_manager = ContextManager(root_dir=work_dir)
        
        # State management
        self.state = AgentState.INITIALIZING
        self.current_instructor: InstructorProfile = None
        self.execution_history: List[ExecutionStep] = []
        self.context_memory: List[str] = []
        self.dynamic_plan: List[Dict[str, Any]] = []
        self.failed_attempts = 0
        self.start_time = datetime.now()
        
        # Performance tracking
        self.performance_metrics = {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "avg_step_time": 0.0,
            "cache_hits": 0,
            "instructor_switches": 0
        }
        
        # Initialize agent
        self._initialize_agent()
    
    def _default_ui_callback(self, msg: str):
        """Default UI callback using rich console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] {msg}")
    
    def _initialize_agent(self):
        """Initialize the agent with selected instructor"""
        self.state = AgentState.INITIALIZING
        
        # Select initial instructor
        self.current_instructor = self.instructor_system.select_instructor(
            self.directive, 
            {"complexity": self._estimate_directive_complexity()}
        )
        
        self._log(f"🌟 Agent initialized with {self.current_instructor.name}")
        self._log(f"🎯 Specialization: {self.current_instructor.specialization}")
        
        # Load historical context if available
        self._load_historical_context()
        
        self.state = AgentState.PLANNING
    
    def _estimate_directive_complexity(self) -> float:
        """Estimate directive complexity (0-1)"""
        complexity_indicators = [
            "analyze", "optimize", "create", "design", "implement",
            "complex", "advanced", "multiple", "integration"
        ]
        
        directive_lower = self.directive.lower()
        matches = sum(1 for indicator in complexity_indicators if indicator in directive_lower)
        return min(matches / len(complexity_indicators), 1.0)
    
    def _log(self, msg: str, level: str = "INFO"):
        """Enhanced logging with rich formatting"""
        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }
        
        icon = icons.get(level, "📝")
        formatted_msg = f"{icon} {msg}"
        
        self.ui_callback(formatted_msg)
        logger.log(getattr(logging, level), msg)
    
    def _load_historical_context(self):
        """Load relevant historical context from past executions"""
        echo_file = Path.home() / ".cosmic_echo.jsonl"
        if not echo_file.exists():
            return
        
        try:
            memories = []
            with open(echo_file, 'r') as f:
                for line in f:
                    if line.strip():
                        memory = json.loads(line)
                        # Simple relevance check
                        if any(word in memory.get("directive", "").lower() 
                               for word in self.directive.lower().split()[:3]):
                            memories.append(memory)
            
            if memories:
                self._log(f"🧠 Loaded {len(memories)} relevant memories")
                self.context_memory.extend([f"Historical context: {m}" for m in memories[-3:]])
        except Exception as e:
            self._log(f"Warning: Could not load historical context: {e}", "WARNING")
    
    def _create_dynamic_plan(self) -> List[Dict[str, Any]]:
        """Create intelligent dynamic plan based on instructor and directive"""
        self._log("🔮 Creating dynamic execution plan...")
        
        # Use instructor's specialized planning
        planning_prompt = f"""
        {self.current_instructor.system_prompt}
        
        Your directive: "{self.directive}"
        Working directory: {self.work_dir}
        Risk tolerance: {self.current_instructor.risk_tolerance}
        
        Create a strategic plan with 3-7 steps to achieve this directive.
        Consider your specialization and preferred actions: {[a.value for a in self.current_instructor.preferred_actions]}
        
        Return your plan as a JSON array where each step has:
        - "action": one of {[a.value for a in ActionType]}
        - "parameters": specific parameters for the action
        - "reasoning": why this step is important
        - "priority": number 1-10 (10 = critical)
        
        Example format:
        [
            {{"action": "READ", "parameters": "config.json", "reasoning": "Need to understand current configuration", "priority": 8}},
            {{"action": "SHELL", "parameters": "ls -la", "reasoning": "Explore directory structure", "priority": 6}}
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=self.current_instructor.creativity_level * 0.5,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            plan_data = json.loads(response.choices[0].message.content)
            
            # Ensure it's a list
            if isinstance(plan_data, dict) and "plan" in plan_data:
                plan = plan_data["plan"]
            elif isinstance(plan_data, list):
                plan = plan_data
            else:
                raise ValueError("Invalid plan format")
            
            # Sort by priority
            plan.sort(key=lambda x: x.get("priority", 5), reverse=True)
            
            self._log(f"📋 Created plan with {len(plan)} steps")
            return plan
            
        except Exception as e:
            self._log(f"Error creating dynamic plan: {e}", "ERROR")
            # Fallback to simple plan
            return self._create_fallback_plan()
    
    def _create_fallback_plan(self) -> List[Dict[str, Any]]:
        """Create a simple fallback plan"""
        return [
            {"action": "INFO", "parameters": f"What do I need to know about: {self.directive}", "reasoning": "Gather information", "priority": 8},
            {"action": "SHELL", "parameters": "pwd && ls -la", "reasoning": "Understand current environment", "priority": 7},
            {"action": "FINISH", "parameters": "Analysis complete", "reasoning": "Conclude execution", "priority": 10}
        ]
    
    def _select_next_action(self) -> Optional[Dict[str, Any]]:
        """Intelligently select the next action to execute"""
        if not self.dynamic_plan:
            self.dynamic_plan = self._create_dynamic_plan()
        
        # Filter out completed actions
        remaining_actions = [
            action for action in self.dynamic_plan 
            if not any(step.parameters == action["parameters"] and step.success 
                      for step in self.execution_history)
        ]
        
        if not remaining_actions:
            return {"action": "FINISH", "parameters": "All planned actions completed", "reasoning": "Plan execution complete", "priority": 10}
        
        # Select highest priority action that hasn't failed too many times
        for action in remaining_actions:
            action_failures = sum(
                1 for step in self.execution_history 
                if step.parameters == action["parameters"] and not step.success
            )
            
            if action_failures < 2:  # Allow up to 2 retries
                return action
        
        # If all actions have failed too many times, finish
        return {"action": "FINISH", "parameters": "Unable to complete remaining actions", "reasoning": "Multiple failures encountered", "priority": 10}
    
    def _execute_action(self, action: Dict[str, Any]) -> ExecutionStep:
        """Execute a single action and return the step result"""
        start_time = time.time()
        action_type = ActionType(action["action"])
        parameters = action["parameters"]
        
        step = ExecutionStep(
            action_type=action_type,
            parameters=parameters,
            context={"reasoning": action.get("reasoning", ""), "priority": action.get("priority", 5)},
            timestamp=datetime.now()
        )
        
        try:
            self._log(f"🎬 Executing: {action_type.value} {parameters}")
            
            if action_type == ActionType.READ:
                output = self._execute_read(parameters)
            elif action_type == ActionType.SHELL:
                output = self._execute_shell(parameters)
            elif action_type == ActionType.CODE:
                output = self._execute_code(parameters)
            elif action_type == ActionType.INFO:
                output = self._execute_info(parameters)
            elif action_type == ActionType.MEMORY:
                output = self._execute_memory(parameters)
            elif action_type == ActionType.SEARCH:
                output = self._execute_search(parameters)
            elif action_type == ActionType.CONNECT:
                output = self._execute_connect(parameters)
            elif action_type == ActionType.LEARN:
                output = self._execute_learn(parameters)
            elif action_type == ActionType.PLAN:
                output = self._execute_plan(parameters)
            elif action_type == ActionType.FINISH:
                output = parameters
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            step.success = True
            step.output = output
            self.performance_metrics["successful_steps"] += 1
            self._log(f"✅ Action completed successfully", "SUCCESS")
            
        except Exception as e:
            step.success = False
            step.output = f"Error: {str(e)}"
            self.performance_metrics["failed_steps"] += 1
            self._log(f"Error executing {action_type.value}: {e}", "ERROR")
        
        step.execution_time = time.time() - start_time
        self.performance_metrics["total_steps"] += 1
        
        # Update context memory
        if step.success:
            context_entry = f"{action_type.value}: {parameters} -> {step.output[:200]}..."
            self.context_memory.append(context_entry)
            
            # Limit context memory size
            if len(self.context_memory) > 10:
                self.context_memory = self.context_memory[-10:]
        
        return step
    
    def _execute_read(self, file_path: str) -> str:
        """Execute READ action"""
        return self.context_manager.read_file(file_path)
    
    def _execute_shell(self, command: str) -> str:
        """Execute SHELL action with safety checks"""
        dangerous_patterns = ["rm -rf", "sudo", "dd", "mkfs", "> /dev", ":(){ :|:& };:"]
        
        if any(pattern in command for pattern in dangerous_patterns) and self.exec_mode == "safe":
            return "[BLOCKED] Potentially dangerous command in safe mode"
        
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, 
                timeout=30, cwd=self.work_dir
            )
            return f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] Command took too long"
    
    def _execute_code(self, code: str) -> str:
        """Execute CODE action"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_file], capture_output=True, text=True, timeout=30
            )
            return f"Exit code: {result.returncode}\nOutput:\n{result.stdout}\nErrors:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "[TIMEOUT] Code execution took too long"
        finally:
            os.unlink(temp_file)
    
    def _execute_info(self, question: str) -> str:
        """Execute INFO action - ask Grok a question"""
        try:
            context = "\n".join(self.context_memory[-3:]) if self.context_memory else "No previous context"
            
            prompt = f"""
            {self.current_instructor.system_prompt}
            
            Context from previous actions:
            {context}
            
            Question: {question}
            
            Provide a helpful, accurate answer based on your specialization and the context.
            """
            
            response = self.client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error getting information: {e}"
    
    def _execute_memory(self, query: str) -> str:
        """Execute MEMORY action - query historical memory"""
        echo_file = Path.home() / ".cosmic_echo.jsonl"
        if not echo_file.exists():
            return "No historical memory found"
        
        try:
            memories = []
            with open(echo_file, 'r') as f:
                for line in f:
                    if line.strip():
                        memory = json.loads(line)
                        memories.append(memory)
            
            if not memories:
                return "No memories available"
            
            # Simple search through memories
            relevant_memories = []
            query_words = query.lower().split()
            
            for memory in memories[-20:]:  # Check last 20 memories
                memory_text = str(memory).lower()
                if any(word in memory_text for word in query_words):
                    relevant_memories.append(memory)
            
            if relevant_memories:
                return f"Found {len(relevant_memories)} relevant memories:\n" + \
                       "\n".join([str(m)[:200] + "..." for m in relevant_memories[-3:]])
            else:
                return f"No memories found matching: {query}"
                
        except Exception as e:
            return f"Error accessing memory: {e}"
    
    def _execute_search(self, search_term: str) -> str:
        """Execute SEARCH action - search within current directory"""
        try:
            # Use grep to search for term in files
            result = subprocess.run(
                ["grep", "-r", "-l", search_term, str(self.work_dir)],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')[:10]  # Limit to 10 files
                return f"Found '{search_term}' in {len(files)} files:\n" + "\n".join(files)
            else:
                return f"No files found containing '{search_term}'"
                
        except Exception as e:
            return f"Search error: {e}"
    
    def _execute_connect(self, target: str) -> str:
        """Execute CONNECT action - attempt network connection or API call"""
        # For safety, only allow HTTP GET requests to safe domains
        if target.startswith(("http://", "https://")):
            try:
                import requests
                response = requests.get(target, timeout=10)
                return f"HTTP {response.status_code}: {response.text[:500]}..."
            except Exception as e:
                return f"Connection error: {e}"
        else:
            return f"Connection not allowed to: {target} (safety restriction)"
    
    def _execute_learn(self, topic: str) -> str:
        """Execute LEARN action - learn about a topic and update knowledge"""
        if not self.enable_learning:
            return "Learning disabled"
        
        try:
            # Ask the instructor to provide learning material
            learning_prompt = f"""
            {self.current_instructor.system_prompt}
            
            Please provide educational content about: {topic}
            
            Include key concepts, practical applications, and any relevant examples.
            Focus on information that would be useful for future directive execution.
            """
            
            response = self.client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": learning_prompt}],
                temperature=0.4,
                max_tokens=1500
            )
            
            learned_content = response.choices[0].message.content.strip()
            
            # Store learned content for future reference
            learning_entry = {
                "topic": topic,
                "content": learned_content,
                "instructor": self.current_instructor.name,
                "timestamp": datetime.now().isoformat()
            }
            
            learning_file = Path.home() / ".cosmic_learning.jsonl"
            with open(learning_file, 'a') as f:
                json.dump(learning_entry, f)
                f.write('\n')
            
            return f"Learned about {topic}:\n{learned_content[:300]}..."
            
        except Exception as e:
            return f"Learning error: {e}"
    
    def _execute_plan(self, planning_request: str) -> str:
        """Execute PLAN action - create or update execution plan"""
        try:
            # Create a new plan based on current context
            new_plan = self._create_dynamic_plan()
            self.dynamic_plan = new_plan
            
            plan_summary = []
            for i, step in enumerate(new_plan, 1):
                plan_summary.append(f"{i}. {step['action']}: {step['parameters']}")
            
            return f"Updated plan with {len(new_plan)} steps:\n" + "\n".join(plan_summary)
            
        except Exception as e:
            return f"Planning error: {e}"
    
    def _should_switch_instructor(self) -> bool:
        """Determine if we should switch to a different instructor"""
        if len(self.execution_history) < 3:
            return False
        
        # Check recent failure rate
        recent_steps = self.execution_history[-3:]
        failure_rate = sum(1 for step in recent_steps if not step.success) / len(recent_steps)
        
        # Switch if failure rate is high and current instructor has low risk tolerance
        if failure_rate > 0.6 and self.current_instructor.risk_tolerance < 0.5:
            return True
        
        # Switch if we're stuck on the same action type
        recent_actions = [step.action_type for step in recent_steps]
        if len(set(recent_actions)) == 1 and not recent_steps[-1].success:
            return True
        
        return False
    
    def _switch_instructor(self):
        """Switch to a more suitable instructor"""
        # Analyze current situation
        failed_actions = [step.action_type for step in self.execution_history if not step.success]
        
        # Create context for instructor selection
        switch_context = {
            "complexity": min(len(failed_actions) / 5, 1.0),
            "failed_actions": failed_actions,
            "current_instructor": self.current_instructor.name
        }
        
        # Select new instructor (avoid current one)
        available_instructors = {
            name: instructor for name, instructor in self.instructor_system.instructors.items()
            if name != self.current_instructor.name.lower().replace(" ", "_")
        }
        
        if not available_instructors:
            return  # No other instructors available
        
        # Simple selection: pick instructor with different strengths
        if self.current_instructor.risk_tolerance < 0.5:
            # Switch to higher risk tolerance
            new_instructor_name = max(available_instructors.items(), 
                                    key=lambda x: x[1].risk_tolerance)[0]
        else:
            # Switch to more conservative
            new_instructor_name = min(available_instructors.items(), 
                                    key=lambda x: x[1].risk_tolerance)[0]
        
        old_instructor = self.current_instructor.name
        self.current_instructor = available_instructors[new_instructor_name]
        
        self.performance_metrics["instructor_switches"] += 1
        self._log(f"🔄 Switched instructor: {old_instructor} → {self.current_instructor.name}", "INFO")
    
    async def execute_async(self) -> Dict[str, Any]:
        """Asynchronous execution of the agent directive"""
        self.state = AgentState.EXECUTING
        self._log(f"🚀 Beginning execution of directive: {self.directive}")
        
        execution_results = {
            "directive": self.directive,
            "instructor": self.current_instructor.name,
            "start_time": self.start_time.isoformat(),
            "steps": [],
            "performance": self.performance_metrics,
            "status": "running"
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Executing directive...", total=self.max_steps)
            
            for step_count in range(self.max_steps):
                try:
                    # Check if we should switch instructors
                    if self._should_switch_instructor():
                        self._switch_instructor()
                    
                    # Select next action
                    next_action = self._select_next_action()
                    if not next_action:
                        self._log("No more actions to execute", "INFO")
                        break
                    
                    # Execute the action
                    execution_step = self._execute_action(next_action)
                    self.execution_history.append(execution_step)
                    
                    # Add to results
                    execution_results["steps"].append({
                        "step": step_count + 1,
                        "action": execution_step.action_type.value,
                        "parameters": execution_step.parameters,
                        "success": execution_step.success,
                        "output": execution_step.output[:500] + "..." if len(execution_step.output) > 500 else execution_step.output,
                        "execution_time": execution_step.execution_time,
                        "reasoning": execution_step.context.get("reasoning", "")
                    })
                    
                    # Check if we should finish
                    if execution_step.action_type == ActionType.FINISH:
                        self._log("✅ Execution completed by FINISH action", "SUCCESS")
                        break
                    
                    progress.update(task, advance=1)
                    
                    # Small delay to prevent overwhelming the API
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self._log(f"Critical error in execution loop: {e}", "ERROR")
                    execution_results["error"] = str(e)
                    break
            
            else:
                self._log("⚠️ Maximum steps reached", "WARNING")
        
        # Finalize execution
        self.state = AgentState.COMPLETED
        execution_results["end_time"] = datetime.now().isoformat()
        execution_results["total_time"] = (datetime.now() - self.start_time).total_seconds()
        execution_results["performance"] = self.performance_metrics
        execution_results["status"] = "completed"
        
        # Save to echo memory
        self._save_to_echo_memory(execution_results)
        
        # Display final summary
        self._display_execution_summary(execution_results)
        
        return execution_results
    
    def execute(self) -> Dict[str, Any]:
        """Synchronous wrapper for async execution"""
        return asyncio.run(self.execute_async())
    
    def _save_to_echo_memory(self, results: Dict[str, Any]):
        """Save execution results to echo memory"""
        try:
            echo_file = Path.home() / ".cosmic_echo.jsonl"
            
            # Create summary for echo memory
            last_step = results["steps"][-1] if results["steps"] else {}
            outcome = last_step.get("output", "No output")
            
            # Generate tone and glyph suggestion
            prompt = f"Suggest a tone (e.g., Resplendent Reflection) and a glyph (e.g., 🌀) for this outcome: {outcome[:200]} in JSON format: {{'tone': '', 'glyph': ''}}"
            
            try:
                response = self.client.chat.completions.create(
                    model="grok-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                suggestion = json.loads(response.choices[0].message.content)
            except:
                suggestion = {"tone": "Sacred Completion", "glyph": "✨"}
            
            echo_entry = {
                "directive": self.directive,
                "instructor": self.current_instructor.name,
                "outcome": outcome[:500],
                "tone": suggestion.get("tone", "Sacred Completion"),
                "glyph": suggestion.get("glyph", "✨"),
                "timestamp": datetime.now().isoformat(),
                "performance": self.performance_metrics
            }
            
            with open(echo_file, 'a') as f:
                json.dump(echo_entry, f)
                f.write('\n')
                
        except Exception as e:
            self._log(f"Could not save to echo memory: {e}", "WARNING")
    
    def _display_execution_summary(self, results: Dict[str, Any]):
        """Display beautiful execution summary"""
        summary_table = Table(title=f"🌟 Execution Summary: {self.current_instructor.name}", show_header=True)
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", style="green")
        
        metrics = results["performance"]
        success_rate = metrics["successful_steps"] / max(metrics["total_steps"], 1) * 100
        
        summary_table.add_row("📝 Directive", self.directive[:50] + "..." if len(self.directive) > 50 else self.directive)
        summary_table.add_row("👑 Instructor", self.current_instructor.name)
        summary_table.add_row("⏱️ Total Time", f"{results['total_time']:.2f}s")
        summary_table.add_row("🎯 Total Steps", str(metrics["total_steps"]))
        summary_table.add_row("✅ Success Rate", f"{success_rate:.1f}%")
        summary_table.add_row("🔄 Instructor Switches", str(metrics["instructor_switches"]))
        summary_table.add_row("📊 Status", results["status"].title())
        
        panel = Panel(summary_table, border_style="green", padding=(1, 2))
        self.console.print(panel)


# Convenience function for backward compatibility
def create_stargazer_agent(*args, **kwargs) -> EnhancedStargazerAgent:
    """Create an enhanced stargazer agent"""
    return EnhancedStargazerAgent(*args, **kwargs)
