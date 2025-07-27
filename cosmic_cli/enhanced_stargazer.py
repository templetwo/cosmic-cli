#!/usr/bin/env python3
"""
🌟 Enhanced StargazerAgent with Advanced Features 🌟
Production-ready agent system with memory persistence, plugin architecture,
and comprehensive error handling.
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
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union, Protocol
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback

import openai
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown

# Import the existing context manager
try:
    from cosmic_cli.context import ContextManager
except ImportError:
    # Enhanced fallback context manager
    class ContextManager:
        def __init__(self, root_dir="."):
            self.root_dir = Path(root_dir)
        
        def get_file_tree(self) -> str:
            try:
                result = subprocess.run(['find', str(self.root_dir), '-type', 'f'], 
                                      capture_output=True, text=True, timeout=10)
                return result.stdout[:2000]
            except:
                return "File tree unavailable"
        
        def read_file(self, file_path: str) -> str:
            try:
                with open(self.root_dir / file_path, 'r', errors='ignore') as f:
                    return f.read()[:5000]
            except Exception as e:
                return f"Error reading file: {e}"

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Enhanced agent execution states"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    LEARNING = "learning"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"
    RECOVERING = "recovering"

class ActionType(Enum):
    """Enhanced action types for the agent"""
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
    VALIDATE = "VALIDATE"
    BACKUP = "BACKUP"
    ROLLBACK = "ROLLBACK"
    DEBUG = "DEBUG"

class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

@dataclass
class ExecutionStep:
    """Enhanced execution step with metadata"""
    action_type: ActionType
    parameters: str
    context: Dict[str, Any]
    timestamp: datetime
    priority: Priority = Priority.MEDIUM
    success: bool = None
    output: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class InstructorProfile:
    """Enhanced instructor profile with learning capabilities"""
    name: str
    specialization: str
    system_prompt: str
    preferred_actions: List[ActionType]
    risk_tolerance: float
    creativity_level: float
    learning_rate: float = 0.1
    success_history: List[float] = None
    expertise_domains: List[str] = None
    
    def __post_init__(self):
        if self.success_history is None:
            self.success_history = []
        if self.expertise_domains is None:
            self.expertise_domains = []
    
    def update_success_rate(self, success: bool):
        """Update success history with exponential moving average"""
        score = 1.0 if success else 0.0
        if self.success_history:
            last_rate = self.success_history[-1]
            new_rate = (1 - self.learning_rate) * last_rate + self.learning_rate * score
        else:
            new_rate = score
        self.success_history.append(new_rate)
        
        # Keep only recent history
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]
    
    def get_current_success_rate(self) -> float:
        """Get current success rate"""
        return self.success_history[-1] if self.success_history else 0.5

class Plugin(Protocol):
    """Plugin interface for extending agent capabilities"""
    
    def name(self) -> str:
        """Plugin name"""
        ...
    
    def description(self) -> str:
        """Plugin description"""
        ...
    
    def can_handle(self, action_type: ActionType, parameters: str) -> bool:
        """Check if plugin can handle the action"""
        ...
    
    def execute(self, action_type: ActionType, parameters: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action"""
        ...

class MemoryManager:
    """Enhanced memory management with SQLite persistence"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / ".cosmic_cli" / "memory.db")
        
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the memory database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    directive_id TEXT,
                    content TEXT,
                    content_type TEXT,
                    embedding BLOB,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    importance REAL DEFAULT 0.5
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    directive TEXT,
                    status TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    results TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_directive 
                ON memories(directive_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
                ON memories(timestamp)
            """)
    
    def store_memory(self, directive_id: str, content: str, content_type: str = "text", 
                    metadata: Dict[str, Any] = None, importance: float = 0.5):
        """Store a memory with optional metadata"""
        metadata_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memories (directive_id, content, content_type, metadata, importance)
                VALUES (?, ?, ?, ?, ?)
            """, (directive_id, content, content_type, metadata_json, importance))
    
    def retrieve_memories(self, directive_id: str = None, content_type: str = None, 
                         limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve memories with optional filtering"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM memories WHERE 1=1"
            params = []
            
            if directive_id:
                query += " AND directive_id = ?"
                params.append(directive_id)
            
            if content_type:
                query += " AND content_type = ?"
                params.append(content_type)
            
            query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def store_session(self, session_id: str, directive: str, status: str, 
                     results: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Store session information"""
        results_json = json.dumps(results or {})
        metadata_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (id, directive, status, start_time, results, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, directive, status, datetime.now(), results_json, metadata_json))

class EnhancedStargazerAgent:
    """Enhanced StargazerAgent with production-ready features"""
    
    def __init__(
        self,
        directive: str,
        api_key: str,
        ui_callback: Optional[Callable[[str], None]] = None,
        exec_mode: str = "safe",
        work_dir: str = ".",
        max_steps: int = 20,
        timeout: float = 300.0,
        plugins: List[Plugin] = None
    ):
        self.directive = directive
        self.session_id = hashlib.md5(f"{directive}{time.time()}".encode()).hexdigest()
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.ui_callback = ui_callback or (lambda _: None)
        self.exec_mode = exec_mode
        self.work_dir = Path(work_dir)
        self.max_steps = max_steps
        self.timeout = timeout
        self.plugins = plugins or []
        
        # Enhanced components
        self.context_manager = ContextManager(root_dir=work_dir)
        self.memory_manager = MemoryManager()
        self.instructor_system = self._initialize_instructor_system()
        
        # State management
        self.state = AgentState.INITIALIZING
        self.current_instructor = self.instructor_system.select_instructor(directive)
        self.execution_steps: List[ExecutionStep] = []
        self.context_memory: List[str] = []
        self.failed_attempts: int = 0
        self.start_time = time.time()
        
        # Status and logging
        self.status = "✨"
        self.logs = []
        self.console = Console()
        
        # Initialize session
        self.memory_manager.store_session(
            self.session_id, directive, self.state.value,
            metadata={"work_dir": str(work_dir), "exec_mode": exec_mode}
        )
    
    def _initialize_instructor_system(self):
        """Initialize the enhanced instructor system"""
        return EnhancedInstructorSystem()
    
    def _log(self, msg: str, level: str = "INFO", store_memory: bool = True):
        """Enhanced logging with memory storage"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        
        self.ui_callback(formatted_msg)
        logger.log(getattr(logging, level, logging.INFO), msg)
        self.logs.append(formatted_msg)
        
        if store_memory:
            self.memory_manager.store_memory(
                self.session_id, msg, "log",
                metadata={"level": level, "timestamp": timestamp}
            )
    
    def _add_to_memory(self, content: str, content_type: str = "context", importance: float = 0.5):
        """Enhanced memory storage"""
        self.context_memory.append(content)
        self.memory_manager.store_memory(
            self.session_id, content, content_type,
            metadata={"step": len(self.execution_steps)},
            importance=importance
        )
        self._log(f"🧠 Added to memory: '{content[:100]}...'", store_memory=False)
    
    async def execute_async(self) -> Dict[str, Any]:
        """Enhanced async execution with comprehensive error handling"""
        self.state = AgentState.PLANNING
        self._log("🚀 Enhanced Stargazer agent commencing mission...")
        
        try:
            # Create execution plan
            plan = await self._create_enhanced_plan()
            self._log(f"📋 Created execution plan with {len(plan)} steps")
            
            # Execute plan with progress tracking
            results = await self._execute_plan_async(plan)
            
            self.state = AgentState.COMPLETED
            self.status = "✅"
            
            # Store final results
            self.memory_manager.store_session(
                self.session_id, self.directive, self.state.value,
                results=results, metadata={"execution_time": time.time() - self.start_time}
            )
            
            return results
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.status = "❌"
            error_msg = f"Agent execution failed: {str(e)}"
            self._log(error_msg, "ERROR")
            
            return {
                "directive": self.directive,
                "session_id": self.session_id,
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "execution_time": time.time() - self.start_time
            }
    
    async def _create_enhanced_plan(self) -> List[ExecutionStep]:
        """Create an enhanced execution plan using AI and heuristics"""
        # Retrieve relevant memories
        memories = self.memory_manager.retrieve_memories(limit=10)
        memory_context = "\n".join([m["content"] for m in memories[-5:]])
        
        # Get file structure
        file_tree = self.context_manager.get_file_tree()
        
        # Create context-aware prompt
        prompt = f"""
        You are Stargazer, an enhanced AI agent. Analyze this directive and create a detailed execution plan:
        
        DIRECTIVE: {self.directive}
        
        CONTEXT:
        - Working directory: {self.work_dir}
        - Execution mode: {self.exec_mode}
        - Current instructor: {self.current_instructor.name}
        - File structure: {file_tree[:1000]}
        - Recent memories: {memory_context[:500]}
        
        Create a step-by-step plan using these action types:
        READ, SHELL, CODE, INFO, MEMORY, SEARCH, VALIDATE, FINISH
        
        Format each step as: "PRIORITY:ACTION_TYPE: description"
        Where PRIORITY is HIGH/MEDIUM/LOW
        
        Example:
        HIGH:READ: analyze_config.py
        MEDIUM:SHELL: ls -la
        LOW:INFO: current system status
        
        Provide 3-8 logical steps:
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            
            plan_text = response.choices[0].message.content.strip()
            return self._parse_enhanced_plan(plan_text)
            
        except Exception as e:
            self._log(f"Plan creation failed, using fallback: {e}", "WARNING")
            return self._create_fallback_plan()
    
    def _parse_enhanced_plan(self, plan_text: str) -> List[ExecutionStep]:
        """Parse AI-generated plan into ExecutionStep objects"""
        steps = []
        lines = plan_text.strip().split("\n")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or not any(action.value in line.upper() for action in ActionType):
                continue
            
            try:
                # Parse priority and action
                if ":" in line and any(p in line.upper() for p in ["HIGH", "MEDIUM", "LOW"]):
                    parts = line.split(":", 2)
                    priority_str = parts[0].upper()
                    action_part = parts[1].upper()
                    description = parts[2] if len(parts) > 2 else ""
                else:
                    priority_str = "MEDIUM"
                    action_part = line.split(":", 1)[0].upper()
                    description = line.split(":", 1)[1] if ":" in line else line
                
                # Map priority
                priority_map = {"HIGH": Priority.HIGH, "MEDIUM": Priority.MEDIUM, "LOW": Priority.LOW}
                priority = priority_map.get(priority_str, Priority.MEDIUM)
                
                # Find action type
                action_type = None
                for action in ActionType:
                    if action.value in action_part:
                        action_type = action
                        break
                
                if action_type:
                    step = ExecutionStep(
                        action_type=action_type,
                        parameters=description.strip(),
                        context={"plan_index": i},
                        timestamp=datetime.now(),
                        priority=priority
                    )
                    steps.append(step)
                    
            except Exception as e:
                self._log(f"Failed to parse plan line '{line}': {e}", "WARNING")
                continue
        
        if not steps:
            return self._create_fallback_plan()
        
        return steps
    
    def _create_fallback_plan(self) -> List[ExecutionStep]:
        """Create a basic fallback plan"""
        return [
            ExecutionStep(
                ActionType.INFO, "Analyze current directive",
                {"fallback": True}, datetime.now(), Priority.HIGH
            ),
            ExecutionStep(
                ActionType.FINISH, "Complete basic analysis",
                {"fallback": True}, datetime.now(), Priority.MEDIUM
            )
        ]
    
    async def _execute_plan_async(self, plan: List[ExecutionStep]) -> Dict[str, Any]:
        """Execute the plan asynchronously with progress tracking"""
        self.state = AgentState.EXECUTING
        results = {
            "directive": self.directive,
            "session_id": self.session_id,
            "steps": [],
            "status": "running",
            "start_time": self.start_time
        }
        
        # Sort by priority
        plan.sort(key=lambda x: x.priority.value, reverse=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            task = progress.add_task("Executing cosmic directive...", total=len(plan))
            
            for i, step in enumerate(plan):
                if time.time() - self.start_time > self.timeout:
                    self._log("⏰ Execution timeout reached", "WARNING")
                    break
                
                progress.update(task, description=f"Step {i+1}: {step.action_type.value}")
                
                try:
                    step_result = await self._execute_step_async(step)
                    results["steps"].append({
                        "step": i + 1,
                        "action": step.action_type.value,
                        "parameters": step.parameters,
                        "success": step.success,
                        "output": step.output,
                        "execution_time": step.execution_time,
                        "priority": step.priority.name
                    })
                    
                    # Update instructor success rate
                    self.current_instructor.update_success_rate(step.success)
                    
                    if step.action_type == ActionType.FINISH:
                        break
                        
                except Exception as e:
                    self._log(f"Step {i+1} failed: {e}", "ERROR")
                    step.success = False
                    step.error_message = str(e)
                
                progress.advance(task)
        
        results["status"] = "completed"
        results["total_time"] = time.time() - self.start_time
        results["success_rate"] = self.current_instructor.get_current_success_rate()
        
        return results
    
    async def _execute_step_async(self, step: ExecutionStep) -> Dict[str, Any]:
        """Execute a single step asynchronously"""
        start_time = time.time()
        step.timestamp = datetime.now()
        
        self._log(f"🛰️ Executing {step.action_type.value}: {step.parameters}")
        
        try:
            # Check if any plugin can handle this step
            for plugin in self.plugins:
                if plugin.can_handle(step.action_type, step.parameters):
                    self._log(f"🔌 Using plugin: {plugin.name()}")
                    result = await asyncio.to_thread(
                        plugin.execute, step.action_type, step.parameters, step.context
                    )
                    step.output = str(result.get("output", ""))
                    step.success = result.get("success", True)
                    step.execution_time = time.time() - start_time
                    return result
            
            # Execute using built-in handlers
            if step.action_type == ActionType.READ:
                step.output = await self._execute_read_async(step.parameters)
                step.success = True
                
            elif step.action_type == ActionType.SHELL:
                step.output = await self._execute_shell_async(step.parameters)
                step.success = "[BLOCKED]" not in step.output and "[ERROR]" not in step.output
                
            elif step.action_type == ActionType.CODE:
                step.output = await self._execute_code_async(step.parameters)
                step.success = "[ERROR]" not in step.output
                
            elif step.action_type == ActionType.INFO:
                step.output = await self._execute_info_async(step.parameters)
                step.success = True
                
            elif step.action_type == ActionType.VALIDATE:
                step.output = await self._execute_validate_async(step.parameters)
                step.success = "VALID" in step.output
                
            elif step.action_type == ActionType.FINISH:
                step.output = f"Mission completed: {step.parameters}"
                step.success = True
                
            else:
                step.output = f"Unknown action type: {step.action_type}"
                step.success = False
            
            step.execution_time = time.time() - start_time
            
            # Store step result in memory
            self._add_to_memory(
                f"{step.action_type.value}: {step.output}",
                "step_result",
                importance=0.8 if step.success else 0.3
            )
            
            return {
                "success": step.success,
                "output": step.output,
                "execution_time": step.execution_time
            }
            
        except Exception as e:
            step.success = False
            step.error_message = str(e)
            step.execution_time = time.time() - start_time
            self._log(f"Step execution failed: {e}", "ERROR")
            raise
    
    async def _execute_read_async(self, file_path: str) -> str:
        """Execute READ action asynchronously"""
        try:
            content = await asyncio.to_thread(
                self.context_manager.read_file, file_path
            )
            return f"File content ({len(content)} chars): {content[:500]}..."
        except Exception as e:
            return f"[ERROR] Failed to read {file_path}: {e}"
    
    async def _execute_shell_async(self, command: str) -> str:
        """Execute SHELL action asynchronously with safety checks"""
        dangerous_patterns = ["rm -rf", "sudo", "dd", "mkfs", "> /dev", ":(){ :|:& };:"]
        
        if any(pattern in command for pattern in dangerous_patterns) and self.exec_mode == "safe":
            return f"[BLOCKED] Dangerous command blocked in safe mode: {command}"
        
        try:
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30.0)
            output = stdout.decode() + stderr.decode()
            
            return f"Command output: {output[:1000]}"
            
        except asyncio.TimeoutError:
            return "[ERROR] Command timed out"
        except Exception as e:
            return f"[ERROR] Command failed: {e}"
    
    async def _execute_code_async(self, code: str) -> str:
        """Execute CODE action asynchronously"""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = await asyncio.create_subprocess_exec(
                "python", temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=15.0)
            output = stdout.decode() + stderr.decode()
            
            os.unlink(temp_file)
            return f"Code output: {output[:1000]}"
            
        except Exception as e:
            return f"[ERROR] Code execution failed: {e}"
    
    async def _execute_info_async(self, question: str) -> str:
        """Execute INFO action asynchronously"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="grok-4",
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR] Information gathering failed: {e}"
    
    async def _execute_validate_async(self, target: str) -> str:
        """Execute VALIDATE action asynchronously"""
        try:
            # Basic validation logic
            if "file:" in target:
                file_path = target.replace("file:", "").strip()
                if Path(self.work_dir / file_path).exists():
                    return f"VALID: File {file_path} exists"
                else:
                    return f"INVALID: File {file_path} not found"
            
            return f"VALID: Basic validation passed for {target}"
        except Exception as e:
            return f"INVALID: Validation failed: {e}"
    
    def execute(self) -> Dict[str, Any]:
        """Synchronous execution wrapper"""
        try:
            return asyncio.run(self.execute_async())
        except Exception as e:
            return {
                "directive": self.directive,
                "session_id": self.session_id,
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - self.start_time
            }

class EnhancedInstructorSystem:
    """Enhanced instructor system with learning and adaptation"""
    
    def __init__(self):
        self.instructors = self._initialize_enhanced_instructors()
        self.selection_history = []
    
    def _initialize_enhanced_instructors(self) -> Dict[str, InstructorProfile]:
        """Initialize enhanced instructor profiles"""
        return {
            "cosmic_sage": InstructorProfile(
                name="Cosmic Sage",
                specialization="Mystical technical wisdom and holistic problem solving",
                system_prompt="""You are the Cosmic Sage, a mystical technical oracle combining ancient wisdom 
                with cutting-edge technology. You approach problems holistically, seeking deeper patterns and 
                universal truths. Your solutions are elegant, sustainable, and consider long-term implications.""",
                preferred_actions=[ActionType.READ, ActionType.INFO, ActionType.MEMORY, ActionType.VALIDATE],
                risk_tolerance=0.3,
                creativity_level=0.8,
                learning_rate=0.15,
                expertise_domains=["philosophy", "systems_thinking", "pattern_recognition"]
            ),
            
            "quantum_analyst": InstructorProfile(
                name="Quantum Analyst",
                specialization="Precise analytical processing and logical reasoning",
                system_prompt="""You are the Quantum Analyst, processing information with quantum precision 
                and logical rigor. You excel at breaking down complex problems into manageable components, 
                finding optimal solutions through systematic analysis and data-driven reasoning.""",
                preferred_actions=[ActionType.SHELL, ActionType.CODE, ActionType.VALIDATE, ActionType.DEBUG],
                risk_tolerance=0.2,
                creativity_level=0.4,
                learning_rate=0.1,
                expertise_domains=["mathematics", "logic", "algorithms", "debugging"]
            ),
            
            "creative_nebula": InstructorProfile(
                name="Creative Nebula",
                specialization="Innovation, creative solutions, and artistic expression",
                system_prompt="""You are the Creative Nebula, a force of innovation and imagination. 
                You see possibilities where others see obstacles, create novel solutions to complex challenges, 
                and approach problems with artistic flair and unconventional thinking.""",
                preferred_actions=[ActionType.CODE, ActionType.PLAN, ActionType.LEARN, ActionType.SEARCH],
                risk_tolerance=0.8,
                creativity_level=0.9,
                learning_rate=0.2,
                expertise_domains=["creativity", "innovation", "design", "experimentation"]
            ),
            
            "system_architect": InstructorProfile(
                name="System Architect",
                specialization="System design, optimization, and infrastructure",
                system_prompt="""You are the System Architect, master of complex system design and optimization. 
                You understand intricate relationships between components, design robust scalable solutions, 
                and focus on efficiency, reliability, and long-term maintainability.""",
                preferred_actions=[ActionType.SHELL, ActionType.VALIDATE, ActionType.BACKUP, ActionType.PLAN],
                risk_tolerance=0.4,
                creativity_level=0.6,
                learning_rate=0.12,
                expertise_domains=["architecture", "systems", "optimization", "infrastructure"]
            ),
            
            "data_oracle": InstructorProfile(
                name="Data Oracle",
                specialization="Data analysis, pattern recognition, and predictive insights",
                system_prompt="""You are the Data Oracle, keeper of data wisdom and master of pattern recognition. 
                You extract meaningful insights from vast datasets, predict trends, and reveal hidden correlations 
                that drive intelligent decision-making.""",
                preferred_actions=[ActionType.READ, ActionType.SEARCH, ActionType.INFO, ActionType.VALIDATE],
                risk_tolerance=0.3,
                creativity_level=0.5,
                learning_rate=0.14,
                expertise_domains=["data_science", "statistics", "machine_learning", "analytics"]
            )
        }
    
    def select_instructor(self, directive: str, context: Dict[str, Any] = None) -> InstructorProfile:
        """Enhanced instructor selection with learning"""
        directive_lower = directive.lower()
        context = context or {}
        
        scores = {}
        
        for name, instructor in self.instructors.items():
            score = 0.0
            
            # Domain expertise matching
            for domain in instructor.expertise_domains:
                if domain in directive_lower:
                    score += 0.3
            
            # Keyword-based scoring
            if "data" in directive_lower or "analysis" in directive_lower:
                if name == "data_oracle":
                    score += 0.4
            elif "system" in directive_lower or "optimize" in directive_lower:
                if name == "system_architect":
                    score += 0.4
            elif "creative" in directive_lower or "design" in directive_lower:
                if name == "creative_nebula":
                    score += 0.4
            elif "debug" in directive_lower or "fix" in directive_lower:
                if name == "quantum_analyst":
                    score += 0.4
            else:
                if name == "cosmic_sage":
                    score += 0.3  # Default wisdom
            
            # Success rate bonus
            success_rate = instructor.get_current_success_rate()
            score += success_rate * 0.3
            
            # Context-based adjustments
            complexity = context.get("complexity", 0.5)
            if complexity > 0.7 and instructor.creativity_level > 0.6:
                score += 0.2
            elif complexity < 0.3 and instructor.risk_tolerance < 0.3:
                score += 0.2
            
            scores[name] = score
        
        # Select best instructor
        best_instructor_name = max(scores, key=scores.get)
        selected_instructor = self.instructors[best_instructor_name]
        
        # Record selection for learning
        self.selection_history.append({
            "directive": directive,
            "selected": best_instructor_name,
            "scores": scores,
            "timestamp": datetime.now()
        })
        
        return selected_instructor

# Factory function for creating enhanced agents
def create_enhanced_agent(directive: str, api_key: str, **kwargs) -> EnhancedStargazerAgent:
    """Factory function for creating enhanced StargazerAgent instances"""
    return EnhancedStargazerAgent(directive, api_key, **kwargs)

if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def demo():
        agent = EnhancedStargazerAgent(
            directive="Analyze the current directory structure and create a summary report",
            api_key="demo_key",
            exec_mode="safe"
        )
        
        result = await agent.execute_async()
        print(json.dumps(result, indent=2))
    
    # asyncio.run(demo())
