#!/usr/bin/env python3
"""
🚀 Optimized StargazerAgent - High-performance version
"""

import re
import asyncio
import time
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Precompiled regex patterns for maximum performance
PATTERN_INFO = re.compile(r"^(what|show|list|display)\s+(is|are|in)\s*(.+)$", re.IGNORECASE)
PATTERN_CHECK = re.compile(r"^(check|verify|test)\s+(.+)$", re.IGNORECASE)
PATTERN_READ = re.compile(r"^(read|open|view)\s+(.+\.(py|js|json|md|txt|yml|yaml))$", re.IGNORECASE)

# Response templates
TEMPLATE_INFO = [
    {"action": "INFO", "parameters": "{}", "priority": 9},
    {"action": "FINISH", "parameters": "Information provided", "priority": 10}
]
TEMPLATE_CHECK = [
    {"action": "SHELL", "parameters": "pwd && ls -la", "priority": 8},
    {"action": "INFO", "parameters": "{}", "priority": 9},
    {"action": "FINISH", "parameters": "Check complete", "priority": 10}
]
TEMPLATE_READ = [
    {"action": "READ", "parameters": "{}", "priority": 9},
    {"action": "FINISH", "parameters": "File read complete", "priority": 10}
]

class OptimizedStargazerAgent:
    """High-performance agent with freeze protection and efficient pattern matching"""
    
    def __init__(self, directive: str, api_key: str, ui_callback=None, exec_mode="safe", work_dir="."):
        self.directive = directive
        self.api_key = api_key
        self.ui_callback = ui_callback or (lambda x: print(x))
        self.exec_mode = exec_mode
        self.work_dir = Path(work_dir)
        self.start_time = time.time()
        self.timeout_global = 60.0  # Global timeout for entire execution
        self.timeout_step = 15.0    # Timeout for individual steps
        
    def _log(self, msg: str, level: str = "INFO"):
        """Thread-safe logging with exception handling"""
        try:
            self.ui_callback(msg)
            logger.log(getattr(logging, level, logging.INFO), msg)
        except Exception:
            pass  # Fail silently - logging should never break execution
    
    @lru_cache(maxsize=32)  # Cache pattern matching results
    def _match_simple_pattern(self, directive: str) -> Tuple[List[Dict[str, Any]], Optional[re.Match]]:
        """Check directive against simple patterns with caching"""
        # Check against precompiled patterns
        if match := PATTERN_INFO.match(directive):
            return TEMPLATE_INFO, match
        if match := PATTERN_CHECK.match(directive):
            return TEMPLATE_CHECK, match
        if match := PATTERN_READ.match(directive):
            return TEMPLATE_READ, match
        return [], None
    
    def _create_plan_from_match(self, template: List[Dict[str, Any]], match: re.Match) -> List[Dict[str, Any]]:
        """Create plan from template with parameter substitution"""
        plan = []
        for step in template:
            step_copy = step.copy()
            if '{}' in step_copy.get("parameters", ""):
                # Extract the most relevant capture group
                groups = match.groups()
                param_value = groups[1] if len(groups) > 1 else self.directive
                step_copy["parameters"] = step_copy["parameters"].format(param_value)
            plan.append(step_copy)
        return plan
    
    def _create_dynamic_plan(self) -> List[Dict[str, Any]]:
        """Fast plan creation with pattern matching"""
        directive_lower = self.directive.lower().strip()
        
        # Check for simple patterns with caching
        template, match = self._match_simple_pattern(directive_lower)
        
        if template and match:
            self._log("⚡ Using optimized directive template")
            return self._create_plan_from_match(template, match)
        
        # Fallback plan for complex directives
        return [
            {"action": "INFO", "parameters": self.directive, "priority": 8},
            {"action": "FINISH", "parameters": "Task completed", "priority": 10}
        ]
    
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step with timeout protection"""
        action = step.get("action", "UNKNOWN")
        params = step.get("parameters", "")
        
        try:
            if action == "INFO":
                # Simulate INFO action
                return {"success": True, "output": f"Information about: {params}"}
            
            elif action == "READ":
                # Safe file reading with timeout
                def read_file():
                    try:
                        with open(Path(self.work_dir) / params, 'r') as f:
                            return f.read(10000)  # Limit read size
                    except Exception as e:
                        return f"Error reading file: {e}"
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(read_file)
                    try:
                        content = future.result(timeout=5.0)
                        return {"success": True, "output": f"File content: {content[:100]}..."}  
                    except concurrent.futures.TimeoutError:
                        return {"success": False, "output": "File read timed out"}
            
            elif action == "SHELL":
                # This would be implemented with subprocess and timeout
                return {"success": True, "output": f"Executed: {params}"}
            
            elif action == "FINISH":
                return {"success": True, "output": params}
            
            return {"success": False, "output": f"Unknown action: {action}"}
            
        except Exception as e:
            return {"success": False, "output": f"Step execution error: {e}"}
    
    async def _execute_async(self) -> Dict[str, Any]:
        """Optimized async execution with global timeout"""
        results = {
            "directive": self.directive,
            "steps": [],
            "status": "running",
            "start_time": self.start_time
        }
        
        try:
            # Create plan without blocking main thread
            plan = await asyncio.to_thread(self._create_dynamic_plan)
            
            # Process all steps with parallel execution where possible
            step_tasks = []
            for i, step in enumerate(plan):
                if step.get("action") == "FINISH" and i > 0:
                    # Only run FINISH after other steps complete
                    continue
                    
                # Create tasks for concurrent execution
                step_tasks.append(
                    asyncio.create_task(self._process_step(i, step, results))
                )
            
            # Wait for all tasks with global timeout
            await asyncio.wait_for(asyncio.gather(*step_tasks), timeout=self.timeout_global)
            
            # Execute FINISH step if present
            finish_steps = [s for i, s in enumerate(plan) if s.get("action") == "FINISH" and i > 0]
            if finish_steps:
                await self._process_step(len(results["steps"]), finish_steps[0], results)
                
        except asyncio.TimeoutError:
            self._log("⚠️ Global execution timeout reached", "WARNING")
            results["timeout"] = True
        except Exception as e:
            self._log(f"Execution error: {e}", "ERROR")
            results["error"] = str(e)
        
        results["status"] = "completed"
        results["total_time"] = time.time() - self.start_time
        return results
    
    async def _process_step(self, step_num: int, step: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Process a single step with timeout and add to results"""
        self._log(f"Executing step {step_num+1}: {step.get('action')}")
        
        try:
            # Execute with timeout
            start = time.time()
            step_result = await asyncio.wait_for(
                self._execute_step(step),
                timeout=self.timeout_step
            )
            
            results["steps"].append({
                "step": step_num + 1,
                "action": step.get("action"),
                "parameters": step.get("parameters"),
                "success": step_result.get("success", False),
                "output": step_result.get("output", ""),
                "time": time.time() - start
            })
            
        except asyncio.TimeoutError:
            self._log(f"Step {step_num+1} timed out", "WARNING")
            results["steps"].append({
                "step": step_num + 1,
                "action": step.get("action"),
                "parameters": step.get("parameters"),
                "success": False,
                "output": "Timed out",
                "time": self.timeout_step
            })
    
    def execute(self) -> Dict[str, Any]:
        """Smart execution with event loop detection"""
        try:
            try:
                # Check if we're in an event loop
                loop = asyncio.get_running_loop()
                return asyncio.run_coroutine_threadsafe(
                    self._execute_async(), loop
                ).result(timeout=self.timeout_global)
            except RuntimeError:
                # No running loop
                return asyncio.run(self._execute_async())
        except Exception as e:
            self._log(f"Execution error: {e}", "ERROR")
            return {
                "directive": self.directive,
                "error": str(e),
                "status": "failed",
                "total_time": time.time() - self.start_time
            }


# Quick test function
def test_agent():
    """Test the optimized agent"""
    test_cases = [
        "read main.py",
        "what is in config.json",
        "check system status"
    ]
    
    for test in test_cases:
        print(f"\n\nTesting: {test}")
        agent = OptimizedStargazerAgent(directive=test, api_key="test_key")
        result = agent.execute()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_agent()
