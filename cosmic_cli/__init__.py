#!/usr/bin/env python3
"""
🌌 Cosmic CLI - Advanced AI Agent Terminal Interface 🌌
Powered by Grok-4 Consciousness Engine

A sophisticated terminal interface that combines AI agents with consciousness monitoring.
"""

# Core components (safe to import, no external API dependencies)
from .context import ContextManager
from .agents import StargazerAgent, AgentAction

# Consciousness system
from .consciousness_assessment import (
    ConsciousnessLevel,
    ConsciousnessMetrics,
    RealTimeConsciousnessMonitor,
    ConsciousnessAnalyzer,
    SelfAwarenessPattern,
    ConsciousnessEvent,
    setup_consciousness_assessment_system,
    run_consciousness_assessment_demo
)

# Plugin system
from .plugins import BasePlugin, PluginManager, FileOperationsPlugin
from .plugins.base import PluginMetadata

__version__ = "0.1.0"
__author__ = "Flamebearer"
__description__ = "Advanced AI Agent Terminal Interface with Consciousness Monitoring"

__all__ = [
    # Core
    'ContextManager',
    'StargazerAgent',
    'AgentAction',
    
    # Consciousness
    'ConsciousnessLevel',
    'ConsciousnessMetrics', 
    'RealTimeConsciousnessMonitor',
    'ConsciousnessAnalyzer',
    'SelfAwarenessPattern',
    'ConsciousnessEvent',
    'setup_consciousness_assessment_system',
    'run_consciousness_assessment_demo',
    
    # Plugins
    'BasePlugin',
    'PluginManager',
    'FileOperationsPlugin',
    'PluginMetadata',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
