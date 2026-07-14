#!/usr/bin/env python3
"""Cosmic CLI — Grok-powered terminal agent (Stargazer)."""

from .agents import StargazerAgent
from .context import ContextManager

__version__ = "0.4.2"
__author__ = "Anthony Vasquez Sr. / Temple of Two"
__description__ = "Grok-powered terminal agent with context-aware Stargazer loop"

# Consciousness modules are optional / legacy. Import on demand so core stays light.
try:
    from .consciousness_assessment import (  # noqa: F401
        ConsciousnessEvent,
        ConsciousnessLevel,
        ConsciousnessMetrics,
        ConsciousnessAnalyzer,
        RealTimeConsciousnessMonitor,
        SelfAwarenessPattern,
        run_consciousness_assessment_demo,
        setup_consciousness_assessment_system,
    )
    _HAS_CONSCIOUSNESS = True
except Exception:  # pragma: no cover
    _HAS_CONSCIOUSNESS = False

try:
    from .plugins import BasePlugin, FileOperationsPlugin, PluginManager  # noqa: F401
    from .plugins.base import PluginMetadata  # noqa: F401
    _HAS_PLUGINS = True
except Exception:  # pragma: no cover
    _HAS_PLUGINS = False

__all__ = [
    "ContextManager",
    "StargazerAgent",
    "__version__",
    "__author__",
    "__description__",
]
