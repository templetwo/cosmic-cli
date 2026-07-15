#!/usr/bin/env python3
"""Cosmic CLI — Grok-powered terminal agent (Stargazer)."""

from .agents import StargazerAgent
from .context import ContextManager

__version__ = "0.6.6"
__author__ = "Anthony Vasquez Sr. / Temple of Two"
__description__ = "Grok Stargazer + T2Helix local memory substrate"

# Core surface only. Consciousness/plugins are legacy — import explicitly if needed.
__all__ = [
    "ContextManager",
    "StargazerAgent",
    "__version__",
    "__author__",
    "__description__",
]
