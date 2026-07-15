#!/usr/bin/env python3
"""Cosmic CLI — Temple runtime avionics (Grok Stargazer + T2Helix)."""

from .agents import StargazerAgent
from .context import ContextManager

__version__ = "0.7.0"
__author__ = "Anthony Vasquez Sr. / Temple of Two"
__description__ = (
    "Temple runtime avionics: Grok Stargazer agent, T2Helix memory, compass-gated tools"
)

__all__ = [
    "ContextManager",
    "StargazerAgent",
    "__version__",
    "__author__",
    "__description__",
]
