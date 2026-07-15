#!/usr/bin/env python3
"""
🔌 Cosmic CLI Plugin System 🔌
Extensible plugin architecture for enhancing agent capabilities
"""

from .base import BasePlugin, PluginManager
from .file_operations import FileOperationsPlugin

__all__ = [
    'BasePlugin',
    'PluginManager',
    'FileOperationsPlugin'
]
