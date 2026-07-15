#!/usr/bin/env python3
"""
🔌 Base Plugin System for Cosmic CLI
"""

import os
import json
import logging
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class PluginStatus(Enum):
    """Plugin status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"

@dataclass
class PluginMetadata:
    """Plugin metadata structure"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    categories: List[str]
    priority: int = 50  # Higher numbers = higher priority
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BasePlugin(ABC):
    """Base class for all Cosmic CLI plugins"""
    
    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.status = PluginStatus.INACTIVE
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize plugin metadata - should be overridden by subclasses"""
        self.metadata = PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="Base plugin",
            author="Cosmic CLI",
            dependencies=[],
            categories=["utility"]
        )
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def can_handle(self, action_type: str, parameters: str, context: Dict[str, Any] = None) -> bool:
        """Check if this plugin can handle the given action type and parameters."""
        pass
    
    @abstractmethod
    def execute(self, action_type: str, parameters: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the action. Return result dictionary with 'success' and 'output' keys."""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return list of action types this plugin can handle."""
        return []
    
    def get_help(self) -> str:
        """Return help text for this plugin."""
        return f"Plugin: {self.metadata.name}\nDescription: {self.metadata.description}"
    
    def cleanup(self):
        """Cleanup resources when plugin is unloaded."""
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for this plugin."""
        return {}
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the plugin with provided settings."""
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        return {
            "status": self.status.value,
            "name": self.metadata.name if self.metadata else "Unknown",
            "healthy": self.status == PluginStatus.ACTIVE
        }

class PluginManager:
    """Manages plugin loading, unloading, and execution"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_dirs = plugin_dirs or [
            str(Path(__file__).parent),
            str(Path.home() / ".cosmic_cli" / "plugins")
        ]
        self.logger = logging.getLogger("PluginManager")
        self._config_file = Path.home() / ".cosmic_cli" / "plugins_config.json"
        self._load_config()
    
    def _load_config(self):
        """Load plugin configuration"""
        self.config = {}
        if self._config_file.exists():
            try:
                with open(self._config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load plugin config: {e}")
    
    def _save_config(self):
        """Save plugin configuration"""
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save plugin config: {e}")
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directories"""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            # Look for Python files (excluding __init__.py and base.py)
            for py_file in plugin_path.glob("*.py"):
                if py_file.stem not in ["__init__", "base"]:
                    discovered.append(py_file.stem)
        
        return discovered
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin by name"""
        try:
            # Try to import from plugins package first
            try:
                module = importlib.import_module(f"cosmic_cli.plugins.{plugin_name}")
            except ImportError:
                # Try direct import from plugin directories
                for plugin_dir in self.plugin_dirs:
                    plugin_path = Path(plugin_dir) / f"{plugin_name}.py"
                    if plugin_path.exists():
                        spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        break
                else:
                    raise ImportError(f"Plugin {plugin_name} not found")
            
            # Find plugin classes in the module
            plugin_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BasePlugin) and 
                    obj != BasePlugin and 
                    not name.startswith('_')):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                self.logger.error(f"No plugin classes found in {plugin_name}")
                return False
            
            # Instantiate the first plugin class found
            plugin_class = plugin_classes[0]
            plugin_instance = plugin_class()
            
            # Initialize the plugin
            if plugin_instance.initialize():
                plugin_instance.status = PluginStatus.ACTIVE
                self.plugins[plugin_name] = plugin_instance
                
                # Apply configuration if available
                if plugin_name in self.config:
                    plugin_instance.configure(self.config[plugin_name])
                
                self.logger.info(f"Successfully loaded plugin: {plugin_name}")
                return True
            else:
                plugin_instance.status = PluginStatus.ERROR
                self.logger.error(f"Failed to initialize plugin: {plugin_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin"""
        if plugin_name in self.plugins:
            try:
                self.plugins[plugin_name].cleanup()
                self.plugins[plugin_name].status = PluginStatus.INACTIVE
                del self.plugins[plugin_name]
                self.logger.info(f"Unloaded plugin: {plugin_name}")
                return True
            except Exception as e:
                self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
                return False
        return False
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins"""
        results = {}
        discovered = self.discover_plugins()
        
        for plugin_name in discovered:
            results[plugin_name] = self.load_plugin(plugin_name)
        
        return results
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a specific plugin instance"""
        return self.plugins.get(plugin_name)
    
    def get_all_plugins(self) -> Dict[str, BasePlugin]:
        """Get all loaded plugins"""
        return self.plugins.copy()
    
    def find_capable_plugins(self, action_type: str, parameters: str, 
                           context: Dict[str, Any] = None) -> List[BasePlugin]:
        """Find all plugins capable of handling the given action"""
        capable_plugins = []
        
        for plugin in self.plugins.values():
            if (plugin.status == PluginStatus.ACTIVE and 
                plugin.can_handle(action_type, parameters, context)):
                capable_plugins.append(plugin)
        
        # Sort by priority (higher priority first)
        capable_plugins.sort(
            key=lambda p: p.metadata.priority if p.metadata else 0, 
            reverse=True
        )
        
        return capable_plugins
    
    def execute_with_plugins(self, action_type: str, parameters: str, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute action with the best available plugin"""
        capable_plugins = self.find_capable_plugins(action_type, parameters, context)
        
        if not capable_plugins:
            return {
                "success": False,
                "output": f"No plugins available to handle action: {action_type}",
                "plugin_used": None
            }
        
        # Use the highest priority plugin
        best_plugin = capable_plugins[0]
        
        try:
            result = best_plugin.execute(action_type, parameters, context)
            result["plugin_used"] = best_plugin.metadata.name if best_plugin.metadata else "Unknown"
            return result
        except Exception as e:
            self.logger.error(f"Plugin {best_plugin.metadata.name} failed: {e}")
            return {
                "success": False,
                "output": f"Plugin execution failed: {e}",
                "plugin_used": best_plugin.metadata.name if best_plugin.metadata else "Unknown"
            }
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins"""
        status = {}
        
        for name, plugin in self.plugins.items():
            health = plugin.health_check()
            status[name] = {
                "metadata": plugin.metadata.to_dict() if plugin.metadata else {},
                "status": plugin.status.value,
                "health": health,
                "capabilities": plugin.get_capabilities()
            }
        
        return status
    
    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a specific plugin"""
        if plugin_name in self.plugins:
            success = self.plugins[plugin_name].configure(config)
            if success:
                # Save configuration
                self.config[plugin_name] = config
                self._save_config()
            return success
        return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin"""
        if plugin_name in self.plugins:
            config = self.config.get(plugin_name, {})
            self.unload_plugin(plugin_name)
            success = self.load_plugin(plugin_name)
            if success and config:
                self.configure_plugin(plugin_name, config)
            return success
        else:
            return self.load_plugin(plugin_name)
    
    def get_help(self, plugin_name: str = None) -> str:
        """Get help for specific plugin or all plugins"""
        if plugin_name:
            if plugin_name in self.plugins:
                return self.plugins[plugin_name].get_help()
            else:
                return f"Plugin '{plugin_name}' not found."
        else:
            help_text = "🔌 Available Plugins:\n\n"
            for name, plugin in self.plugins.items():
                help_text += f"• {name}: {plugin.metadata.description if plugin.metadata else 'No description'}\n"
                help_text += f"  Status: {plugin.status.value}\n"
                help_text += f"  Capabilities: {', '.join(plugin.get_capabilities())}\n\n"
            return help_text

# Global plugin manager instance
_plugin_manager = None

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

def initialize_plugins():
    """Initialize the global plugin system"""
    manager = get_plugin_manager()
    results = manager.load_all_plugins()
    
    logger.info(f"Plugin initialization results: {results}")
    return results
