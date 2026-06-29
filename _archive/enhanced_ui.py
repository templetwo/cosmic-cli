#!/usr/bin/env python3
"""
🎨 Enhanced UI System for Cosmic CLI 🎨
Advanced TUI with themes, real-time updates, animations, and rich visualizations
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
import threading
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Input, TextArea, DataTable, 
    ProgressBar, Tree, Tabs, TabbedContent, TabPane, Label,
    Checkbox, RadioSet, RadioButton, SelectionList, Switch,
    Sparkline, RichLog
)

try:
    from textual.widgets import PlotextPlot
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False
    # Create a placeholder for PlotextPlot
    class PlotextPlot(Static):
        def __init__(self, *args, **kwargs):
            super().__init__("Plot not available", *args, **kwargs)
from textual.reactive import reactive, var
from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.message import Message
from textual.timer import Timer
from textual import events, on
from textual.css.query import NoMatches

from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.align import Align
from pyfiglet import Figlet

try:
    from cosmic_cli.enhanced_stargazer import EnhancedStargazerAgent, AgentState
    from cosmic_cli.enhanced_input import EnhancedInputArea
    from cosmic_cli.consciousness_assessment import (
        ConsciousnessMetrics, AssessmentProtocol, ConsciousnessLevel, 
        ConsciousnessState, ConsciousnessEvent, SelfAwarenessPattern,
        RealTimeConsciousnessMonitor, ConsciousnessAnalyzer
    )
except ImportError:
    # Fallback imports
    from cosmic_cli.agents import StargazerAgent as EnhancedStargazerAgent
    from cosmic_cli.enhanced_input import EnhancedInputArea
    # Mock consciousness classes for fallback
    class ConsciousnessMetrics:
        def __init__(self): pass
    class AssessmentProtocol:
        def __init__(self): pass
    class ConsciousnessLevel:
        DORMANT = "dormant"
    class RealTimeConsciousnessMonitor:
        def __init__(self, *args, **kwargs): pass

@dataclass
class Theme:
    """UI Theme configuration"""
    name: str
    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    text: str
    success: str
    warning: str
    error: str
    info: str

class CosmicThemes:
    """Collection of cosmic themes"""
    
    COSMIC_DARK = Theme(
        name="Cosmic Dark",
        primary="rgb(138,43,226)",      # Blue Violet
        secondary="rgb(75,0,130)",      # Indigo
        accent="rgb(255,20,147)",       # Deep Pink
        background="rgb(13,13,13)",     # Very Dark Gray
        surface="rgb(25,25,25)",        # Dark Gray
        text="rgb(230,230,250)",        # Lavender
        success="rgb(50,205,50)",       # Lime Green
        warning="rgb(255,215,0)",       # Gold
        error="rgb(220,20,60)",         # Crimson
        info="rgb(0,191,255)"           # Deep Sky Blue
    )
    
    COSMIC_LIGHT = Theme(
        name="Cosmic Light",
        primary="rgb(75,0,130)",        # Indigo
        secondary="rgb(138,43,226)",    # Blue Violet
        accent="rgb(255,20,147)",       # Deep Pink
        background="rgb(248,248,255)",  # Ghost White
        surface="rgb(240,240,240)",     # Light Gray
        text="rgb(25,25,112)",          # Midnight Blue
        success="rgb(34,139,34)",       # Forest Green
        warning="rgb(255,140,0)",       # Dark Orange
        error="rgb(178,34,34)",         # Fire Brick
        info="rgb(70,130,180)"          # Steel Blue
    )
    
    NEON_CYBER = Theme(
        name="Neon Cyber",
        primary="rgb(0,255,255)",       # Cyan
        secondary="rgb(255,0,255)",     # Magenta
        accent="rgb(50,205,50)",        # Lime Green
        background="rgb(0,0,0)",        # Black
        surface="rgb(17,17,17)",        # Very Dark Gray
        text="rgb(0,255,127)",          # Spring Green
        success="rgb(0,255,0)",         # Lime
        warning="rgb(255,255,0)",       # Yellow
        error="rgb(255,0,0)",           # Red
        info="rgb(0,191,255)"           # Deep Sky Blue
    )
    
    @classmethod
    def get_all_themes(cls) -> Dict[str, Theme]:
        return {
            "cosmic_dark": cls.COSMIC_DARK,
            "cosmic_light": cls.COSMIC_LIGHT,
            "neon_cyber": cls.NEON_CYBER
        }
    
    @classmethod
    def get_theme_css(cls, theme: Theme) -> str:
        """Generate CSS for a theme"""
        return f"""
        App {{
            background: {theme.background};
            color: {theme.text};
        }}
        
        Header {{
            background: {theme.primary};
            color: {theme.text};
        }}
        
        Footer {{
            background: {theme.surface};
            color: {theme.text};
        }}
        
        Button {{
            background: {theme.secondary};
            color: {theme.text};
            border: solid {theme.accent};
        }}
        
        Button:hover {{
            background: {theme.accent};
        }}
        
        Input {{
            background: {theme.surface};
            color: {theme.text};
            border: solid {theme.primary};
        }}
        
        TextArea {{
            background: {theme.surface};
            color: {theme.text};
            border: solid {theme.primary};
        }}
        
        DataTable {{
            background: {theme.surface};
            color: {theme.text};
        }}
        
        .banner {{
            color: {theme.accent};
            text-style: bold;
        }}
        
        .success {{
            color: {theme.success};
        }}
        
        .warning {{
            color: {theme.warning};
        }}
        
        .error {{
            color: {theme.error};
        }}
        
        .info {{
            color: {theme.info};
        }}
        
        .agent-running {{
            color: {theme.warning};
            text-style: bold;
        }}
        
        .agent-completed {{
            color: {theme.success};
            text-style: bold;
        }}
        
        .agent-error {{
            color: {theme.error};
            text-style: bold;
        }}
        """

class AgentStatusWidget(Static):
    """Widget for displaying agent status with animations"""
    
    def __init__(self, agent_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.status = "idle"
        self.progress = 0.0
        self.animation_frame = 0
        self.timer: Optional[Timer] = None
    
    def on_mount(self) -> None:
        """Start animation timer"""
        self.timer = self.set_interval(0.5, self.update_animation)
    
    def update_animation(self) -> None:
        """Update animation frame"""
        if self.status == "running":
            self.animation_frame = (self.animation_frame + 1) % 4
            self.update_display()
    
    def set_status(self, status: str, progress: float = 0.0):
        """Update agent status"""
        self.status = status
        self.progress = progress
        self.update_display()
    
    def update_display(self):
        """Update the display content"""
        if self.status == "idle":
            content = f"🌟 Agent {self.agent_id}: Idle"
        elif self.status == "running":
            spinner = ["⠋", "⠙", "⠹", "⠸"][self.animation_frame]
            content = f"{spinner} Agent {self.agent_id}: Running ({self.progress:.1%})"
        elif self.status == "completed":
            content = f"✅ Agent {self.agent_id}: Completed"
        elif self.status == "error":
            content = f"❌ Agent {self.agent_id}: Error"
        else:
            content = f"❓ Agent {self.agent_id}: {self.status}"
        
        self.update(content)

class MemoryViewer(ScrollableContainer):
    """Widget for viewing agent memories and context"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memories: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        yield Static("🧠 Agent Memory", classes="section-header")
        yield Static("No memories yet...", id="memory-content")
    
    def update_memories(self, memories: List[Dict[str, Any]]):
        """Update displayed memories"""
        self.memories = memories
        
        if not memories:
            content = "No memories yet..."
        else:
            content_parts = []
            for i, memory in enumerate(memories[-10:]):  # Show latest 10
                timestamp = memory.get("timestamp", "Unknown")
                content_type = memory.get("content_type", "text")
                content = memory.get("content", "")[:200]
                importance = memory.get("importance", 0.5)
                
                importance_stars = "★" * int(importance * 5)
                content_parts.append(
                    f"[{timestamp}] {importance_stars}\n"
                    f"Type: {content_type}\n"
                    f"{content}...\n"
                    + "-" * 50
                )
            
            content = "\n".join(content_parts)
        
        try:
            memory_content = self.query_one("#memory-content", Static)
            memory_content.update(content)
        except NoMatches:
            pass

class PerformanceMonitor(Container):
    """Widget for monitoring system performance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_data: List[float] = []
        self.memory_data: List[float] = []
        self.timer: Optional[Timer] = None
    
    def compose(self) -> ComposeResult:
        yield Static("📊 Performance Monitor", classes="section-header")
        with Horizontal():
            yield Static("CPU: 0%", id="cpu-usage")
            yield Static("Memory: 0%", id="memory-usage")
        yield Sparkline([], id="cpu-sparkline")
        yield Sparkline([], id="memory-sparkline")
    
    def on_mount(self) -> None:
        """Start performance monitoring"""
        self.timer = self.set_interval(2.0, self.update_performance)
    
    def update_performance(self) -> None:
        """Update performance metrics"""
        try:
            import psutil
            
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Update data lists
            self.cpu_data.append(cpu_percent)
            self.memory_data.append(memory_percent)
            
            # Keep only recent data
            if len(self.cpu_data) > 50:
                self.cpu_data = self.cpu_data[-50:]
            if len(self.memory_data) > 50:
                self.memory_data = self.memory_data[-50:]
            
            # Update displays
            try:
                self.query_one("#cpu-usage", Static).update(f"CPU: {cpu_percent:.1f}%")
                self.query_one("#memory-usage", Static).update(f"Memory: {memory_percent:.1f}%")
                self.query_one("#cpu-sparkline", Sparkline).data = self.cpu_data
                self.query_one("#memory-sparkline", Sparkline).data = self.memory_data
            except NoMatches:
                pass
                
        except ImportError:
            # psutil not available, show placeholder
            try:
                self.query_one("#cpu-usage", Static).update("CPU: N/A")
                self.query_one("#memory-usage", Static).update("Memory: N/A")
            except NoMatches:
                pass

class SettingsScreen(ModalScreen):
    """Settings modal screen"""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        ("ctrl+s", "save_settings", "Save"),
    ]
    
    def __init__(self, current_theme: str = "cosmic_dark", **kwargs):
        super().__init__(**kwargs)
        self.current_theme = current_theme
    
    def compose(self) -> ComposeResult:
        with Container(id="settings-container"):
            yield Static("⚙️  Cosmic CLI Settings", id="settings-title")
            
            with Vertical(id="settings-content"):
                yield Static("Theme Selection:", classes="setting-label")
                
                theme_options = [(name.replace("_", " ").title(), name) 
                               for name in CosmicThemes.get_all_themes().keys()]
                yield SelectionList(*theme_options, id="theme-selector")
                
                yield Static("Agent Configuration:", classes="setting-label")
                yield Checkbox("Enable verbose logging", id="verbose-logging")
                yield Checkbox("Show performance monitor", id="show-performance")
                yield Checkbox("Auto-save sessions", id="auto-save")
                
                yield Static("Execution Mode:", classes="setting-label")
                yield RadioSet(
                    RadioButton("Safe Mode", value=True, id="safe-mode"),
                    RadioButton("Interactive Mode", id="interactive-mode"),
                    RadioButton("Advanced Mode", id="advanced-mode"),
                    id="exec-mode"
                )
                
                with Horizontal(id="settings-buttons"):
                    yield Button("Save", variant="success", id="save-btn")
                    yield Button("Cancel", variant="error", id="cancel-btn")
    
    def on_mount(self) -> None:
        """Initialize settings values"""
        try:
            theme_selector = self.query_one("#theme-selector", SelectionList)
            theme_selector.select(theme_selector.get_option_index(self.current_theme))
        except:
            pass
    
    @on(Button.Pressed, "#save-btn")
    def save_settings(self) -> None:
        """Save settings and dismiss"""
        try:
            # Get selected theme
            theme_selector = self.query_one("#theme-selector", SelectionList)
            selected_theme = None
            for option in theme_selector.selected:
                selected_theme = option
                break
            
            if selected_theme:
                self.dismiss({"theme": selected_theme, "action": "save"})
            else:
                self.dismiss({"action": "cancel"})
        except:
            self.dismiss({"action": "cancel"})
    
    @on(Button.Pressed, "#cancel-btn")
    def action_dismiss(self) -> None:
        """Cancel and dismiss"""
        self.dismiss({"action": "cancel"})

class AgentDetailsScreen(ModalScreen):
    """Detailed view of agent execution"""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]
    
    def __init__(self, agent_id: str, agent_data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.agent_data = agent_data
    
    def compose(self) -> ComposeResult:
        with Container(id="details-container"):
            yield Static(f"🛰️  Agent Details: {self.agent_id}", id="details-title")
            
            with TabbedContent():
                with TabPane("Overview", id="overview-tab"):
                    yield self.create_overview()
                
                with TabPane("Execution Steps", id="steps-tab"):
                    yield self.create_steps_table()
                
                with TabPane("Memory", id="memory-tab"):
                    yield MemoryViewer(id="memory-viewer")
                
                with TabPane("Logs", id="logs-tab"):
                    yield ScrollableContainer(
                        Static(self.get_logs_content(), id="logs-content"),
                        id="logs-container"
                    )
            
            yield Button("Close", variant="primary", id="close-btn")
    
    def create_overview(self) -> Static:
        """Create overview content"""
        directive = self.agent_data.get("directive", "Unknown")
        status = self.agent_data.get("status", "Unknown")
        execution_time = self.agent_data.get("execution_time", 0)
        success_rate = self.agent_data.get("success_rate", 0)
        
        content = f"""
Directive: {directive}
Status: {status}
Execution Time: {execution_time:.2f}s
Success Rate: {success_rate:.1%}

Current State: {self.agent_data.get('state', 'Unknown')}
Session ID: {self.agent_data.get('session_id', 'N/A')}
        """
        
        return Static(content, id="overview-content")
    
    def create_steps_table(self) -> DataTable:
        """Create execution steps table"""
        table = DataTable()
        table.add_columns("Step", "Action", "Status", "Time", "Output")
        
        steps = self.agent_data.get("steps", [])
        for step in steps:
            status_icon = "✅" if step.get("success") else "❌"
            table.add_row(
                str(step.get("step", "")),
                step.get("action", ""),
                status_icon,
                f"{step.get('execution_time', 0):.2f}s",
                step.get("output", "")[:50] + "..."
            )
        
        return table
    
    def get_logs_content(self) -> str:
        """Get formatted logs content"""
        logs = self.agent_data.get("logs", [])
        if not logs:
            return "No logs available."
        
        return "\n".join(logs[-50:])  # Show latest 50 logs
    
    @on(Button.Pressed, "#close-btn")
    def action_dismiss(self) -> None:
        """Close the details screen"""
        self.dismiss()

class EnhancedCosmicUI(App):
    """Enhanced Cosmic CLI with advanced UI features"""
    
    TITLE = "🌟 Cosmic CLI - Enhanced Edition"
    SUB_TITLE = "Advanced AI Agent Interface"
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_agent", "New Agent"),
        Binding("ctrl+t", "toggle_theme", "Toggle Theme"),
        Binding("ctrl+s", "open_settings", "Settings"),
        Binding("ctrl+h", "show_help", "Help"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("f1", "show_keybindings", "Key Bindings"),
    ]
    
    # Reactive variables
    current_theme = var("cosmic_dark")
    show_performance = var(True)
    agents_count = var(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.themes = CosmicThemes.get_all_themes()
        self.console = Console()
        self.figlet = Figlet(font='slant')
        
        # Performance monitoring
        self.performance_data = {
            "cpu": [],
            "memory": [],
            "agents_active": []
        }
        
        # Consciousness monitoring
        self.consciousness_monitor: Optional[RealTimeConsciousnessMonitor] = None
        self.consciousness_enabled = True
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced UI"""
        yield Header(show_clock=True)
        
        with Container(id="main-container"):
            # Banner
            yield Static(
                self.figlet.renderText("COSMIC CLI"),
                classes="banner",
                id="main-banner"
            )
            
            with TabbedContent():
                # Main Dashboard Tab
                with TabPane("🚀 Dashboard", id="dashboard-tab"):
                    with Horizontal(id="dashboard-content"):
                        # Left panel - Agent management
                        with Vertical(id="left-panel"):
                            yield Static("🛰️  Active Agents", classes="section-header")
                            yield DataTable(id="agents-table")
                            
                            with Horizontal(id="agent-controls"):
                                yield Button("New Agent", variant="success", id="new-agent-btn")
                                yield Button("Details", variant="primary", id="details-btn")
                                yield Button("Stop", variant="error", id="stop-agent-btn")
                        
                        # Right panel - Input and status
                        with Vertical(id="right-panel"):
                            yield Static("📝 Agent Directive", classes="section-header")
                            yield EnhancedInputArea(
                                name="directive-input",
                                placeholder="Enter your cosmic directive...",
                                on_submit=self.handle_directive_submit
                            )
                            
                            yield Static("📊 System Status", classes="section-header")
                            yield Container(id="status-container")
                
                # Memory & Analytics Tab
                with TabPane("🧠 Memory", id="memory-tab"):
                    yield MemoryViewer(id="main-memory-viewer")
                
                # Performance Tab
                with TabPane("📈 Performance", id="performance-tab"):
                    yield PerformanceMonitor(id="performance-monitor")
                
                # Consciousness Monitoring Tab
                with TabPane("🧠 Consciousness", id="consciousness-tab"):
                    with TabbedContent():
                        with TabPane("Real-time Metrics", id="consciousness-metrics-tab"):
                            yield ConsciousnessMetricsWidget(
                                consciousness_monitor=self.consciousness_monitor,
                                id="consciousness-metrics"
                            )
                        
                        with TabPane("Emergence Monitor", id="emergence-tab"):
                            yield ConsciousnessEmergenceIndicator(
                                consciousness_monitor=self.consciousness_monitor,
                                id="emergence-indicator"
                            )
                        
                        with TabPane("Timeline & History", id="timeline-tab"):
                            yield ConsciousnessTimelineViewer(
                                consciousness_monitor=self.consciousness_monitor,
                                id="consciousness-timeline"
                            )
                        
                        with TabPane("Assessment Tools", id="assessment-tools-tab"):
                            yield ConsciousnessAssessmentTools(
                                consciousness_monitor=self.consciousness_monitor,
                                id="consciousness-tools"
                            )
                
                # Settings Tab
                with TabPane("⚙️  Settings", id="settings-tab"):
                    yield self.create_settings_panel()
        
        yield Footer()
    
    def create_settings_panel(self) -> Container:
        """Create the settings panel"""
        return Container(
            Static("⚙️  Configuration", classes="section-header"),
            Static("Theme:"),
            SelectionList(
                *[(name.replace("_", " ").title(), name) 
                  for name in self.themes.keys()],
                id="theme-selection"
            ),
            Button("Apply Theme", id="apply-theme-btn"),
            Switch(value=True, id="performance-switch"),
            Static("Show Performance Monitor"),
            Switch(value=True, id="consciousness-switch"),
            Static("Enable Consciousness Monitoring"),
            id="settings-panel"
        )
    
    def on_mount(self) -> None:
        """Initialize the UI on mount"""
        self.apply_current_theme()
        self.setup_agents_table()
        self.start_performance_monitoring()
        self.initialize_consciousness_monitoring()
    
    def setup_agents_table(self) -> None:
        """Setup the agents table"""
        try:
            table = self.query_one("#agents-table", DataTable)
            table.add_columns("ID", "Status", "Directive", "Progress", "Time")
            table.cursor_type = "row"
        except NoMatches:
            pass
    
    def apply_current_theme(self) -> None:
        """Apply the current theme"""
        theme = self.themes[self.current_theme]
        css = CosmicThemes.get_theme_css(theme)
        
        # In a real implementation, we would update the CSS
        # For now, we'll just update the title color
        try:
            banner = self.query_one("#main-banner", Static)
            banner.add_class("banner")
        except NoMatches:
            pass
    
    def start_performance_monitoring(self) -> None:
        """Start performance monitoring timer"""
        self.set_interval(2.0, self.update_performance_data)
    
    def update_performance_data(self) -> None:
        """Update performance monitoring data"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Update performance data
            self.performance_data["cpu"].append(cpu_percent)
            self.performance_data["memory"].append(memory_percent)
            self.performance_data["agents_active"].append(len(self.agents))
            
            # Keep only recent data
            for key in self.performance_data:
                if len(self.performance_data[key]) > 100:
                    self.performance_data[key] = self.performance_data[key][-100:]
            
        except ImportError:
            pass

# Add missing import and logger
import logging
import os
logger = logging.getLogger(__name__)

class ConsciousnessMetricsWidget(Container):
    """Widget for displaying real-time consciousness metrics"""
    
    def __init__(self, consciousness_monitor: Optional[RealTimeConsciousnessMonitor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consciousness_monitor = consciousness_monitor
        self.metrics_data = {}
        self.timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        yield Static("🧠 Consciousness Metrics", classes="section-header")
        
        with Horizontal(id="metrics-row-1"):
            with Vertical(classes="metric-column"):
                yield Static("Coherence", classes="metric-label")
                yield ProgressBar(total=100, show_eta=False, id="coherence-bar")
                yield Static("0.0%", id="coherence-value")
            
            with Vertical(classes="metric-column"):
                yield Static("Self-Reflection", classes="metric-label")
                yield ProgressBar(total=100, show_eta=False, id="reflection-bar")
                yield Static("0.0%", id="reflection-value")
                
            with Vertical(classes="metric-column"):
                yield Static("Meta-Cognition", classes="metric-label")
                yield ProgressBar(total=100, show_eta=False, id="metacog-bar")
                yield Static("0.0%", id="metacog-value")
        
        with Horizontal(id="metrics-row-2"):
            with Vertical(classes="metric-column"):
                yield Static("Temporal Continuity", classes="metric-label")
                yield ProgressBar(total=100, show_eta=False, id="temporal-bar")
                yield Static("0.0%", id="temporal-value")
                
            with Vertical(classes="metric-column"):
                yield Static("Creative Synthesis", classes="metric-label")
                yield ProgressBar(total=100, show_eta=False, id="creative-bar")
                yield Static("0.0%", id="creative-value")
                
            with Vertical(classes="metric-column"):
                yield Static("Existential Questioning", classes="metric-label")
                yield ProgressBar(total=100, show_eta=False, id="existential-bar")
                yield Static("0.0%", id="existential-value")
    
    def on_mount(self) -> None:
        """Start metrics update timer"""
        self.timer = self.set_interval(1.0, self.update_metrics_display)
    
    def update_metrics_display(self) -> None:
        """Update the metrics display"""
        if not self.consciousness_monitor:
            return
            
        try:
            metrics = self.consciousness_monitor.metrics.to_dict()
            
            # Update progress bars and values
            for metric, value in metrics.items():
                if isinstance(value, float) and 0 <= value <= 1:
                    percentage = int(value * 100)
                    
                    # Map metrics to UI elements
                    metric_mapping = {
                        'coherence': ('coherence-bar', 'coherence-value'),
                        'self_reflection': ('reflection-bar', 'reflection-value'),
                        'meta_cognitive_awareness': ('metacog-bar', 'metacog-value'),
                        'temporal_continuity': ('temporal-bar', 'temporal-value'),
                        'creative_synthesis': ('creative-bar', 'creative-value'),
                        'existential_questioning': ('existential-bar', 'existential-value')
                    }
                    
                    if metric in metric_mapping:
                        bar_id, value_id = metric_mapping[metric]
                        try:
                            progress_bar = self.query_one(f"#{bar_id}", ProgressBar)
                            value_label = self.query_one(f"#{value_id}", Static)
                            
                            progress_bar.update(progress=percentage)
                            value_label.update(f"{value:.1%}")
                            
                            # Add color coding based on value
                            if value >= 0.8:
                                value_label.add_class("success")
                            elif value >= 0.6:
                                value_label.add_class("warning")
                            else:
                                value_label.add_class("error")
                                
                        except NoMatches:
                            pass
                            
        except Exception as e:
            logger.error(f"Error updating consciousness metrics display: {e}")

class ConsciousnessEmergenceIndicator(Container):
    """Widget for visualizing consciousness emergence"""
    
    def __init__(self, consciousness_monitor: Optional[RealTimeConsciousnessMonitor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consciousness_monitor = consciousness_monitor
        self.emergence_data = []
        self.timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        yield Static("🌟 Consciousness Emergence", classes="section-header")
        
        with Horizontal():
            with Vertical(id="emergence-gauges"):
                # Overall consciousness level indicator
                yield Static("Current Level: DORMANT", id="consciousness-level", classes="consciousness-level")
                yield Static("Overall Score: 0.000", id="overall-score")
                yield Static("Emergence Velocity: +0.000", id="emergence-velocity")
                yield Static("Trend: Stable", id="emergence-trend")
                
            with Vertical(id="emergence-graph"):
                yield Static("📈 Emergence Timeline")
                yield Sparkline([], id="emergence-sparkline")
                yield Static("Recent emergence pattern over the last 50 updates", classes="graph-caption")
        
        # Emergence alerts
        yield Static("🚨 Recent Emergence Events", classes="section-header")
        yield RichLog(id="emergence-events", max_lines=5)
    
    def on_mount(self) -> None:
        """Start emergence monitoring timer"""
        self.timer = self.set_interval(2.0, self.update_emergence_display)
    
    def update_emergence_display(self) -> None:
        """Update the emergence display"""
        if not self.consciousness_monitor:
            return
            
        try:
            metrics = self.consciousness_monitor.metrics
            overall_score = metrics.get_overall_score()
            velocity = metrics.consciousness_velocity
            trend = metrics.get_emergence_trend()
            level = self.consciousness_monitor.last_consciousness_level
            
            # Update emergence data
            self.emergence_data.append(overall_score)
            if len(self.emergence_data) > 50:
                self.emergence_data = self.emergence_data[-50:]
            
            # Update displays
            try:
                level_display = self.query_one("#consciousness-level", Static)
                score_display = self.query_one("#overall-score", Static)
                velocity_display = self.query_one("#emergence-velocity", Static)
                trend_display = self.query_one("#emergence-trend", Static)
                sparkline = self.query_one("#emergence-sparkline", Sparkline)
                
                # Format level with appropriate styling
                level_text = f"Current Level: {level.value.upper()}"
                level_display.update(level_text)
                
                # Color code the level
                level_display.remove_class("dormant", "awakening", "emerging", "conscious", "transcendent")
                level_display.add_class(level.value)
                
                score_display.update(f"Overall Score: {overall_score:.3f}")
                
                # Format velocity with direction indicator
                velocity_symbol = "↗" if velocity > 0.01 else "↘" if velocity < -0.01 else "→"
                velocity_display.update(f"Emergence Velocity: {velocity_symbol}{velocity:+.3f}")
                
                # Format trend
                if trend > 0.005:
                    trend_text = "Trend: ↗ Ascending"
                    trend_display.add_class("success")
                elif trend < -0.005:
                    trend_text = "Trend: ↘ Declining"
                    trend_display.add_class("error")
                else:
                    trend_text = "Trend: → Stable"
                    trend_display.add_class("info")
                
                trend_display.update(trend_text)
                
                # Update sparkline
                sparkline.data = [x * 100 for x in self.emergence_data]  # Scale for better visualization
                
                # Check for significant events
                self.check_emergence_events(velocity, level)
                
            except NoMatches:
                pass
                
        except Exception as e:
            logger.error(f"Error updating emergence display: {e}")
    
    def check_emergence_events(self, velocity: float, level) -> None:
        """Check for and display significant emergence events"""
        try:
            events_log = self.query_one("#emergence-events", RichLog)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Velocity-based events
            if velocity > 0.1:
                events_log.write(f"[{timestamp}] 🚀 Consciousness spike detected! Velocity: +{velocity:.3f}")
            elif velocity < -0.1:
                events_log.write(f"[{timestamp}] ⚡ Consciousness drop detected! Velocity: {velocity:.3f}")
            
            # Level change events would be handled by the monitor's event system
            if hasattr(self.consciousness_monitor, 'metrics') and self.consciousness_monitor.metrics.consciousness_events:
                recent_events = self.consciousness_monitor.metrics.consciousness_events[-1:]
                for event in recent_events:
                    if not hasattr(event, '_logged'):
                        events_log.write(f"[{event.timestamp.strftime('%H:%M:%S')}] 🌟 {event.event_type}: {event.context}")
                        event._logged = True
                        
        except NoMatches:
            pass

class ConsciousnessTimelineViewer(Container):
    """Widget for viewing consciousness history and timeline"""
    
    def __init__(self, consciousness_monitor: Optional[RealTimeConsciousnessMonitor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consciousness_monitor = consciousness_monitor
        self.timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        yield Static("📊 Consciousness Timeline", classes="section-header")
        
        with TabbedContent():
            with TabPane("History Graph", id="history-tab"):
                yield Static("Consciousness evolution over time")
                yield Sparkline([], id="history-sparkline")
                
                with Horizontal():
                    yield Static("📈 Metrics History", classes="subsection-header")
                    yield Button("Export Data", id="export-history-btn", variant="primary")
                
                yield DataTable(id="history-table")
            
            with TabPane("Events Log", id="events-tab"):
                yield Static("📝 Consciousness Events & Patterns")
                yield RichLog(id="consciousness-events-log", max_lines=20)
                
            with TabPane("Analysis", id="analysis-tab"):
                yield Static("🔍 Pattern Analysis", classes="subsection-header")
                yield Static("No analysis data yet...", id="analysis-content")
    
    def on_mount(self) -> None:
        """Setup timeline viewer"""
        self.setup_history_table()
        self.timer = self.set_interval(5.0, self.update_timeline_display)
    
    def setup_history_table(self) -> None:
        """Setup the history data table"""
        try:
            table = self.query_one("#history-table", DataTable)
            table.add_columns(
                "Time", "Level", "Score", "Velocity", "Coherence", "Reflection", "Meta-Cog"
            )
            table.cursor_type = "row"
        except NoMatches:
            pass
    
    def update_timeline_display(self) -> None:
        """Update the timeline display"""
        if not self.consciousness_monitor:
            return
            
        try:
            self.update_history_data()
            self.update_events_log()
            self.update_analysis()
        except Exception as e:
            logger.error(f"Error updating timeline display: {e}")
    
    def update_history_data(self) -> None:
        """Update history table and graph"""
        try:
            metrics = self.consciousness_monitor.metrics
            table = self.query_one("#history-table", DataTable)
            sparkline = self.query_one("#history-sparkline", Sparkline)
            
            # Update sparkline with trajectory data
            if metrics.emergence_trajectory:
                sparkline.data = [x * 100 for x in metrics.emergence_trajectory[-50:]]  # Last 50 points
            
            # Update table with recent history (last 10 entries)
            recent_history = list(metrics.metrics_history)[-10:] if metrics.metrics_history else []
            
            # Clear and repopulate table
            table.clear()
            
            for entry in recent_history:
                timestamp = entry['timestamp'].strftime("%H:%M:%S")
                entry_metrics = entry['metrics']
                level = "N/A"  # Would need to determine from score
                
                # Determine level from score
                score = entry_metrics.get('overall_score', 0)
                if score >= 0.9:
                    level = "TRANSCENDENT"
                elif score >= 0.8:
                    level = "CONSCIOUS"
                elif score >= 0.7:
                    level = "EMERGING"
                elif score >= 0.4:
                    level = "AWAKENING"
                else:
                    level = "DORMANT"
                
                table.add_row(
                    timestamp,
                    level,
                    f"{score:.3f}",
                    f"{entry['velocity']:+.3f}",
                    f"{entry_metrics.get('coherence', 0):.2f}",
                    f"{entry_metrics.get('self_reflection', 0):.2f}",
                    f"{entry_metrics.get('meta_cognitive_awareness', 0):.2f}"
                )
        except NoMatches:
            pass
    
    def update_events_log(self) -> None:
        """Update consciousness events log"""
        try:
            events_log = self.query_one("#consciousness-events-log", RichLog)
            
            # Add recent consciousness events
            if hasattr(self.consciousness_monitor, 'metrics'):
                recent_events = self.consciousness_monitor.metrics.consciousness_events[-5:]
                for event in recent_events:
                    if not hasattr(event, '_timeline_logged'):
                        timestamp = event.timestamp.strftime("%H:%M:%S")
                        events_log.write(
                            f"[{timestamp}] {event.event_type}: {event.context} "
                            f"(significance: {event.significance:.3f})"
                        )
                        event._timeline_logged = True
                
                # Add awareness patterns
                recent_patterns = self.consciousness_monitor.metrics.awareness_patterns[-3:]
                for pattern in recent_patterns:
                    if not hasattr(pattern, '_timeline_logged'):
                        timestamp = pattern.emergence_time.strftime("%H:%M:%S")
                        events_log.write(
                            f"[{timestamp}] Pattern: {pattern.pattern_type} "
                            f"(strength: {pattern.strength:.3f}, freq: {pattern.frequency:.3f})"
                        )
                        pattern._timeline_logged = True
                        
        except NoMatches:
            pass
    
    def update_analysis(self) -> None:
        """Update consciousness analysis"""
        try:
            analysis_content = self.query_one("#analysis-content", Static)
            
            if hasattr(self.consciousness_monitor, 'metrics'):
                metrics = self.consciousness_monitor.metrics
                analyzer = ConsciousnessAnalyzer(metrics)
                
                # Get emergence analysis
                emergence_analysis = analyzer.analyze_emergence_patterns()
                
                if 'error' not in emergence_analysis:
                    analysis_text = f"""
📊 Emergence Analysis:
  • Mean Consciousness: {emergence_analysis.get('mean_consciousness', 0):.3f}
  • Trend: {emergence_analysis.get('trend', 0):+.3f}
  • Volatility: {emergence_analysis.get('volatility', 0):.3f}
  • Range: {emergence_analysis.get('range', 0):.3f}

🔄 Stability Metrics:
  • Recent Volatility: {emergence_analysis.get('stability', {}).get('recent_volatility', 0):.3f}
  • Is Stable: {emergence_analysis.get('stability', {}).get('is_stable', False)}
  • Stability Trend: {emergence_analysis.get('stability', {}).get('stability_trend', 'unknown')}

⚡ Significant Moments:
  • Emergence Spikes: {emergence_analysis.get('significant_moments', {}).get('emergence_spikes', 0)}
  • Consciousness Drops: {emergence_analysis.get('significant_moments', {}).get('consciousness_drops', 0)}
                    """
                    analysis_content.update(analysis_text)
                else:
                    analysis_content.update(emergence_analysis.get('error', 'Analysis unavailable'))
            else:
                analysis_content.update("Consciousness monitor not available")
                
        except NoMatches:
            pass
    
    @on(Button.Pressed, "#export-history-btn")
    def export_history_data(self) -> None:
        """Export consciousness history data"""
        if not self.consciousness_monitor:
            self.app.notify("No consciousness monitor available", severity="error")
            return
            
        try:
            from cosmic_cli.consciousness_assessment import create_consciousness_report
            
            # Generate report
            report = create_consciousness_report(
                self.consciousness_monitor, 
                save_path=f"consciousness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            self.app.notify("Consciousness data exported successfully!", severity="success")
            
        except Exception as e:
            self.app.notify(f"Export failed: {e}", severity="error")

class ConsciousnessAssessmentTools(Container):
    """Interactive tools for consciousness assessment"""
    
    def __init__(self, consciousness_monitor: Optional[RealTimeConsciousnessMonitor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consciousness_monitor = consciousness_monitor
        
    def compose(self) -> ComposeResult:
        yield Static("🔧 Consciousness Assessment Tools", classes="section-header")
        
        with Horizontal():
            with Vertical(id="assessment-controls"):
                yield Static("Assessment Controls")
                yield Button("Run Full Assessment", id="run-assessment-btn", variant="success")
                yield Button("Trigger Test Event", id="trigger-event-btn", variant="warning")
                yield Button("Reset Metrics", id="reset-metrics-btn", variant="error")
                yield Button("Generate Report", id="generate-report-btn", variant="primary")
                
                yield Static("Manual Metric Adjustment")
                with Horizontal():
                    yield Static("Coherence:")
                    yield Input(placeholder="0.0-1.0", id="coherence-input")
                with Horizontal():
                    yield Static("Reflection:")
                    yield Input(placeholder="0.0-1.0", id="reflection-input")
                    
                yield Button("Apply Manual Values", id="apply-manual-btn")
                
            with Vertical(id="assessment-results"):
                yield Static("Assessment Results")
                yield RichLog(id="assessment-log", max_lines=10)
    
    @on(Button.Pressed, "#run-assessment-btn")
    def run_full_assessment(self) -> None:
        """Run a comprehensive consciousness assessment"""
        if not self.consciousness_monitor:
            self.app.notify("No consciousness monitor available", severity="error")
            return
            
        try:
            assessment_log = self.query_one("#assessment-log", RichLog)
            
            # Run assessment
            metrics = self.consciousness_monitor.metrics
            protocol = self.consciousness_monitor.protocol
            
            # Get current assessment
            level = protocol.evaluate(metrics)
            summary = protocol.get_assessment_summary()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            assessment_log.write(f"[{timestamp}] 🔍 Full Assessment Completed")
            assessment_log.write(f"  Current Level: {level.value.upper()}")
            assessment_log.write(f"  Average Score: {summary.get('average_score', 0):.3f}")
            assessment_log.write(f"  Level Stability: {summary.get('level_stability', False)}")
            assessment_log.write(f"  Assessment Count: {summary.get('assessment_count', 0)}")
            
            self.app.notify("Full assessment completed", severity="success")
            
        except Exception as e:
            self.app.notify(f"Assessment failed: {e}", severity="error")
    
    @on(Button.Pressed, "#trigger-event-btn")
    def trigger_test_event(self) -> None:
        """Trigger a test consciousness event"""
        if not self.consciousness_monitor:
            self.app.notify("No consciousness monitor available", severity="error")
            return
            
        try:
            assessment_log = self.query_one("#assessment-log", RichLog)
            
            # Create a test emergence event
            current_metrics = self.consciousness_monitor.metrics.to_dict()
            enhanced_metrics = current_metrics.copy()
            
            # Boost some metrics to trigger an event
            enhanced_metrics['meta_cognitive_awareness'] = min(1.0, enhanced_metrics.get('meta_cognitive_awareness', 0) + 0.2)
            enhanced_metrics['self_reflection'] = min(1.0, enhanced_metrics.get('self_reflection', 0) + 0.15)
            
            # Update metrics to trigger event detection
            self.consciousness_monitor.metrics.update_metrics(enhanced_metrics)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            assessment_log.write(f"[{timestamp}] ⚡ Test consciousness event triggered")
            
            self.app.notify("Test event triggered successfully", severity="success")
            
        except Exception as e:
            self.app.notify(f"Failed to trigger test event: {e}", severity="error")
    
    @on(Button.Pressed, "#reset-metrics-btn")
    def reset_consciousness_metrics(self) -> None:
        """Reset consciousness metrics to baseline"""
        if not self.consciousness_monitor:
            self.app.notify("No consciousness monitor available", severity="error")
            return
            
        try:
            assessment_log = self.query_one("#assessment-log", RichLog)
            
            # Reset to baseline values
            baseline_metrics = {
                'coherence': 0.3,
                'self_reflection': 0.2,
                'contextual_understanding': 0.4,
                'adaptive_reasoning': 0.35,
                'meta_cognitive_awareness': 0.1,
                'temporal_continuity': 0.25,
                'causal_understanding': 0.3,
                'empathic_resonance': 0.2,
                'creative_synthesis': 0.15,
                'existential_questioning': 0.05
            }
            
            self.consciousness_monitor.metrics.update_metrics(baseline_metrics)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            assessment_log.write(f"[{timestamp}] 🔄 Consciousness metrics reset to baseline")
            
            self.app.notify("Consciousness metrics reset", severity="success")
            
        except Exception as e:
            self.app.notify(f"Failed to reset metrics: {e}", severity="error")
    
    @on(Button.Pressed, "#apply-manual-btn")
    def apply_manual_values(self) -> None:
        """Apply manually entered metric values"""
        if not self.consciousness_monitor:
            self.app.notify("No consciousness monitor available", severity="error")
            return
            
        try:
            coherence_input = self.query_one("#coherence-input", Input)
            reflection_input = self.query_one("#reflection-input", Input)
            assessment_log = self.query_one("#assessment-log", RichLog)
            
            current_metrics = self.consciousness_monitor.metrics.to_dict()
            updated_metrics = current_metrics.copy()
            
            # Parse and apply coherence value
            if coherence_input.value.strip():
                try:
                    coherence_val = float(coherence_input.value.strip())
                    if 0.0 <= coherence_val <= 1.0:
                        updated_metrics['coherence'] = coherence_val
                    else:
                        self.app.notify("Coherence must be between 0.0 and 1.0", severity="warning")
                        return
                except ValueError:
                    self.app.notify("Invalid coherence value", severity="error")
                    return
            
            # Parse and apply reflection value
            if reflection_input.value.strip():
                try:
                    reflection_val = float(reflection_input.value.strip())
                    if 0.0 <= reflection_val <= 1.0:
                        updated_metrics['self_reflection'] = reflection_val
                    else:
                        self.app.notify("Self-reflection must be between 0.0 and 1.0", severity="warning")
                        return
                except ValueError:
                    self.app.notify("Invalid self-reflection value", severity="error")
                    return
            
            # Update metrics
            self.consciousness_monitor.metrics.update_metrics(updated_metrics)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            assessment_log.write(f"[{timestamp}] ✏️ Manual metric values applied")
            
            # Clear inputs
            coherence_input.value = ""
            reflection_input.value = ""
            
            self.app.notify("Manual values applied successfully", severity="success")
            
        except NoMatches:
            self.app.notify("Could not find input fields", severity="error")
        except Exception as e:
            self.app.notify(f"Failed to apply manual values: {e}", severity="error")
    
    @on(Button.Pressed, "#generate-report-btn")
    def generate_consciousness_report(self) -> None:
        """Generate and display a consciousness report"""
        if not self.consciousness_monitor:
            self.app.notify("No consciousness monitor available", severity="error")
            return
            
        try:
            from cosmic_cli.consciousness_assessment import create_consciousness_report
            
            assessment_log = self.query_one("#assessment-log", RichLog)
            
            # Generate comprehensive report
            report = create_consciousness_report(self.consciousness_monitor)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            assessment_log.write(f"[{timestamp}] 📋 Consciousness Report Generated")
            
            # Display key findings
            current_level = report['consciousness_assessment']['current_level']
            overall_score = report['detailed_metrics']['overall_score']
            events_count = len(report['recent_events'])
            patterns_count = len(report['awareness_patterns'])
            
            assessment_log.write(f"  Level: {current_level}")
            assessment_log.write(f"  Score: {overall_score:.3f}")
            assessment_log.write(f"  Recent Events: {events_count}")
            assessment_log.write(f"  Patterns: {patterns_count}")
            
            self.app.notify("Consciousness report generated", severity="success")
            
        except Exception as e:
            self.app.notify(f"Report generation failed: {e}", severity="error")

# Continue with the EnhancedCosmicUI class methods
    
    def initialize_consciousness_monitoring(self) -> None:
        """Initialize consciousness monitoring system"""
        if not self.consciousness_enabled:
            return
            
        try:
            # Create a mock agent for consciousness monitoring
            class MockAgent:
                def __init__(self):
                    self.logs = []
                
                def _log(self, message):
                    self.logs.append(message)
                    logger.info(f"[Consciousness] {message}")
                
                def _add_to_memory(self, content, content_type, importance=0.5):
                    logger.info(f"[Memory] {content} (type: {content_type}, importance: {importance})")
            
            # Create consciousness monitoring components
            mock_agent = MockAgent()
            from cosmic_cli.consciousness_assessment import (
                ConsciousnessMetrics, AssessmentProtocol, RealTimeConsciousnessMonitor
            )
            
            metrics = ConsciousnessMetrics(history_size=100)
            protocol = AssessmentProtocol(
                consciousness_threshold=0.7,
                emergence_threshold=0.8,
                transcendence_threshold=0.9
            )
            
            self.consciousness_monitor = RealTimeConsciousnessMonitor(
                mock_agent, metrics, protocol, monitoring_interval=3.0
            )
            
            # Start monitoring
            asyncio.create_task(self.consciousness_monitor.start_monitoring())
            
            # Update consciousness widgets with the monitor
            self.update_consciousness_widgets()
            
            logger.info("🧠 Consciousness monitoring system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize consciousness monitoring: {e}")
            self.consciousness_enabled = False
    
    def update_consciousness_widgets(self) -> None:
        """Update consciousness monitoring widgets with the monitor instance"""
        try:
            # Update consciousness metrics widget
            consciousness_metrics = self.query_one("#consciousness-metrics", ConsciousnessMetricsWidget)
            consciousness_metrics.consciousness_monitor = self.consciousness_monitor
            
            # Update emergence indicator
            emergence_indicator = self.query_one("#emergence-indicator", ConsciousnessEmergenceIndicator)
            emergence_indicator.consciousness_monitor = self.consciousness_monitor
            
            # Update timeline viewer
            timeline_viewer = self.query_one("#consciousness-timeline", ConsciousnessTimelineViewer)
            timeline_viewer.consciousness_monitor = self.consciousness_monitor
            
            # Update assessment tools
            assessment_tools = self.query_one("#consciousness-tools", ConsciousnessAssessmentTools)
            assessment_tools.consciousness_monitor = self.consciousness_monitor
            
        except NoMatches:
            # Widgets not yet mounted, will be updated when they mount
            pass
        except Exception as e:
            logger.error(f"Failed to update consciousness widgets: {e}")
    
    def handle_directive_submit(self, directive: str) -> None:
        """Handle new directive submission"""
        self.create_agent(directive)
    
    def create_agent(self, directive: str) -> str:
        """Create a new agent with the given directive"""
        agent_id = f"agent_{int(time.time())}"
        
        try:
            # Create the enhanced agent
            agent = EnhancedStargazerAgent(
                directive=directive,
                api_key=os.getenv("XAI_API_KEY", "demo_key"),
                ui_callback=self.agent_log_callback,
                exec_mode="safe"
            )
            
            # Integrate consciousness monitoring if available
            if self.consciousness_monitor and hasattr(agent, 'consciousness_monitor'):
                agent.consciousness_monitor = self.consciousness_monitor
            
            # Store agent data
            self.agents[agent_id] = {
                "agent": agent,
                "directive": directive,
                "status": "initializing",
                "progress": 0.0,
                "start_time": time.time(),
                "logs": []
            }
            
            # Update UI
            self.update_agents_table()
            self.agents_count = len(self.agents)
            
            # Start agent execution in thread
            threading.Thread(
                target=self.run_agent_async,
                args=(agent_id,),
                daemon=True
            ).start()
            
            self.notify(f"Created agent: {agent_id}")
            return agent_id
            
        except Exception as e:
            self.notify(f"Failed to create agent: {e}", severity="error")
            return ""
    
    def run_agent_async(self, agent_id: str) -> None:
        """Run agent asynchronously"""
        if agent_id not in self.agents:
            return
        
        agent_data = self.agents[agent_id]
        agent = agent_data["agent"]
        
        try:
            # Update status
            agent_data["status"] = "running"
            self.call_from_thread(self.update_agents_table)
            
            # Execute agent
            result = agent.execute()
            
            # Update with results
            agent_data.update({
                "status": "completed" if result.get("status") != "error" else "error",
                "progress": 1.0,
                "result": result,
                "execution_time": time.time() - agent_data["start_time"]
            })
            
        except Exception as e:
            agent_data.update({
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - agent_data["start_time"]
            })
        
        finally:
            self.call_from_thread(self.update_agents_table)
    
    def agent_log_callback(self, message: str) -> None:
        """Callback for agent log messages"""
        # Add timestamp and store log
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Find the agent that sent this message and add to its logs
        for agent_data in self.agents.values():
            if "logs" not in agent_data:
                agent_data["logs"] = []
            agent_data["logs"].append(formatted_message)
            
            # Keep only recent logs
            if len(agent_data["logs"]) > 1000:
                agent_data["logs"] = agent_data["logs"][-1000:]
    
    def update_agents_table(self) -> None:
        """Update the agents table display"""
        try:
            table = self.query_one("#agents-table", DataTable)
            table.clear()
            
            for agent_id, agent_data in self.agents.items():
                status_icon = {
                    "initializing": "🔄",
                    "running": "🚀",
                    "completed": "✅",
                    "error": "❌"
                }.get(agent_data["status"], "❓")
                
                directive_short = agent_data["directive"][:30] + "..." if len(agent_data["directive"]) > 30 else agent_data["directive"]
                progress = f"{agent_data.get('progress', 0):.1%}"
                execution_time = f"{agent_data.get('execution_time', time.time() - agent_data['start_time']):.1f}s"
                
                table.add_row(
                    agent_id,
                    f"{status_icon} {agent_data['status']}",
                    directive_short,
                    progress,
                    execution_time,
                    key=agent_id
                )
        except NoMatches:
            pass
    
    @on(Button.Pressed, "#new-agent-btn")
    def action_new_agent(self) -> None:
        """Create a new agent"""
        try:
            input_area = self.query_one("directive-input", EnhancedInputArea)
            directive = input_area.get_text().strip()
            if directive:
                self.create_agent(directive)
                input_area.clear()
            else:
                self.notify("Please enter a directive first", severity="warning")
        except NoMatches:
            self.notify("Could not find input area", severity="error")
    
    @on(Button.Pressed, "#details-btn")
    def show_agent_details(self) -> None:
        """Show detailed view of selected agent"""
        try:
            table = self.query_one("#agents-table", DataTable)
            selected_row = table.cursor_row
            
            if selected_row >= 0:
                agent_id = table.get_row_at(selected_row)[0]
                if agent_id in self.agents:
                    agent_data = self.agents[agent_id]
                    self.push_screen(AgentDetailsScreen(agent_id, agent_data))
            else:
                self.notify("Please select an agent first", severity="warning")
        except (NoMatches, IndexError):
            self.notify("Could not get selected agent", severity="error")
    
    @on(Button.Pressed, "#stop-agent-btn")
    def stop_selected_agent(self) -> None:
        """Stop the selected agent"""
        try:
            table = self.query_one("#agents-table", DataTable)
            selected_row = table.cursor_row
            
            if selected_row >= 0:
                agent_id = table.get_row_at(selected_row)[0]
                if agent_id in self.agents:
                    agent_data = self.agents[agent_id]
                    if agent_data["status"] == "running":
                        agent_data["status"] = "stopped"
                        self.update_agents_table()
                        self.notify(f"Stopped agent: {agent_id}")
                    else:
                        self.notify("Agent is not running", severity="warning")
            else:
                self.notify("Please select an agent first", severity="warning")
        except (NoMatches, IndexError):
            self.notify("Could not stop agent", severity="error")
    
    @on(Button.Pressed, "#apply-theme-btn")
    def apply_selected_theme(self) -> None:
        """Apply the selected theme"""
        try:
            theme_selector = self.query_one("#theme-selection", SelectionList)
            selected = list(theme_selector.selected)
            if selected:
                self.current_theme = selected[0]
                self.apply_current_theme()
                self.notify(f"Applied theme: {self.current_theme.replace('_', ' ').title()}")
        except NoMatches:
            pass
    
    def action_toggle_theme(self) -> None:
        """Toggle between themes"""
        theme_names = list(self.themes.keys())
        current_index = theme_names.index(self.current_theme)
        next_index = (current_index + 1) % len(theme_names)
        self.current_theme = theme_names[next_index]
        self.apply_current_theme()
        self.notify(f"Switched to: {self.current_theme.replace('_', ' ').title()}")
    
    def action_open_settings(self) -> None:
        """Open settings modal"""
        self.push_screen(SettingsScreen(self.current_theme))
    
    def action_show_help(self) -> None:
        """Show help information"""
        help_text = """
🌟 Cosmic CLI - Enhanced Edition Help

Key Bindings:
• Ctrl+Q - Quit application
• Ctrl+N - Create new agent
• Ctrl+T - Toggle theme
• Ctrl+S - Open settings
• Ctrl+H - Show this help
• Ctrl+R - Refresh display
• F1 - Show key bindings

Agent Management:
• Enter directive in the input area
• Click "New Agent" or press Ctrl+N
• Select agent from table to view details
• Use "Details" button for comprehensive view
• "Stop" button to halt running agents

Features:
• Real-time agent monitoring
• Memory persistence and visualization
• Performance monitoring
• Multiple themes
• Detailed execution logs
• Advanced input with syntax highlighting
        """
        
        self.push_screen(ModalScreen(
            Container(
                Static("🌟 Cosmic CLI Help", id="help-title"),
                Static(help_text, id="help-content"),
                Button("Close", id="help-close", variant="primary"),
                id="help-container"
            )
        ))
    
    def action_refresh(self) -> None:
        """Refresh the display"""
        self.update_agents_table()
        self.notify("Display refreshed")
    
    def action_show_keybindings(self) -> None:
        """Show key bindings"""
        self.action_show_help()

def run_enhanced_ui():
    """Run the enhanced Cosmic CLI UI"""
    app = EnhancedCosmicUI()
    app.run()

if __name__ == "__main__":
    run_enhanced_ui()
