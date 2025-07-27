#!/usr/bin/env python3
"""
🌟 Enhanced UI for Cosmic CLI with improved input handling
"""

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static, Footer
from textual.containers import Vertical, Container
from textual.reactive import reactive
from textual.binding import Binding
from pyfiglet import Figlet

from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console
from rich.text import Text

from cosmic_cli.agents import StargazerAgent
from cosmic_cli.enhanced_input import EnhancedInputArea, FileDropArea
from cosmic_cli.file_browser import cosmic_file_picker

import threading
import time
import asyncio


class DirectivesUIEnhanced(App):
    """Enhanced COSMIC CLI - Terminal UI with improved input handling"""
    
    TITLE = "COSMIC CLI - Sacred Terminal Companion"
    SUB_TITLE = "Powered by xAI's Grok-4"
    
    CSS = """
    #header {
        dock: top;
        height: 8;
        background: $surface;
        padding: 1;
        border-bottom: solid $primary;
    }
    
    #banner {
        width: 100%;
        content-align: center middle;
        color: $accent;
        text-style: bold;
    }
    
    #table_container {
        height: 10;
        width: 100%;
        margin: 1 0 1 0;
        min-height: 5;
        max-height: 15;
    }
    
    #directives {
        width: 100%;
        height: 100%;
    }
    
    #input_container {
        width: 100%;
        min-height: 7;
        margin-bottom: 1;
    }
    
    #logs_container {
        width: 100%;
        height: 1fr;
        min-height: 10;
        border: solid $primary-darken-1;
        background: $surface-darken-2;
        overflow: auto;
        padding: 1;
    }
    
    .log_panel {
        margin-bottom: 1;
        padding: 1;
    }
    
    #file_actions {
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    
    Footer {
        background: $primary-background-darken-1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_directive", "New Directive"),
        Binding("ctrl+o", "open_file", "Open File"),
        Binding("f5", "refresh", "Refresh"),
    ]
    
    def __init__(self):
        super().__init__()
        self.agents = {}
        self.show_logs = {}
        self.figlet = Figlet(font='slant')  # More readable font
        self.console = Console()
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced Cosmic CLI UI"""
        # Header with banner
        with Container(id="header"):
            yield Static(self.figlet.renderText('COSMIC CLI'), id="banner")
        
        # File actions row
        with Container(id="file_actions"):
            yield Static("[b]🔍 DIRECTIVES TABLE[/b] - Manage running tasks")
        
        # Directives table
        with Container(id="table_container"):
            yield DataTable(id="directives")
        
        # Enhanced input area
        with Container(id="input_container"):
            yield FileDropArea(self.handle_file_dropped)
            yield EnhancedInputArea(
                name="directive_input",
                placeholder="Enter your cosmic directive here... (supports multi-line prompts and Ctrl+Enter to submit)",
                on_submit=self.on_directive_submit,
                syntax="markdown"
            )
        
        # Logs container
        yield Vertical(id="logs_container")
        
        # Footer
        yield Footer()
    
    def on_mount(self):
        """Setup the UI when mounted"""
        # Configure the directives table
        table = self.query_one("#directives", DataTable)
        table.add_columns("Directive", "Status", "Logs", "Actions")
        table.cursor_type = "row"
        
        # Welcome message in logs
        logs_container = self.query_one("#logs_container", Vertical)
        logs_container.mount(
            Static(
                Panel(
                    Markdown(
                        "# 🌟 Welcome to Cosmic CLI\n\n"
                        "Enter a directive in the input area below to begin.\n\n"
                        "**Features:**\n"
                        "- Multi-line directive input (use Ctrl+Enter to submit)\n"
                        "- File drag-and-drop for content\n"
                        "- Syntax highlighting\n"
                        "- Character and line counting\n"
                    ),
                    title="Cosmic CLI - Enhanced Input",
                    border_style="cyan"
                ),
                classes="log_panel"
            )
        )
    
    def handle_file_dropped(self, content: str) -> None:
        """Handle file content dropped onto the input area"""
        input_area = self.query_one(EnhancedInputArea)
        input_area.append_text(content)
        self.notify("📄 File content added to input", severity="information")
    
    def on_directive_submit(self, directive: str) -> None:
        """Handle submitted directive from enhanced input"""
        if directive:
            # Auto-format if it looks like a workflow
            if ' ' in directive and not directive.startswith('/'):
                directive = f"workflow {directive}"
            
            # Truncate for display in table if too long
            display_directive = directive[:50] + "..." if len(directive) > 50 else directive
            self.add_directive(directive, display_directive)
            
            # Show a notification
            self.notify(f"🚀 Directive launched: {display_directive}", severity="information")
    
    def add_directive(self, directive: str, display_name: str = None) -> None:
        """Add a new directive to be executed"""
        # Check if directive already exists
        if directive in self.agents:
            self.notify(f"Directive '{display_name or directive}' already deployed!", severity="warning")
            return
        
        # Create the agent
        try:
            agent = StargazerAgent(directive, self._refresh_callback)
            self.agents[directive] = agent
            self.show_logs[directive] = True  # Show logs by default
            
            # Start agent in a thread
            self._run_agent_in_thread(agent)
            
            # Update the UI
            self._refresh_ui()
        except Exception as e:
            self.notify(f"Error creating agent: {e}", severity="error")
    
    def _run_agent_in_thread(self, agent: StargazerAgent) -> None:
        """Run the agent in a background thread"""
        def run_agent():
            try:
                agent.execute()
            except Exception as e:
                # Use call_from_thread to safely update UI from background thread
                self.call_from_thread(
                    self.notify, f"Agent error: {e}", "error"
                )
        
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
    
    def _refresh_callback(self, msg: str) -> None:
        """Callback for agent to refresh UI"""
        # Use call_from_thread to safely update UI from background thread
        self.call_from_thread(self._refresh_ui)
    
    def _refresh_ui(self) -> None:
        """Refresh the UI to reflect current state"""
        # Update directives table
        table = self.query_one("#directives", DataTable)
        table.clear()
        
        for dir_text, agent in self.agents.items():
            # Create display name (truncated if needed)
            display_name = dir_text[:50] + "..." if len(dir_text) > 50 else dir_text
            
            # Determine log action text
            log_action = f"Hide Logs" if self.show_logs.get(dir_text, False) else f"Show Logs"
            
            # Get agent status
            status = getattr(agent, "status", "⏳")
            
            # Add row with actions
            table.add_row(
                display_name, 
                status, 
                log_action,
                "🗑️ Remove",
                key=dir_text
            )
        
        # Update logs container
        logs_container = self.query_one("#logs_container", Vertical)
        logs_container.remove_children()
        
        # Add logs for visible agents
        for dir_text, show in self.show_logs.items():
            if show and dir_text in self.agents:
                agent = self.agents[dir_text]
                
                # Get agent logs
                logs = getattr(agent, "logs", [])
                if not logs:
                    logs = ["No logs available yet"]
                
                # Display logs
                log_text = "\n".join(logs)
                logs_container.mount(
                    Static(
                        Panel(
                            Text(log_text),
                            title=f"🛰️ Stargazer Logs: {dir_text[:30]}{'...' if len(dir_text) > 30 else ''}",
                            border_style="magenta"
                        ),
                        classes="log_panel"
                    )
                )
    
    def on_data_table_row_selected(self, event) -> None:
        """Handle row selection in directives table"""
        table = event.data_table
        row_key = event.row_key
        cell_value = table.get_cell_at((event.row, 2))
        
        if "Logs" in cell_value:
            # Toggle logs visibility
            self.toggle_logs(row_key)
        elif "Remove" in table.get_cell_at((event.row, 3)):
            # Remove agent
            self.remove_directive(row_key)
    
    def toggle_logs(self, directive: str) -> None:
        """Toggle visibility of logs for a directive"""
        if directive in self.agents:
            self.show_logs[directive] = not self.show_logs.get(directive, False)
            self._refresh_ui()
        else:
            self.notify(f"Directive '{directive}' not found!", severity="error")
    
    def remove_directive(self, directive: str) -> None:
        """Remove a directive and its agent"""
        if directive in self.agents:
            # Clean up agent resources if needed
            # This depends on agent implementation
            
            # Remove from our collections
            self.agents.pop(directive)
            if directive in self.show_logs:
                self.show_logs.pop(directive)
            
            # Update UI
            self._refresh_ui()
            self.notify(f"Directive removed", severity="information")
    
    def action_new_directive(self) -> None:
        """Create a new empty directive (keyboard shortcut)"""
        input_area = self.query_one(EnhancedInputArea)
        input_area.clear()
        input_area.focus()
    
    def action_open_file(self) -> None:
        """Open a file to load content into input"""
        def open_file_async():
            # This simulates file picking since we can't do it directly
            # In a real app, this would use the cosmic_file_picker
            
            # Notify user that we're opening file browser
            self.notify("Opening file browser...", severity="information")
            
            # Simulate delay for file picking
            time.sleep(0.5)
            
            # Simulate file content
            file_content = "# Sample Cosmic Directive\n\nRead the main.py file and explain what it does.\n\nThen analyze the file for potential improvements."
            
            # Update input area with file content
            def update_input():
                input_area = self.query_one(EnhancedInputArea)
                input_area.set_text(file_content)
                self.notify("File content loaded", severity="success")
            
            # Use call_from_thread to safely update UI
            self.call_from_thread(update_input)
        
        # Run file opening in a thread
        threading.Thread(target=open_file_async, daemon=True).start()
    
    def action_refresh(self) -> None:
        """Refresh the UI (F5 key)"""
        self._refresh_ui()
        self.notify("UI refreshed", severity="information")


# Function to run the enhanced UI
def run_enhanced_ui():
    """Run the enhanced Cosmic CLI UI"""
    app = DirectivesUIEnhanced()
    app.run()


if __name__ == "__main__":
    run_enhanced_ui()
