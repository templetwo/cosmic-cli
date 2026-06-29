#!/usr/bin/env python3
"""
🚀 Enhanced Input Component for Cosmic CLI
Improved multi-line input handling with syntax highlighting and large prompts support
"""

from typing import Callable, Optional, List, Dict, Any
import os
import re
import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import TextArea, Button, Label, Static
from textual.binding import Binding
from textual.reactive import reactive
from textual.events import Key
from textual.css.query import NoMatches

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown

# Try to import syntax highlighting features
try:
    import pygments
    from pygments.lexers import get_lexer_by_name, guess_lexer
    SYNTAX_HIGHLIGHTING = True
except ImportError:
    SYNTAX_HIGHLIGHTING = False


class EnhancedInputArea(Container):
    """Enhanced input area with support for large prompts"""
    
    DEFAULT_CSS = """
    EnhancedInputArea {
        width: 100%;
        height: auto;
        min-height: 5;
        background: $surface;
        border: solid $primary;
        padding: 0 1 0 1;
    }
    
    EnhancedInputArea TextArea {
        height: auto;
        min-height: 3;
        max-height: 20;  /* Maximum height before scrolling */
        border: none;
        background: $surface-darken-1;
    }
    
    EnhancedInputArea #controls {
        width: 100%;
        height: 1;
        align: right middle;
    }
    
    EnhancedInputArea Button {
        margin: 0 1 0 0;
    }
    
    EnhancedInputArea #status {
        color: $text-muted;
    }
    
    EnhancedInputArea #expand_toggle {
        width: 3;
        background: $primary-background;
    }
    
    .expanded TextArea {
        min-height: 10;
        max-height: 30;
    }
    """
    
    expanded = reactive(False)
    char_count = reactive(0)
    line_count = reactive(1)
    
    def __init__(
        self, 
        name: str = "enhanced_input", 
        placeholder: str = "Enter or paste your directive here... (supports multi-line input)",
        on_submit: Optional[Callable[[str], None]] = None,
        syntax: str = "markdown",
    ):
        super().__init__(id=name)
        self.placeholder = placeholder
        self.on_submit_callback = on_submit
        self.syntax_language = syntax
        self._last_text = ""
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced input component"""
        # Main text input area
        yield TextArea(
            language=self.syntax_language if SYNTAX_HIGHLIGHTING else None,
            id="input_text"
        )
        
        # Controls row
        with Horizontal(id="controls"):
            yield Label("0 chars | 1 line", id="status")
            yield Button("Expand", id="expand_toggle", variant="primary")
            yield Button("Clear", id="clear_button", variant="error")
            yield Button("Submit", id="submit_button", variant="success")
    
    def on_mount(self) -> None:
        """Initialize after mounting to DOM"""
        self.text_area = self.query_one("#input_text", TextArea)
        
        # Set up keyboard bindings
        self.text_area.focus()
        
        # Set up watches for text changes
        self.watch(self.text_area, "text", self._on_text_changed)
    
    def _on_text_changed(self, value: str) -> None:
        """Handle text changes in the input area"""
        if value is None:
            value = ""
        self.char_count = len(value)
        self.line_count = value.count('\n') + 1
        
        # Update status
        status = self.query_one("#status", Label)
        status.update(f"{self.char_count} chars | {self.line_count} lines")
        
        # Store last text
        self._last_text = value
    
    def watch_expanded(self, expanded: bool) -> None:
        """Handle expanded state changes"""
        if expanded:
            self.add_class("expanded")
            self.query_one("#expand_toggle", Button).label = "Shrink"
        else:
            self.remove_class("expanded")
            self.query_one("#expand_toggle", Button).label = "Expand"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "expand_toggle":
            self.expanded = not self.expanded
        
        elif button_id == "clear_button":
            self.text_area.clear()
            self.text_area.focus()
        
        elif button_id == "submit_button":
            self.submit_text()
    
    def on_key(self, event: Key) -> None:
        """Handle key presses"""
        # Check for Ctrl+Enter to submit
        if event.key == "ctrl+enter":
            self.submit_text()
            event.prevent_default()
            event.stop()
    
    def submit_text(self) -> None:
        """Submit the current text"""
        text = self.text_area.text.strip()
        if text and self.on_submit_callback:
            self.on_submit_callback(text)
            self.text_area.clear()
        self.text_area.focus()
    
    def set_text(self, text: str) -> None:
        """Set text content"""
        self.text_area.text = text
        self.text_area.focus()
    
    def append_text(self, text: str) -> None:
        """Append text to current content"""
        current = self.text_area.text
        if current and not current.endswith("\n"):
            self.text_area.text = current + "\n" + text
        else:
            self.text_area.text = current + text
        self.text_area.focus()
    
    def get_text(self) -> str:
        """Get current text content"""
        return self.text_area.text
    
    def clear(self) -> None:
        """Clear the input"""
        self.text_area.clear()
        self.text_area.focus()


class FileDropArea(Static):
    """Support for drag-and-drop file content"""
    
    DEFAULT_CSS = """
    FileDropArea {
        width: 100%;
        height: 2;
        margin: 0 0 1 0;
        background: $surface-darken-2;
        color: $text-muted;
        text-align: center;
        text-style: italic;
        border: dashed $primary-darken-2;
    }
    FileDropArea.active {
        background: $primary-background;
        border: dashed $primary;
    }
    """
    
    def __init__(self, on_file_dropped: Callable[[str], None], name: str = "file_drop"):
        super().__init__("📄 Drag and drop a file here", id=name)
        self.on_file_dropped = on_file_dropped
        self.dragging = False
    
    def on_dragover(self, event) -> None:
        """Handle drag over event"""
        self.add_class("active")
        event.prevent_default()
    
    def on_dragleave(self, event) -> None:
        """Handle drag leave event"""
        self.remove_class("active")
    
    def on_drop(self, event) -> None:
        """Handle file drop"""
        self.remove_class("active")
        event.prevent_default()
        
        # In a real implementation, this would process the dropped file
        # For now we'll simulate it with a notification
        if hasattr(event, 'files') and event.files:
            file_path = event.files[0]
            try:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    self.on_file_dropped(file_content)
            except Exception as e:
                self.app.notify(f"Error reading file: {e}", severity="error")
        else:
            self.app.notify("Drop file content feature requires browser integration", severity="warning")


class EnhancedPromptApp(App):
    """Demo app to showcase the enhanced input area"""
    
    TITLE = "Cosmic CLI - Enhanced Input"
    
    CSS = """
    #container {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    #title {
        width: 100%;
        height: 3;
        background: $primary-background;
        color: $text;
        text-align: center;
        padding: 1;
        margin-bottom: 1;
    }
    #response_area {
        width: 100%;
        height: 1fr;
        min-height: 5;
        margin-top: 1;
        background: $surface-darken-1;
        border: solid $primary;
        padding: 1;
        overflow: auto;
    }
    """
    
    def __init__(self):
        super().__init__()
        self._history: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the enhanced input demo app"""
        with Vertical(id="container"):
            yield Static("🌟 Cosmic CLI Enhanced Input 🌟", id="title")
            yield FileDropArea(self.handle_file_dropped)
            yield EnhancedInputArea(
                name="directive_input", 
                placeholder="Enter your cosmic directive here... (supports large multi-line prompts)",
                on_submit=self.handle_submit
            )
            yield Static("", id="response_area")
    
    def handle_submit(self, text: str) -> None:
        """Handle text submission"""
        self.query_one("#response_area").update(
            f"Processing directive: {text[:100]}..." if len(text) > 100 else f"Processing directive: {text}"
        )
        
        # In a real app, this would send the text to the agent for processing
        self._history.append({
            "type": "input",
            "content": text
        })
        
        # Simulate response
        self.simulate_response(text)
    
    def handle_file_dropped(self, content: str) -> None:
        """Handle dropped file content"""
        input_area = self.query_one(EnhancedInputArea)
        input_area.append_text(content)
        self.notify("File content added to input", severity="information")
    
    def simulate_response(self, text: str) -> None:
        """Simulate agent response for demo purposes"""
        def update_response():
            response_area = self.query_one("#response_area")
            response_area.update(
                Panel(
                    Markdown(f"Received directive with {len(text)} characters and {text.count(chr(10))+1} lines.\n\n"
                           f"First line: `{text.split(chr(10))[0]}`\n\n"
                           f"This enhanced input component supports:\n"
                           f"- Large multi-line directives\n"
                           f"- Syntax highlighting\n"
                           f"- File drag-and-drop\n"
                           f"- Character and line counting\n"),
                    title="✨ Cosmic Response",
                    border_style="cyan"
                )
            )
        
        # Schedule the update to simulate async processing
        asyncio.get_event_loop().call_later(0.5, update_response)


# Demo function to show the enhanced input
def demo_enhanced_input():
    """Run the enhanced input demo"""
    app = EnhancedPromptApp()
    app.run()


if __name__ == "__main__":
    demo_enhanced_input()
