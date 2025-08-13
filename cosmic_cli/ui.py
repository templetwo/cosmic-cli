import os
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static, Input, TextArea
from textual.containers import Vertical
from textual.reactive import reactive
from pyfiglet import Figlet
from cosmic_cli.agents import StargazerAgent

class DirectivesUI(App):
    """COSMIC CLI - Dynamic banner with Textual TUI"""
    
    CSS = """
    DataTable { margin: 1; }
    Static { border: double magenta; padding: 1; }
    .banner { color: cyan; }
    .log_panel { margin-top: 1; }
    """

    def __init__(self, testing: bool = False):
        super().__init__()
        self.agents = {}
        self.show_logs = {}
        self.figlet = Figlet(font='doom')  # Switched to 'doom' for clear, letter-by-letter style
        self.testing = testing

    def compose(self) -> ComposeResult:
        yield Static(self.figlet.renderText('COSMIC CLI'), classes="banner")
        yield DataTable(id="directives")
        yield TextArea(id="directive_input")
        yield Vertical(id="logs_container")

    def on_mount(self):
        table = self.query_one("#directives", DataTable)
        table.add_columns("Directive", "Status", "Level", "Score", "Logs")
        table.cursor_type = "row"
        input_widget = self.query_one("#directive_input", TextArea)
        # Add submit binding
        self.bind("enter", self.on_submit)
        # Add a timer to refresh the panel every second, but not in testing mode
        if not self.testing:
            self.set_interval(1, self._refresh_panel)

    def on_submit(self):
        input_widget = self.query_one("#directive_input", TextArea)
        directive = input_widget.text.strip()
        if directive:
            # Auto-format if it looks like a workflow
            if ' ' in directive and not directive.startswith('/'):
                directive = f"workflow {directive}"
            self.add_directive(directive)
            input_widget.clear()

    def thread_safe_refresh(self, agent=None):
        """A thread-safe way to call _refresh_panel."""
        self.call_from_thread(self._refresh_panel, agent)

    def add_directive(self, directive):
        if directive in self.agents:
            self.notify(f"Directive '{directive}' already deployed!", severity="warning")
            return

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            self.notify("XAI_API_KEY not found in environment variables.", severity="error")
            return

        agent = StargazerAgent(
            directive=directive,
            api_key=api_key,
            ui_callback=self.thread_safe_refresh
        )
        self.agents[directive] = agent
        self.show_logs[directive] = False
        agent.run()
        self._refresh_panel()

    def toggle_logs(self, directive):
        if directive in self.agents:
            self.show_logs[directive] = not self.show_logs[directive]
            self._refresh_panel()
        else:
            self.notify(f"Directive '{directive}' not found!", severity="error")

    def _refresh_panel(self, agent=None):
        table = self.query_one("#directives", DataTable)
        table.clear()
        for dir, ag in self.agents.items():
            level = ag.consciousness_monitor.last_consciousness_level.value if hasattr(ag, 'consciousness_monitor') else "N/A"
            score = ag.consciousness_metrics.get_overall_score() if hasattr(ag, 'consciousness_metrics') else 0.0
            log_action = f"Show Logs ({len(ag.logs)})" if not self.show_logs[dir] else f"Hide Logs ({len(ag.logs)})"
            table.add_row(dir, ag.status, level, f"{score:.3f}", log_action, key=dir)

        logs_container = self.query_one("#logs_container", Vertical)
        logs_container.remove_children()
        for dir, show in self.show_logs.items():
            if show and dir in self.agents:
                ag = self.agents[dir]
                log_text = "\n".join(ag.logs)
                logs_container.mount(Static(f"[bold magenta]Stargazer Logs: {dir}[/]\n{log_text}", classes="log_panel"))

    def on_data_table_row_selected(self, event):
        directive = event.data_table.get_row(event.row_key)[0]
        self.toggle_logs(directive) 