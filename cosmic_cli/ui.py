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

    def __init__(self):
        super().__init__()
        self.agents = {}
        self.show_logs = {}
        self.figlet = Figlet(font='doom')  # Switched to 'doom' for clear, letter-by-letter style

    def compose(self) -> ComposeResult:
        yield Static(self.figlet.renderText('COSMIC CLI'), classes="banner")
        yield DataTable(id="directives")
        yield TextArea(id="directive_input")
        yield Vertical(id="logs_container")

    def on_mount(self):
        table = self.query_one("#directives", DataTable)
        table.add_columns("Directive", "Status", "Logs")
        table.cursor_type = "row"
        input_widget = self.query_one("#directive_input", TextArea)
        # Add submit binding
        self.bind("enter", self.on_submit)

    def on_submit(self):
        input_widget = self.query_one("#directive_input", TextArea)
        directive = input_widget.text.strip()
        if directive:
            # Auto-format if it looks like a workflow
            if ' ' in directive and not directive.startswith('/'):
                directive = f"workflow {directive}"
            self.add_directive(directive)
            input_widget.clear()

    def add_directive(self, directive):
        if directive in self.agents:
            self.notify(f"Directive '{directive}' already deployed!", severity="warning")
            return
        agent = StargazerAgent(directive, self._refresh_panel)
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
            log_action = f"Show Logs ({len(ag.logs)})" if not self.show_logs[dir] else f"Hide Logs ({len(ag.logs)})"
            table.add_row(dir, ag.status, log_action, key=dir)

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