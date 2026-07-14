import os
from textual.app import App, ComposeResult
from textual.widgets import (
    DataTable,
    Static,
    Input,
    Button,
    Header,
    Footer,
    RichLog,
)
from textual.containers import Vertical, Horizontal, Container
from textual.screen import ModalScreen
from textual.reactive import reactive
from pyfiglet import Figlet
from cosmic_cli.agents import StargazerAgent


class APIKeyScreen(ModalScreen[str]):
    """Modal prompt for xAI API key using Textual best practices.
    Password-masked Input, supports Enter (on_submitted) and explicit Button.
    """

    def compose(self) -> ComposeResult:
        yield Static("🔑 Enter your xAI API key (Ctrl+K to open)", id="prompt")
        yield Input(
            placeholder="xai-... or sk-...",
            password=True,
            id="api_key_input",
        )
        with Horizontal(id="key_buttons"):
            yield Button("Submit", variant="primary", id="submit_key")
            yield Button("Cancel", variant="default", id="cancel_key")

    def on_mount(self) -> None:
        self.query_one("#api_key_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "api_key_input":
            self._submit_key()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit_key":
            self._submit_key()
        elif event.button.id == "cancel_key":
            self.dismiss(None)

    def _submit_key(self) -> None:
        key_input = self.query_one("#api_key_input", Input)
        key = key_input.value.strip()
        if key:
            os.environ["XAI_API_KEY"] = key
            self.dismiss(key)
        else:
            self.app.notify("API key cannot be empty.", severity="error")


class DirectivesUI(App):
    """COSMIC CLI - Modern Textual TUI (2026 best practices).
    Uses ModalScreen for API key, RichLog, Header/Footer, proper Input+Button,
    bindings (ctrl+k), containers, safe refresh.
    """

    BINDINGS = [
        ("ctrl+k", "prompt_api_key", "Set API Key"),
        ("ctrl+c", "quit", "Quit"),
        ("q", "quit", "Quit"),
        ("enter", "deploy_directive", "Deploy"),
    ]

    CSS = """
    DataTable { margin: 1; height: 12; }
    Static.banner { color: cyan; text-align: center; margin: 1; }
    RichLog { height: 10; border: round cyan; margin: 1; }
    Horizontal#input_row { margin: 1; }
    #prompt { text-align: center; color: yellow; margin-bottom: 1; }
    #key_buttons { margin-top: 1; }
    """

    def __init__(self, testing: bool = False):
        super().__init__()
        self.agents = {}
        self.show_logs = {}
        self.figlet = Figlet(font='doom')
        self.testing = testing

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(self.figlet.renderText('COSMIC CLI'), classes="banner")
        yield DataTable(id="directives")
        with Horizontal(id="input_row"):
            yield Input(
                placeholder="Enter directive (e.g. analyze data.txt)  |  Ctrl+K for API key",
                id="directive_input",
            )
            yield Button("Deploy", id="deploy_btn", variant="primary")
        yield RichLog(id="logs", highlight=True, markup=True)
        yield Footer()

    def on_mount(self):
        table = self.query_one("#directives", DataTable)
        table.add_columns("Directive", "Status", "Level", "Score", "Logs")
        table.cursor_type = "row"
        # Prompt for key early (on_mount) unless testing or already set
        if not (os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")) and not self.testing:
            self.action_prompt_api_key()
        if not self.testing:
            self.set_interval(1, self._refresh_panel)

    def action_prompt_api_key(self) -> None:
        """Open the API key modal. Used on_mount and before deploy if missing."""
        def handle_result(key: str | None) -> None:
            if key:
                self.notify("🔑 API key set for this session.", severity="information")
            # If no key and was pending deploy, user can re-attempt after setting via Ctrl+K
        self.push_screen(APIKeyScreen(), handle_result)

    def action_deploy_directive(self) -> None:
        """Action for enter / deploy binding."""
        self._handle_deploy()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "directive_input":
            self._handle_deploy()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "deploy_btn":
            self._handle_deploy()

    def _handle_deploy(self) -> None:
        input_widget = self.query_one("#directive_input", Input)
        directive = input_widget.value.strip()
        if not directive:
            return
        self.add_directive(directive)
        input_widget.value = ""

    def thread_safe_refresh(self, agent=None):
        """A thread-safe way to call _refresh_panel."""
        self.call_from_thread(self._refresh_panel, agent)

    def add_directive(self, directive):
        if directive in self.agents:
            self.notify(f"Directive '{directive}' already deployed!", severity="warning")
            return

        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            self.notify("No API key set. Opening prompt (or use Ctrl+K)...", severity="warning")
            self.action_prompt_api_key()
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
        """Safe refresh: clear/rebuild table + write to RichLog (no remove_children hacks on dynamic Statics)."""
        try:
            table = self.query_one("#directives", DataTable)
            table.clear()
            for dir, ag in self.agents.items():
                level = getattr(ag, "model", "N/A")
                score = float(getattr(ag, "steps_taken", 0) or 0)
                log_action = (
                    f"Show Logs ({len(ag.logs)})"
                    if not self.show_logs.get(dir)
                    else f"Hide Logs ({len(ag.logs)})"
                )
                table.add_row(dir, ag.status, level, f"{score:.3f}", log_action, key=dir)

            # Modern logs via RichLog
            log_widget = self.query_one("#logs", RichLog)
            log_widget.clear()
            for dir, show in list(self.show_logs.items()):
                if show and dir in self.agents:
                    ag = self.agents[dir]
                    log_widget.write(f"[bold magenta]--- Stargazer Logs: {dir} ---[/]")
                    for line in ag.logs[-50:]:  # safety cap
                        log_widget.write(str(line))
        except Exception:
            # Guard against refresh during shutdown / testing partial mounts
            pass

    def on_data_table_row_selected(self, event):
        directive = event.data_table.get_row(event.row_key)[0]
        self.toggle_logs(directive)
