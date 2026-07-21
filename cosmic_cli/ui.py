import os
from datetime import datetime
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
from rich.text import Text
from cosmic_cli.agents import StargazerAgent
from cosmic_cli import theme


class APIKeyScreen(ModalScreen[str]):
    """Modal prompt for xAI API key using Textual best practices.
    Password-masked Input, supports Enter (on_submitted) and explicit Button.

    Wears the 1d skin: dim scrim over the board, cyan focus border, and copy
    that says plainly what happens to the key.
    """

    BINDINGS = [("escape", "cancel_key", "Cancel")]

    DEFAULT_CSS = f"""
    APIKeyScreen {{
        align: center middle;
        /* `<color> <percentage>` is the form Textual composites over the
           screen below; rgba() renders opaque and hides the board. */
        background: #04060a 45%;
    }}
    APIKeyScreen > #key_dialog {{
        width: 54;
        height: auto;
        padding: 1 2;
        background: {theme.SURFACE};
        border: round {theme.BORDER};
    }}
    APIKeyScreen #prompt {{
        color: {theme.CYAN};
        text-style: bold;
        text-align: left;
        margin-bottom: 0;
    }}
    APIKeyScreen #key_hint {{
        color: {theme.MUTED};
        margin-bottom: 1;
    }}
    APIKeyScreen #api_key_input {{
        background: {theme.PAGE};
        border: tall {theme.BORDER};
    }}
    APIKeyScreen #api_key_input:focus {{
        border: tall {theme.CYAN};
    }}
    APIKeyScreen #key_buttons {{
        height: auto;
        margin-top: 1;
    }}
    APIKeyScreen #key_env {{
        color: {theme.FAINT};
        margin-top: 1;
    }}
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="key_dialog"):
            yield Static("✦ xAI API key", id="prompt")
            yield Static("session only — never written to disk", id="key_hint")
            yield Input(
                placeholder="xai-… or sk-…",
                password=True,
                id="api_key_input",
            )
            with Horizontal(id="key_buttons"):
                yield Button("⏎ SUBMIT", variant="primary", id="submit_key")
                yield Button("esc CANCEL", variant="default", id="cancel_key")
            yield Static("env: XAI_API_KEY", id="key_env")

    def on_mount(self) -> None:
        self.query_one("#api_key_input", Input).focus()

    def action_cancel_key(self) -> None:
        self.dismiss(None)

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
        ("ctrl+k", "prompt_api_key", "api key"),
        ("ctrl+c", "quit", "quit"),
        ("q", "quit", "quit"),
        ("enter", "deploy_directive", "deploy"),
    ]

    TITLE = "✦ COSMIC CLI"
    SUB_TITLE = "stargazer"

    # The 1d skin. Truecolor hexes; nearest ANSI-256 degrades fine.
    CSS = f"""
    Screen {{ background: {theme.PAGE}; color: {theme.TEXT}; }}
    Header {{ background: {theme.SURFACE}; color: {theme.MUTED}; }}
    HeaderTitle {{ color: {theme.CYAN}; text-style: bold; }}
    Static.banner {{ text-align: center; margin: 1 0 0 0; }}
    DataTable {{ margin: 1 2; height: 12; background: {theme.PAGE}; }}
    DataTable > .datatable--header {{ color: {theme.MUTED}; text-style: bold; }}
    DataTable > .datatable--cursor {{ background: {theme.CURSOR}; }}
    Input {{ background: {theme.SURFACE}; border: tall {theme.BORDER}; }}
    Input:focus {{ border: tall {theme.CYAN}; }}
    Button#deploy_btn {{ background: {theme.BLUE}; color: #ffffff; text-style: bold; }}
    RichLog {{
        height: 10;
        margin: 0 2 1 2;
        background: {theme.PANEL};
        border: round {theme.BORDER};
    }}
    Horizontal#input_row {{ height: auto; margin: 1 2; }}
    Horizontal#input_row > Input {{ width: 1fr; }}
    Horizontal#input_row > Button {{ width: auto; min-width: 14; margin-left: 1; }}
    Footer {{ background: {theme.SURFACE}; }}
    FooterKey {{ background: {theme.SURFACE}; color: {theme.MUTED}; }}
    FooterKey > .footer-key--key {{ background: {theme.CYAN}; color: {theme.PAGE}; }}
    FooterKey > .footer-key--description {{ color: {theme.MUTED}; }}
    """

    def __init__(self, testing: bool = False):
        super().__init__()
        self.agents = {}
        self.show_logs = {}
        # Arrival times for agent log lines. StargazerAgent stores log strings
        # without timestamps, so the honest stamp is when this UI first saw the
        # line — recorded once, never re-derived, so old lines don't drift.
        self.log_times = {}
        self.figlet = theme.cosmic_figlet('doom')
        self.testing = testing

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            theme.gradient_banner(self.figlet.renderText('COSMIC CLI')),
            classes="banner",
            markup=True,
        )
        yield DataTable(id="directives")
        with Horizontal(id="input_row"):
            yield Input(
                placeholder="directive… (ctrl+k api key)",
                id="directive_input",
            )
            yield Button("▸ DEPLOY", id="deploy_btn", variant="primary")
        yield RichLog(id="logs", highlight=True, markup=True)
        yield Footer()

    def on_mount(self):
        table = self.query_one("#directives", DataTable)
        table.add_columns("DIRECTIVE", "STATUS", "MODEL", "STEPS", "LOGS")
        table.cursor_type = "row"
        self.sub_title = theme.joined(["stargazer", theme.helix_mark()])
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

    def _stamp_logs(self, directive: str, count: int) -> list:
        """Return arrival stamps for `directive`, minting them for new lines only."""
        stamps = self.log_times.setdefault(directive, [])
        if len(stamps) < count:
            now = datetime.now().strftime("%H:%M:%S")
            stamps.extend([now] * (count - len(stamps)))
        return stamps

    def _refresh_panel(self, agent=None):
        """Safe refresh: clear/rebuild table + write to RichLog (no remove_children hacks on dynamic Statics)."""
        try:
            table = self.query_one("#directives", DataTable)
            table.clear()
            for dir, ag in self.agents.items():
                model = getattr(ag, "model", "N/A")
                status = str(getattr(ag, "status", "ready"))
                bar = theme.step_bar(
                    getattr(ag, "steps_taken", 0),
                    getattr(ag, "max_steps", 20),
                    status,
                )
                count = len(ag.logs)
                logs_cell = (
                    f"[{theme.CYAN}]▾ {count}[/]"
                    if self.show_logs.get(dir)
                    else f"[{theme.MUTED}]{count}[/]"
                )
                table.add_row(
                    dir,
                    Text.from_markup(theme.status_markup(status)),
                    model,
                    Text.from_markup(bar),
                    Text.from_markup(logs_cell),
                    key=dir,
                )

            # Modern logs via RichLog
            log_widget = self.query_one("#logs", RichLog)
            log_widget.clear()
            for dir, show in list(self.show_logs.items()):
                if show and dir in self.agents:
                    ag = self.agents[dir]
                    stamps = self._stamp_logs(dir, len(ag.logs))
                    log_widget.write(theme.log_header(dir))
                    for line, stamp in list(zip(ag.logs, stamps))[-50:]:  # safety cap
                        log_widget.write(theme.agent_log_line(line, stamp))
        except Exception:
            # Guard against refresh during shutdown / testing partial mounts
            pass

    def on_data_table_row_selected(self, event):
        directive = event.data_table.get_row(event.row_key)[0]
        self.toggle_logs(directive)
