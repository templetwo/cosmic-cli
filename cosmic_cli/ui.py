import os

from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from cosmic_cli import theme
from cosmic_cli.agents import StargazerAgent
from cosmic_cli.tui.app import PilotApp


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

    def compose(self):
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


DirectivesUI = PilotApp
