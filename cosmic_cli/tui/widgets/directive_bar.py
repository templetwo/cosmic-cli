"""Directive input. ctrl+k is the API-key action, not delete_right_all."""

from __future__ import annotations

from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Input


class DirectiveInput(Input):
    """Input that yields ctrl+k to the app instead of deleting to end-of-line."""

    BINDINGS = [
        Binding("ctrl+k", "prompt_api_key", "api key", show=False),
    ]

    def action_prompt_api_key(self) -> None:
        self.app.action_prompt_api_key()


class DirectiveBar(Horizontal):
    def compose(self):
        yield DirectiveInput(
            placeholder="directive… (ctrl+k api key)",
            id="directive_input",
        )
        yield Button("▸ DEPLOY", id="deploy_btn", variant="primary")
