"""Operator PAUSE modal. Never receives or renders a token."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from cosmic_cli import theme
from cosmic_cli.pause_authority import PauseHandle
from cosmic_cli.tui.format import escape_markup, trunc


class PauseApproveScreen(ModalScreen[str]):
    """Returns 'approved', 'declined', or None. Token never shown."""

    BINDINGS = [
        ("escape", "cancel_pause", "Cancel"),
        ("y", "approve_pause", "Approve"),
        ("n", "decline_pause", "Decline"),
    ]

    DEFAULT_CSS = f"""
    PauseApproveScreen {{
        align: center middle;
        background: #04060a 45%;
    }}
    PauseApproveScreen > #pause_dialog {{
        width: 64;
        height: auto;
        padding: 1 2;
        background: {theme.SURFACE};
        border: round {theme.WARN};
    }}
    PauseApproveScreen #pause_title {{
        color: {theme.WARN};
        text-style: bold;
        margin-bottom: 1;
    }}
    PauseApproveScreen #pause_summary, PauseApproveScreen #pause_rule {{
        color: {theme.TEXT};
        margin-bottom: 1;
    }}
    PauseApproveScreen #pause_hint {{
        color: {theme.MUTED};
        margin-bottom: 1;
    }}
    PauseApproveScreen #pause_buttons {{
        height: auto;
        margin-top: 1;
    }}
    """

    def __init__(self, handle: PauseHandle) -> None:
        super().__init__()
        self.handle = handle

    def compose(self) -> ComposeResult:
        summary = escape_markup(trunc(self.handle.action_summary or "(action)", 80))
        rule = escape_markup(trunc(self.handle.rule_id or self.handle.reason or "—", 60))
        with Vertical(id="pause_dialog"):
            yield Static("⏸ PAUSE — operator approval", id="pause_title")
            yield Static(summary, id="pause_summary")
            yield Static(f"rule: {rule}", id="pause_rule")
            yield Static(
                "token never shown to the model · TTY L2 only"
                + ("\nHelix decline unavailable; Esc leaves the gate pending."
                   if self.handle.channel == "helix" else ""),
                id="pause_hint",
            )
            with Horizontal(id="pause_buttons"):
                yield Button("y APPROVE", variant="primary", id="pause_approve")
                yield Button("n DECLINE", variant="default", id="pause_decline",
                             disabled=self.handle.channel == "helix")

    def action_cancel_pause(self) -> None:
        self.dismiss(None)

    def action_approve_pause(self) -> None:
        self.dismiss("approved")

    def action_decline_pause(self) -> None:
        if self.handle.channel == "helix":
            return
        self.dismiss("declined")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pause_approve":
            self.dismiss("approved")
        elif event.button.id == "pause_decline":
            self.action_decline_pause()
