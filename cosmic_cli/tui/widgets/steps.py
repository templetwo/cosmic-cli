"""Center column: selected-mission step tape."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import RichLog, Static


class StepColumn(Vertical):
    def compose(self):
        yield Static("", id="step_header")
        yield RichLog(
            id="step_tape",
            markup=True,
            highlight=False,
            wrap=True,
            min_width=16,
        )
