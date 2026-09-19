"""Collapsible mutation peek. Starts hidden."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import RichLog, Static


class DiffPeek(Vertical):
    def compose(self):
        yield Static("DIFF", id="diff_header", classes="section-label")
        yield RichLog(id="diff_body", markup=True, wrap=True, min_width=20)
