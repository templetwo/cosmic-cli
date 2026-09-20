"""Left rail: mission table keyed by mission_id."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import DataTable, Static


class MissionRail(Vertical):
    def compose(self):
        yield Static("MISSIONS", classes="section-label")
        yield DataTable(id="mission_table")
