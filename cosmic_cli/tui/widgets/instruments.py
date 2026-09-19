"""Right rail: compass counts, pending list, session meta."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Static


class PendingList(Static):
    can_focus = True


class InstrumentStack(Vertical):
    def compose(self):
        yield Static("INSTRUMENTS", classes="section-label")
        yield Static("", id="compass_pulse", markup=True)
        yield PendingList("", id="pending", markup=True)
        yield Static("", id="session_meta", markup=True)
