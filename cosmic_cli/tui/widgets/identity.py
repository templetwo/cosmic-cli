"""One-line identity. Health colors come from live flags, not sample copy."""

from __future__ import annotations

from textual.widgets import Static


class IdentityBar(Static):
    def __init__(self, content: str = "") -> None:
        super().__init__(content, id="identity", markup=True)
