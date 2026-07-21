"""The 1d skin, as a system.

Single source of truth for the Cosmic CLI surface: palette, banner gradient,
status glyphs, step bars and log-line format. `ui.py` (Textual) and `main.py`
(Rich) both import from here so the TUI, the help screen and the chat
transcript cannot drift apart.

Colors are truecolor hexes; terminals without truecolor degrade to the nearest
ANSI-256 entry, which the palette was picked to survive.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Optional

# --- palette -----------------------------------------------------------------
# Surfaces
PAGE = "#0b0e14"        # app background
SURFACE = "#11151d"     # header, footer, input
PANEL = "#0d1118"       # log panel
BORDER = "#232b3b"      # widget borders
RULE = "#1c2230"        # hairline separators
CURSOR = "#152238"      # selected row

# Text
TEXT = "#c3c2b7"        # body
MUTED = "#898781"       # labels, secondary
FAINT = "#4c5568"       # timestamps, hints
SPECK = "#3a4356"       # star specks, empty bar segments
BRIGHT = "#ffffff"      # emphasis inside a line

# Accents
CYAN = "#56c8ff"        # primary — the brand
BLUE = "#3987e5"        # actions, links
MAGENTA = "#cf8de8"     # the model's voice

# Semantic
GOOD = "#16bd16"
WARN = "#fab219"
SERIOUS = "#ec835a"
CRIT = "#d03b3b"

# --- banner ------------------------------------------------------------------
# Cyan → magenta, applied top-to-bottom across the figlet rows.
GRADIENT = ["#56c8ff", "#6dbcfa", "#8bacf7", "#a89cf3", "#bd92ee", "#cf8de8"]

SPECKS = "·      ✦        ·          ˚         ✧      ·         ˚      ✦   ·"

# --- status ------------------------------------------------------------------
# Keys are the statuses StargazerAgent actually sets (agents.py).
STATUS_STYLE: dict[str, tuple[str, str]] = {
    "ready": ("○", MUTED),
    "running": ("●", CYAN),
    "complete": ("●", GOOD),
    "passed": ("●", GOOD),
    "blocked": ("●", CRIT),
    "error": ("●", SERIOUS),
    "max_steps": ("●", WARN),
}

BAR_SLOTS = 20  # display width of a step bar, regardless of max_steps


def status_color(status: str) -> str:
    """Accent color for an agent status. Unknown statuses read as muted."""
    return STATUS_STYLE.get(str(status), ("●", MUTED))[1]


def status_markup(status: str) -> str:
    """`● running` etc. as Rich markup."""
    glyph, color = STATUS_STYLE.get(str(status), ("●", MUTED))
    return f"[{color}]{glyph} {status}[/]"


def _as_int(value, default: int) -> int:
    """Coerce to int, tolerating None and non-numeric stand-ins (e.g. mocks)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def step_bar(taken: int, max_steps: int, status: str = "running") -> str:
    """`▰▰▰▱▱▱ 7/20` as Rich markup, colored by status.

    The bar is always BAR_SLOTS wide so rows stay aligned no matter what
    --max-steps a directive was deployed with.
    """
    taken = max(0, _as_int(taken, 0))
    max_steps = max(1, _as_int(max_steps, 20))
    taken = min(taken, max_steps)
    filled = round(BAR_SLOTS * taken / max_steps)
    filled = min(BAR_SLOTS, max(0, filled))
    color = status_color(status)
    return (
        f"[{color}]{'▰' * filled}[/][{SPECK}]{'▱' * (BAR_SLOTS - filled)}[/]"
        f" [{MUTED}]{taken}/{max_steps}[/]"
    )


def gradient_banner(text: str) -> str:
    """Color figlet art line-by-line down the cyan→magenta gradient.

    Returns Rich markup. Blank lines are dropped so the gradient spreads across
    the art itself rather than being wasted on padding.
    """
    lines = [ln for ln in text.rstrip("\n").split("\n") if ln.strip()]
    if not lines:
        return ""
    # Pad every row to the same width. The banner is centered as a block, and
    # centering ragged rows individually would shear the letterforms apart.
    width = max(len(ln) for ln in lines)
    out = []
    for i, line in enumerate(lines):
        # Spread the gradient across however many rows the font produced.
        idx = 0 if len(lines) == 1 else round(i * (len(GRADIENT) - 1) / (len(lines) - 1))
        out.append(f"[{GRADIENT[idx]}]{_escape(line.ljust(width))}[/]")
    out.append(f"[{SPECK}]{_specks(width)}[/]")
    return "\n".join(out)


def cosmic_figlet(font: str = "doom"):
    """A Figlet tuned for the banner.

    pyfiglet's default smushing for `doom` merges the vertical rules between
    letters, which reads as mush at terminal sizes. smushMode 0 keeps kerning
    without the merge — the separated letterforms the design calls for. Guarded
    because the attribute is a pyfiglet internal.
    """
    from pyfiglet import Figlet

    figlet = Figlet(font=font)
    try:
        figlet.Font.smushMode = 0
    except Exception:
        pass
    return figlet


def _specks(width: int) -> str:
    """The star-speck line, cropped or padded to exactly `width`.

    Kept flush with the art so the banner is one rectangle — a wider speck row
    would shift the whole block once the Static centers it.
    """
    if len(SPECKS) >= width:
        start = (len(SPECKS) - width) // 2
        return SPECKS[start:start + width]
    return SPECKS.center(width)


def log_line(message: str, step: Optional[int] = None, max_steps: Optional[int] = None,
             when: Optional[datetime] = None) -> str:
    """`21:13:02 → 10/20 read gateway.py` as Rich markup.

    Agent logs already arrive shaped as `→ 7/20 <head>`; when they do, pass the
    raw line and it is re-colored rather than double-prefixed.
    """
    stamp = (when or datetime.now()).strftime("%H:%M:%S")
    body = str(message)
    prefix = ""
    if step is not None and max_steps is not None:
        prefix = f"[{CYAN}]→ {step}/{max_steps}[/] "
    return f"[{FAINT}]{stamp}[/] {prefix}{body}"


_STEP_PREFIX = re.compile(r"^(→ \d+/\d+)\s+(.*)$", re.DOTALL)


def agent_log_line(raw: str, stamp: str) -> str:
    """Re-color one line of StargazerAgent output for the log panel.

    Agent lines already arrive shaped as `→ 7/20 <head>`; that prefix is lifted
    into cyan rather than duplicated. `stamp` is the time the UI first saw the
    line — the agent does not record per-line times, so this is what we honestly
    have. Everything else is escaped so agent output can't inject markup.
    """
    body = str(raw)
    match = _STEP_PREFIX.match(body)
    if match:
        arrow, rest = match.group(1), match.group(2)
        return f"[{FAINT}]{stamp}[/] [{CYAN}]{_escape(arrow)}[/] {_escape(rest)}"
    if body.lstrip().startswith("✔"):
        return f"[{FAINT}]{stamp}[/] [{GOOD}]{_escape(body)}[/]"
    if body.lstrip().startswith(("[Error]", "⚠")):
        return f"[{FAINT}]{stamp}[/] [{SERIOUS}]{_escape(body)}[/]"
    return f"[{FAINT}]{stamp}[/] {_escape(body)}"


def log_header(directive: str, width: int = 56) -> str:
    """`── stargazer · refactor gateway retries ──────` as Rich markup."""
    label = f"── stargazer · {directive} "
    pad = max(0, width - len(label))
    return f"[{MAGENTA}]{_escape(label)}{'─' * pad}[/]"


def _escape(text: str) -> str:
    """Protect figlet/user text from being read as Rich markup."""
    return text.replace("[", "\\[")


def helix_mark() -> str:
    """`helix ✓` when the memory substrate resolves, `helix ·` when it doesn't."""
    try:
        from cosmic_cli.helix_bridge import resolve_t2helix_root

        return "helix ✓" if resolve_t2helix_root() else "helix ·"
    except Exception:
        return "helix ·"


def rule(width: int = 57) -> str:
    return f"[{SPECK}]{'─' * width}[/]"


def joined(parts: Iterable[str], sep: str = " · ") -> str:
    return sep.join(p for p in parts if p)
