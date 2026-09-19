"""Pilot Board paint. Dynamic strings are escaped; health colors are data-driven."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from cosmic_cli import theme
from cosmic_cli.events import COMPASS_CLASSES
from cosmic_cli.tui.state import PendingPause, StepEvent

_COMPASS_COLOR = {
    "OPEN": theme.GOOD,
    "PAUSE": theme.WARN,
    "WITNESS": theme.CRIT,
}


def escape_markup(text: object) -> str:
    return str(text).replace("[", "\\[")


def trunc(text: object, width: int) -> str:
    raw = str(text or "").replace("\n", " ").replace("\r", " ")
    if width <= 0 or len(raw) <= width:
        return raw
    if width == 1:
        return "…"
    return raw[: width - 1] + "…"


def hhmmss(ts: str) -> str:
    if "T" in ts:
        clock = ts.split("T", 1)[1]
        if len(clock) >= 8 and clock[2] == ":" and clock[5] == ":":
            return clock[:8]
    if len(ts) >= 8 and ts[2] == ":":
        return ts[:8]
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def identity_line(
    *,
    version: str,
    commit: str = "",
    model: str = "",
    helix_on: bool,
    floor_ok: Optional[bool],
    goal: Optional[str] = None,
) -> str:
    ver = escape_markup(version)
    commit_s = escape_markup(commit) if commit else ""
    model_s = escape_markup(model) if model else "—"
    helix = (
        f"[{theme.GOOD}]helix:on[/]"
        if helix_on
        else f"[{theme.MUTED}]helix:off[/]"
    )
    if floor_ok is True:
        floor = f"[{theme.GOOD}]floor:ok[/]"
    elif floor_ok is False:
        floor = f"[{theme.CRIT}]floor:fail[/]"
    else:
        floor = f"[{theme.MUTED}]floor:unknown[/]"
    bits = [
        f"[{theme.CYAN} bold]✦ COSMIC[/] [{theme.TEXT}]{ver}[/]",
    ]
    if commit_s:
        bits.append(f"[{theme.FAINT}]{commit_s}[/]")
    bits.append(f"[{theme.TEXT}]{model_s}[/]")
    bits.append(helix)
    bits.append(floor)
    if goal:
        bits.append(
            f"[{theme.MUTED}]goal:[/] [{theme.TEXT}]{escape_markup(trunc(goal, 40))}[/]"
        )
    return theme.joined(bits)


def step_line(step: StepEvent) -> str:
    stamp = escape_markup(hhmmss(step.ts))
    compass = step.compass
    color = _COMPASS_COLOR.get(str(compass), theme.CYAN)
    glyph = "⏸" if compass == "PAUSE" else "●"
    kind = escape_markup(trunc(step.kind, 10))
    summary = escape_markup(trunc(step.summary, 96))
    return (
        f"[{theme.FAINT}]{stamp}[/] [{color}]{glyph}[/] "
        f"[{theme.BRIGHT}]{kind}[/] [{theme.TEXT}]{summary}[/]"
    )


def compass_line(today: dict) -> str:
    parts = []
    for name in COMPASS_CLASSES:
        n = today.get(name, 0)
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 0
        color = _COMPASS_COLOR.get(name, theme.MUTED)
        parts.append(f"[{color}]{name} {n}[/]")
    return "  ".join(parts)


def pending_line(pauses: list[PendingPause]) -> str:
    if not pauses:
        return f"[{theme.MUTED}]PENDING[/]\n[{theme.FAINT}](none)[/]"
    rows = [f"[{theme.WARN}]PENDING {len(pauses)}[/]"]
    for pause in pauses[-8:]:
        summary = escape_markup(trunc(pause.action_summary or "pause", 34))
        mission = escape_markup(trunc(pause.mission_key or "", 16))
        rule = escape_markup(trunc(pause.rule or "", 18))
        extra = f" [{theme.FAINT}]{rule}[/]" if rule else ""
        rows.append(f"[{theme.WARN}]⏸[/] {summary} [{theme.FAINT}]{mission}[/]{extra}")
    return "\n".join(rows)


def session_meta_line(
    *,
    cwd: Optional[str],
    verify_cmd: Optional[str],
    mode: Optional[str],
    session: Optional[str] = None,
) -> str:
    cwd_s = escape_markup(trunc(cwd or "—", 42))
    verify = escape_markup(trunc(verify_cmd or "—", 42))
    mode_s = escape_markup(mode or "—")
    session_s = escape_markup(trunc(session or "—", 28))
    return (
        f"[{theme.MUTED}]SESSION[/]\n"
        f"[{theme.FAINT}]cwd[/] [{theme.TEXT}]{cwd_s}[/]\n"
        f"[{theme.FAINT}]verify[/] [{theme.TEXT}]{verify}[/]\n"
        f"[{theme.FAINT}]mode[/] [{theme.TEXT}]{mode_s}[/]\n"
        f"[{theme.FAINT}]session[/] [{theme.TEXT}]{session_s}[/]"
    )


def diff_body(path: Optional[str], diff: Optional[str], checkpoint: Optional[str]) -> str:
    if not path and not diff:
        return f"[{theme.FAINT}]no mutation on the selected mission[/]"
    bits = []
    if path:
        bits.append(f"[{theme.MUTED}]path[/] [{theme.TEXT}]{escape_markup(path)}[/]")
    if checkpoint:
        bits.append(
            f"[{theme.MUTED}]checkpoint[/] [{theme.TEXT}]{escape_markup(checkpoint)}[/]"
        )
    if diff:
        bits.append(escape_markup(trunc(diff, 4000)))
    return "\n".join(bits)
