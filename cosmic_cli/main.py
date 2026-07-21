"""Cosmic CLI entrypoint — Grok chat + Stargazer agent."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from click.core import Command, Group
from rich.console import Console
from rich.panel import Panel
from xai_sdk import Client
from xai_sdk.chat import assistant, system, user

from cosmic_cli.agents import DEFAULT_MODEL, SESSION_DIR, StargazerAgent
from cosmic_cli import helix_bridge
from cosmic_cli.init_cmd import write_init
from cosmic_cli.policy import ActionType, Disposition, evaluate_rules
from cosmic_cli.principles import DIFFERENTIATORS
from cosmic_cli.review import (
    build_bundle,
    latest_session_id,
    run_review,
)
from cosmic_cli.rules import load_rules_from_markdown
from cosmic_cli.shell_guard import check_shell
from cosmic_cli import theme
from cosmic_cli.ui import DirectivesUI

# Quiet noisy library logs by default; --verbose re-enables agent INFO.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("cosmic_cli").setLevel(logging.WARNING)


def load_manual_env() -> None:
    """Load .env from multiple locations (later paths win).

    Order: package root → ~/.cosmic-cli/.env → cwd/.env
    So project-local overrides home, home overrides install defaults.
    Auth keys from file always override a stale shell export.
    """
    package_root = Path(__file__).resolve().parent.parent
    possible_paths = [
        package_root / ".env",
        Path.home() / ".cosmic-cli" / ".env",
        Path.cwd() / ".env",
    ]
    override_keys = {
        "XAI_API_KEY",
        "GROK_API_KEY",
        "COSMIC_GROK_MODEL",
    }
    # Later paths win, but warn if cwd .env overrides auth (audit M1).
    cwd_env = Path.cwd() / ".env"
    for dotenv_path in possible_paths:
        if not dotenv_path.exists():
            continue
        try:
            with open(dotenv_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key, value = key.strip(), value.strip()
                        if key in override_keys:
                            prev = os.environ.get(key)
                            os.environ[key] = value
                            if (
                                prev
                                and prev != value
                                and dotenv_path.resolve() == cwd_env.resolve()
                                and key == "XAI_API_KEY"
                            ):
                                # deferred print — console may not exist yet
                                os.environ["_COSMIC_ENV_OVERRIDE_WARN"] = (
                                    f"{key} overridden by {dotenv_path}"
                                )
                        else:
                            os.environ.setdefault(key, value)
        except OSError:
            pass


load_manual_env()

console = Console(stderr=False)
if os.environ.pop("_COSMIC_ENV_OVERRIDE_WARN", None):
    # Printed once at import when cwd .env overrode auth keys
    pass  # avoid noise on every import in tests; doctor surfaces helix/env

COSMIC_QUOTES = [
    "From the edge of the universe:",
    "The stars whisper from the void:",
    "A cosmic thought emerges:",
    "The stars align for this response:",
    "Behold, a message from the cosmos:",
]

MEMORY_FILE = Path.home() / ".cosmic_cli_memory.json"


def get_api_key(*, prompt: bool = True) -> Optional[str]:
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if api_key:
        return api_key
    if not prompt:
        return None
    api_key = click.prompt("Enter your xAI API key", type=str, hide_input=True)
    if api_key:
        os.environ["XAI_API_KEY"] = api_key
        return api_key
    console.print("[bold red]API key required.[/bold red]")
    return None


def set_verbose(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.getLogger("cosmic_cli").setLevel(level)
    logging.getLogger("cosmic_cli.agents").setLevel(level)


class CosmicCLI:
    def __init__(self) -> None:
        self.chat_instance: Optional[Any] = None
        self.client_instance: Optional[Any] = None

    def save_memory(self, messages: List[Any]) -> None:
        try:
            serializable: List[Dict[str, str]] = []
            for msg in messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    content = (
                        msg.content
                        if isinstance(msg.content, str)
                        else "<non-text>"
                    )
                    serializable.append({"role": msg.role, "content": content})
                elif isinstance(msg, dict) and "role" in msg:
                    serializable.append(
                        {"role": msg["role"], "content": str(msg.get("content", ""))}
                    )
            MEMORY_FILE.write_text(
                json.dumps(serializable, indent=2), encoding="utf-8"
            )
        except OSError as e:
            console.print(f"[red]Error saving history: {e}[/red]")

    def load_memory(self) -> List[Dict[str, str]]:
        if not MEMORY_FILE.exists():
            return []
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def append_to_memory(self, role: str, content: str) -> None:
        history = self.load_memory()
        history.append({"role": role, "content": content})
        self.save_memory(history)

    def initialize_chat(
        self, load_history: bool = True, model: str = DEFAULT_MODEL
    ) -> Optional[Any]:
        api_key = get_api_key()
        if not api_key:
            return None
        self.client_instance = Client(api_key=api_key)
        self.chat_instance = self.client_instance.chat.create(model=model)
        self.chat_instance.append(
            system(
                "You are Cosmic CLI, a concise terminal companion powered by Grok. "
                "Be accurate first. Light cosmic tone is optional."
            )
        )
        if load_history:
            for msg in self.load_memory():
                role, content = msg.get("role"), msg.get("content", "")
                if role == "user":
                    self.chat_instance.append(user(content))
                elif role == "assistant":
                    self.chat_instance.append(assistant(content))
        return self.chat_instance


cosmic_cli = CosmicCLI()


def _cosmic_version() -> str:
    """Installed distribution version, falling back to the package constant."""
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("cosmic-cli")
    except Exception:
        try:
            from cosmic_cli import __version__

            return __version__
        except Exception:
            return "dev"


# Help is grouped by what you are trying to do, not by alphabet. Any command
# not named here still shows up, under MORE — the map curates order, it never
# hides a command.
COMMAND_GROUPS: List[tuple] = [
    ("MISSIONS", ["do", "tui", "review", "stargazer", "workflow"]),
    ("TALK", ["chat", "ask", "analyze"]),
    ("GROUND CONTROL", ["doctor", "dashboard", "helix", "sessions", "init", "run", "gate"]),
]


def _lockup() -> str:
    """The one-line brand lockup that replaces the figlet panel on --help."""
    title = "C O S M I C  C L I"
    tagline = "Grok agent for the terminal"
    meta = theme.joined(
        [f"v{_cosmic_version()}", f"default model {DEFAULT_MODEL}", theme.helix_mark()]
    )
    top_fill = max(3, 66 - len(title) - len(tagline) - 9)
    bot_fill = max(3, 66 - len(meta) - 6)
    return (
        f"[{theme.SPECK}]╭─ ✦ [/][{theme.CYAN}]{title}[/]"
        f"[{theme.SPECK}] {'─' * top_fill} [/]{tagline}[{theme.SPECK}] ─╮[/]\n"
        f"[{theme.SPECK}]╰──[/] {meta} [{theme.SPECK}]{'─' * bot_fill}╯[/]"
    )


class CosmicHelpFormatter(click.HelpFormatter):
    # `highlight=False` throughout: Rich's auto-highlighter otherwise recolors
    # version numbers, model names and parenthesised text inside help copy.
    def write_usage(self, prog: str, args: str = "", prefix: Optional[str] = None) -> None:
        console.print(_lockup(), highlight=False)
        console.print()
        line = " ".join(part for part in (prog, args) if part).strip()
        console.print(f"[{theme.MUTED}]USAGE[/]   {line}", highlight=False)

    def write_heading(self, heading: str) -> None:
        console.print(f"\n[{theme.MUTED}]{heading.upper()}[/]", highlight=False)

    def write_text(self, text: str) -> None:
        console.print(text, highlight=False)

    def write_dl(
        self, rows, col_max: int = 30, col_spacing: int = 2
    ) -> None:
        rows = list(rows)
        width = min(col_max, max((len(term) for term, _ in rows), default=0) + 2)
        for term, definition in rows:
            console.print(
                f"  [{theme.CYAN}][b]{term.ljust(width)}[/b][/] {definition}",
                highlight=False,
            )


class CosmicCommand(Command):
    def get_help(self, ctx):
        formatter = CosmicHelpFormatter(width=ctx.terminal_width)
        self.format_help(ctx, formatter)
        return formatter.getvalue().rstrip("\n")


class CosmicGroup(Group):
    command_class = CosmicCommand
    group_class = type

    def get_help(self, ctx):
        formatter = CosmicHelpFormatter(width=ctx.terminal_width)
        self.format_help(ctx, formatter)
        return formatter.getvalue().rstrip("\n")

    def format_commands(self, ctx, formatter) -> None:
        """Emit commands under MISSIONS / TALK / GROUND CONTROL instead of one
        flat `Commands` list. Anything uncategorised lands in MORE so a newly
        added command can never silently vanish from --help."""
        available = {}
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is not None and not getattr(cmd, "hidden", False):
                available[name] = cmd

        for heading, names in COMMAND_GROUPS:
            rows = [
                (name, available.pop(name).get_short_help_str(60))
                for name in names
                if name in available
            ]
            if rows:
                formatter.write_heading(heading)
                formatter.write_dl(rows)

        if available:  # never drop a command we did not think to categorise
            formatter.write_heading("MORE")
            formatter.write_dl(
                [(n, c.get_short_help_str(60)) for n, c in sorted(available.items())]
            )

        console.print(
            f"\n[{theme.FAINT}]try[/]  cosmic-cli do [{theme.TEXT}]'analyze data.txt'[/]"
            f"   [{theme.FAINT}]·[/]   cosmic-cli tui",
            highlight=False,
        )


@click.group(
    cls=CosmicGroup,
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose agent logs")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Cosmic CLI — Grok agent for the terminal. Default model: grok-4.5."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    set_verbose(verbose)
    # The security gate runs as the PreToolUse hook on every tool call and must
    # stay a pure, side-effect-free classifier — no dashboard spawn, no write
    # into the ranking-protected token-store dir on that hot path. Auto-start the
    # dashboard only for human/interactive invocations, never for `gate` (which
    # also covers --boot-canary/--floor-canary/--verb-check as flags).
    if ctx.invoked_subcommand not in _NO_DASHBOARD_SUBCOMMANDS:
        _ensure_dashboard()


DASHBOARD_PORT = int(os.environ.get("COSMIC_DASHBOARD_PORT", "4333"))
# Subcommands that must never trigger dashboard side effects. `gate` is the
# PreToolUse hook (fires on every tool call); its --boot-canary/--floor-canary/
# --verb-check are flags on it, so excluding `gate` covers them too.
_NO_DASHBOARD_SUBCOMMANDS = {"gate"}


def _dashboard_log_path() -> Path:
    """Dashboard logs live OUTSIDE ~/.cosmic-cli — that dir is the ranking-
    protected token store and must not accumulate writes from this path."""
    d = Path.home() / ".cosmic-cli-logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "dashboard.log"


def _dashboard_running(port: int = DASHBOARD_PORT) -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _ensure_dashboard(port: int = DASHBOARD_PORT) -> None:
    """Best-effort: spawn the Mission Control server if it isn't listening.

    Opt out with COSMIC_NO_DASHBOARD=1. Never blocks or fails the CLI.
    """
    if os.environ.get("COSMIC_NO_DASHBOARD") == "1":
        return
    if _dashboard_running(port):
        return
    try:
        log = open(_dashboard_log_path(), "ab")
        subprocess.Popen(
            [sys.executable, "-m", "cosmic_cli.dashboard", str(port)],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    except Exception:
        pass


@cli.command("dashboard")
@click.option("--port", default=DASHBOARD_PORT, show_default=True, help="Dashboard port")
@click.option("--no-open", is_flag=True, help="Don't open the browser")
def dashboard_cmd(port: int, no_open: bool) -> None:
    """Mission Control — live dashboard over the shared chronicle."""
    url = f"http://localhost:{port}"
    if not _dashboard_running(port):
        _ensure_dashboard(port)
        for _ in range(20):
            if _dashboard_running(port):
                break
            time.sleep(0.1)
    status = "running" if _dashboard_running(port) else "failed to start (see ~/.cosmic-cli/dashboard.log)"
    console.print(f"Mission Control · {status} · [link={url}]{url}[/link]")
    if not no_open and _dashboard_running(port):
        subprocess.run(["open", url], check=False)


def _run_stargazer(
    directive: str,
    *,
    mode: str = "safe",
    max_steps: int = 20,
    model: str = DEFAULT_MODEL,
    quiet: bool = False,
    auto_verify: bool = True,
    use_helix: bool = True,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = get_api_key()
    if not api_key:
        sys.exit(2)

    def on_update(msg: str) -> None:
        # dim progress, bright errors
        if msg.startswith("[Error]") or msg.startswith("  · [Error]"):
            console.print(f"[red]{msg}[/red]")
        elif msg.startswith("[BLOCKED]") or "blocked ·" in msg:
            console.print(f"[bold red]{msg}[/bold red]")
        elif msg.startswith("→"):
            console.print(f"[bold cyan]{msg}[/bold cyan]")
        elif msg.startswith("steer"):
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            console.print(f"[dim]{msg}[/dim]")

    helix_ctx = ""
    # Stable seat session for compass approvals (survives re-run after confirm)
    seat_session = session_id or helix_bridge.current_session_id() if use_helix else session_id
    notes = helix_bridge.load_project_notes(Path.cwd())
    if notes:
        helix_ctx = notes + "\n"
        console.print(f"[dim]project notes · loaded from cwd[/dim]")
    if use_helix and helix_bridge.available():
        boot = helix_bridge.boot(directive, top_k=5)
        helix_ctx += helix_bridge.format_boot_context(boot)
        if boot.get("ok"):
            console.print(
                f"[dim]helix · {helix_ctx.splitlines()[0] if helix_ctx else 'booted'}[/dim]"
            )
            # Prefer seat session from boot if we don't have one yet
            if not seat_session:
                inner = boot.get("result") or {}
                if isinstance(inner, dict) and inner.get("sessionId"):
                    seat_session = str(inner["sessionId"])
            helix_bridge.set_goal(
                directive[:500],
                why="cosmic-cli do",
                session_id=seat_session,
            )
            st = helix_bridge.get_state(seat_session)
            state_txt = helix_bridge.format_state_context(st)
            if state_txt:
                helix_ctx += "\n" + state_txt
                console.print(f"[dim]helix · open threads loaded[/dim]")
        else:
            console.print(
                f"[yellow]helix offline:[/yellow] {boot.get('error', '?')[:120]}"
            )
    elif use_helix:
        console.print(
            "[dim]helix unavailable (set T2HELIX_ROOT or install "
            "github.com/templetwo/t2helix)[/dim]"
        )

    console.print(
        f"[dim]{model} · {mode} · ≤{max_steps} steps · cwd={os.getcwd()}"
        f"{f' · seat={seat_session[:12]}…' if seat_session else ''}[/dim]"
    )
    agent = StargazerAgent(
        directive=directive,
        api_key=api_key,
        ui_callback=on_update,
        exec_mode=mode,
        work_dir=os.getcwd(),
        model=model,
        max_steps=max_steps,
        quiet=quiet,
        show_progress=False,  # progress bar fights step logs — keep off
        auto_verify=auto_verify,
        use_helix=use_helix,
        helix_context=helix_ctx,
        session_id=seat_session,
    )
    result = agent.execute()
    status = result.get("status", "?")
    color = {
        "complete": "green",
        "passed": "yellow",
        "max_steps": "yellow",
        "error": "red",
        "blocked": "red",
    }.get(status, "white")
    console.print(
        f"[{color}]{status}[/{color}]  "
        f"[dim]session {result.get('session')} · {result.get('steps_taken', '?')} steps[/dim]"
    )
    if status == "blocked" and result.get("block_message"):
        console.print(Panel(result["block_message"][:2000], title="blocked", border_style="red"))
    # steps_taken may only be on agent; mirror from results len if missing
    edited = list(dict.fromkeys(result.get("edited") or []))
    result["edited"] = edited
    if edited:
        console.print(f"[yellow]edited[/yellow] {', '.join(edited)}")
    for w in result.get("warnings") or []:
        console.print(f"[red]warn[/red] {w}")
    if result.get("results") and status != "blocked":
        last = result["results"][-1]
        body = str(last.get("result", "")).strip()
        lines = [ln for ln in body.splitlines() if ln.strip()]
        body = "\n".join(lines)
        if body:
            console.print(Panel(body[:4000], title="result", border_style=color))
    return result


def _print_review(report: Dict[str, Any]) -> None:
    verdict = str(report.get("verdict", "?")).upper()
    color = {"CLEAR": "green", "WARN": "yellow", "BLOCK": "red"}.get(verdict, "white")
    console.print(f"[{color}]review verdict: {verdict}[/{color}]")
    for f in report.get("findings") or []:
        sev = str(f.get("severity", "NOTE")).upper()
        sc = {"FATAL": "red", "WARN": "yellow", "NOTE": "dim"}.get(sev, "white")
        path = f.get("path") or ""
        title = f.get("title") or ""
        detail = f.get("detail") or ""
        console.print(f"  [{sc}]{sev}[/{sc}] {path} — {title}")
        if detail:
            console.print(f"         {detail[:400]}")
    if report.get("residual_risk"):
        console.print(f"[dim]residual:[/dim] {report['residual_risk']}")
    tests = report.get("suggested_tests") or []
    if tests:
        console.print("[dim]suggested tests:[/dim]")
        for t in tests:
            console.print(f"  · {t}")


@cli.command("do")
@click.argument("directive")
@click.option(
    "--mode",
    type=click.Choice(["safe", "interactive", "full"]),
    default="safe",
)
@click.option("--max-steps", default=20, show_default=True, type=int)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("-q", "--quiet", is_flag=True, help="Less streaming noise")
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Auto py_compile on edited Python before FINISH",
)
@click.option(
    "--review/--no-review",
    default=False,
    help="Run independent review seat on edited files after mission",
)
@click.option(
    "--helix/--no-helix",
    default=True,
    help="Use T2Helix for memory/goal (local SQLite; not Sovereign Stack)",
)
@click.option(
    "--session",
    default=None,
    help=(
        "Stable seat session id for compass approvals (default: Helix "
        ".current_session / CLAUDE_SESSION_ID). Re-use after helix confirm."
    ),
)
@click.pass_context
def do_cmd(
    ctx: click.Context,
    directive: str,
    mode: str,
    max_steps: int,
    model: str,
    quiet: bool,
    verify: bool,
    review: bool,
    helix: bool,
    session: Optional[str],
) -> None:
    """Run Stargazer on a directive (primary daily command)."""
    if ctx.obj and ctx.obj.get("verbose"):
        set_verbose(True)
    result = _run_stargazer(
        directive,
        mode=mode,
        max_steps=max_steps,
        model=model,
        quiet=quiet,
        auto_verify=verify,
        use_helix=helix,
        session_id=session,
    )
    if review and result.get("edited"):
        api_key = get_api_key(prompt=False)
        if api_key:
            console.print("\n[bold]second seat · review[/bold]")
            bundle = build_bundle(
                Path.cwd(),
                session_id=result.get("session"),
                paths=list(result.get("edited") or []),
                directive=directive,
            )
            report = run_review(bundle, api_key=api_key, model=model)
            _print_review(report)
            if str(report.get("verdict", "")).upper() == "BLOCK":
                sys.exit(3)
    # blocked = compass gate handed control up (exit 4 so cockpits can branch)
    if result.get("status") == "blocked":
        sys.exit(4)
    if result.get("status") != "complete":
        sys.exit(1)


@cli.command("review")
@click.option(
    "--session",
    default=None,
    help="Session id (default: latest under ~/.cosmic-cli/sessions)",
)
@click.option(
    "--path",
    "paths",
    multiple=True,
    help="Explicit path(s) to review (repeatable). Overrides session edited list if set.",
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option(
    "--directive",
    default="",
    help="Optional mission directive context for the reviewer",
)
def review_cmd(
    session: Optional[str], paths: tuple, model: str, directive: str
) -> None:
    """Independent second-pass review of diffs (Cold Eye seat)."""
    api_key = get_api_key()
    if not api_key:
        sys.exit(2)
    sid = session or latest_session_id()
    path_list = list(paths) if paths else None
    if not sid and not path_list:
        console.print(
            "[red]No session found and no --path given. Run a mission first.[/red]"
        )
        sys.exit(2)
    console.print(
        f"[dim]review · model={model} · session={sid or '—'} · "
        f"paths={', '.join(path_list or []) or '(from session)'}[/dim]"
    )
    bundle = build_bundle(
        Path.cwd(),
        session_id=sid,
        paths=path_list,
        directive=directive,
    )
    if not bundle.paths and not bundle.diffs:
        console.print("[yellow]Nothing to review (no edited paths / diffs).[/yellow]")
        sys.exit(0)
    console.print(f"[dim]reviewing: {', '.join(bundle.paths)}[/dim]")
    try:
        report = run_review(bundle, api_key=api_key, model=model)
    except Exception as e:
        console.print(f"[red]review failed: {e}[/red]")
        sys.exit(1)
    _print_review(report)
    # persist beside session
    if sid:
        out = SESSION_DIR / f"{sid}.review.json"
        try:
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            console.print(f"[dim]wrote {out}[/dim]")
        except OSError:
            pass
    if str(report.get("verdict", "")).upper() == "BLOCK":
        sys.exit(3)


def _chat_meta(response, elapsed: float) -> str:
    """`412 tokens · 1.8s` — the token count is omitted when the SDK does not
    report one, rather than guessed."""
    tokens = None
    usage = getattr(response, "usage", None)
    if usage is not None:
        for attr in ("completion_tokens", "output_tokens", "total_tokens"):
            value = getattr(usage, attr, None)
            if isinstance(value, int) and value > 0:
                tokens = value
                break
    parts = ([f"{tokens} tokens"] if tokens else []) + [f"{elapsed:.1f}s"]
    return " · ".join(parts)


@cli.command("chat")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def chat_cmd(model: str) -> None:
    """Interactive chat with conversation memory."""
    chat_instance = cosmic_cli.initialize_chat(load_history=True, model=model)
    if chat_instance is None:
        return

    def _header(instance) -> None:
        loaded = len(getattr(instance, "messages", []) or [])
        console.print(
            f"[{theme.CYAN}][b]✦ cosmic chat[/b][/]"
            f"[{theme.MUTED}] · {model} · history loaded ({loaded} msgs)[/]",
            highlight=False,
        )
        console.print(theme.rule(), highlight=False)

    _header(chat_instance)
    console.print(
        f"[{theme.FAINT}]reset clears history · quit to leave[/]", highlight=False
    )
    while True:
        user_input = click.prompt(
            click.style("◉ you", fg=(255, 255, 255), bold=True),
            type=str,
            prompt_suffix="   ",
        )
        if user_input.lower() in ("quit", "exit"):
            cosmic_cli.save_memory(getattr(chat_instance, "messages", []))
            console.print(f"[{theme.MUTED}]bye[/]", highlight=False)
            break
        if user_input.lower() == "reset":
            chat_instance = cosmic_cli.initialize_chat(load_history=False, model=model)
            cosmic_cli.save_memory([])
            console.print(f"[{theme.MUTED}]history cleared[/]", highlight=False)
            _header(chat_instance)
            continue
        chat_instance.append(user(user_input))
        try:
            started = time.time()
            response = chat_instance.sample()
            elapsed = time.time() - started
            prefix = random.choice(COSMIC_QUOTES)
            console.print()
            # The cosmic quote is the model clearing its throat — dim italic on
            # its own line, so the answer proper starts clean underneath.
            console.print(
                f"[{theme.MAGENTA}][b]✦ grok  [/b][/][{theme.FAINT}][i]{prefix}[/i][/]",
                highlight=False,
            )
            for line in str(response.content).splitlines() or [""]:
                console.print(f"        {line}", highlight=False)
            console.print(f"[{theme.FAINT}]        {_chat_meta(response, elapsed)}[/]",
                          highlight=False)
            console.print(theme.rule(), highlight=False)
            chat_instance.append(assistant(response.content))
            cosmic_cli.save_memory(getattr(chat_instance, "messages", []))
        except Exception as e:
            console.print(f"[{theme.CRIT}]        {e}[/]", highlight=False)
            console.print(theme.rule(), highlight=False)


@cli.command("ask")
@click.argument("prompt")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def ask_cmd(prompt: str, model: str) -> None:
    """One-shot question (no agent loop)."""
    chat_instance = cosmic_cli.initialize_chat(load_history=False, model=model)
    if chat_instance is None:
        return
    try:
        chat_instance.append(user(prompt))
        response = chat_instance.sample()
        console.print(response.content)
        cosmic_cli.append_to_memory("user", prompt)
        cosmic_cli.append_to_memory("assistant", response.content)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@cli.command("analyze")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def analyze_cmd(file_path: str, model: str) -> None:
    """Analyze a text file with Grok."""
    chat_instance = cosmic_cli.initialize_chat(load_history=False, model=model)
    if chat_instance is None:
        return
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        analysis_prompt = (
            f"Analyze this file ({file_path}). Be concrete.\n\n```\n{content}\n```"
        )
        chat_instance.append(user(analysis_prompt))
        response = chat_instance.sample()
        console.print(response.content)
    except UnicodeDecodeError:
        console.print(f"[red]binary/non-utf8: {file_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@cli.command("run")
@click.argument("command_to_run")
@click.option(
    "--mode",
    type=click.Choice(["safe", "interactive", "full"]),
    default="safe",
    show_default=True,
    help="Same blast-radius modes as `do` (full skips local blocklist).",
)
def run_cmd(command_to_run: str, mode: str) -> None:
    """Run a shell command after local policy + blocklist + confirmation.

    Not a side door around compass: COSMIC.md rules and check_shell apply
    before the human confirm prompt (Claude exercise finding).
    """
    console.print(f"[red]run:[/red] {command_to_run}")

    # 1. Local COSMIC.md policy (when present)
    cosmic_md = Path.cwd() / "COSMIC.md"
    if cosmic_md.is_file() and mode != "full":
        try:
            rules = load_rules_from_markdown(cosmic_md)
            if rules:
                decision = evaluate_rules(rules, ActionType.SHELL, command_to_run)
                if decision.disposition == Disposition.WITNESS:
                    rid = (
                        decision.matches[0].rule.rule_id
                        if decision.matches
                        else "policy"
                    )
                    console.print(
                        f"[red][BLOCKED] local policy WITNESS: {rid}[/red]"
                    )
                    sys.exit(4)
                if decision.disposition == Disposition.PAUSE:
                    console.print(
                        "[yellow][BLOCKED] local policy PAUSE — use "
                        "`cosmic-cli do` for token flow, or --mode full if intentional[/yellow]"
                    )
                    sys.exit(4)
        except Exception as e:
            console.print(f"[red][BLOCKED] policy load failed: {e}[/red]")
            sys.exit(4)

    # 2. check_shell
    blocked = check_shell(command_to_run, exec_mode=mode)
    if blocked:
        console.print(f"[red]{blocked}[/red]")
        sys.exit(4)

    if not click.confirm("Proceed?", default=False):
        console.print("[yellow]cancelled[/yellow]")
        return
    result = subprocess.run(
        command_to_run, shell=True, capture_output=True, text=True, check=False
    )
    if result.stdout:
        console.print(result.stdout, end="")
    if result.stderr:
        console.print(f"[yellow]{result.stderr}[/yellow]", end="")
    console.print(f"[dim]exit {result.returncode}[/dim]")
    sys.exit(result.returncode)


@cli.command("doctor")
def doctor_cmd() -> None:
    """Check API key, model access, and local tooling."""
    console.print(f"[bold]Cosmic doctor[/bold] · default model [cyan]{DEFAULT_MODEL}[/cyan]")
    key = get_api_key(prompt=False)
    if not key:
        console.print("[red]✗[/red] XAI_API_KEY not set (.env or env)")
    else:
        console.print(f"[green]✓[/green] API key present ({key[:7]}… len={len(key)})")
        try:
            import urllib.request

            req = urllib.request.Request(
                "https://api.x.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            ids = sorted(m["id"] for m in data.get("data", []))
            console.print(f"[green]✓[/green] models API OK ({len(ids)} models)")
            if DEFAULT_MODEL in ids:
                console.print(f"[green]✓[/green] default model available: {DEFAULT_MODEL}")
            else:
                console.print(
                    f"[yellow]![/yellow] default {DEFAULT_MODEL} not in list: {ids}"
                )
            # show chat-ish models
            chat_models = [i for i in ids if not i.startswith("grok-imagine")]
            console.print(f"  chat models: {', '.join(chat_models)}")
        except Exception as e:
            console.print(f"[red]✗[/red] models API failed: {e}")

    console.print(f"  cwd: {os.getcwd()}")
    console.print(f"  package: {Path(__file__).resolve().parent.parent}")
    console.print(f"  echo file: {Path.home() / '.cosmic_echo.jsonl'}")
    console.print(f"  sessions: {SESSION_DIR}")
    console.print(f"  chat memory: {MEMORY_FILE}")
    console.print(f"  home config: {Path.home() / '.cosmic-cli' / '.env'}")
    n_sess = len(list(SESSION_DIR.glob('*.jsonl'))) if SESSION_DIR.is_dir() else 0
    console.print(f"  session count: {n_sess}")
    # Helix substrate
    if helix_bridge.available():
        h = helix_bridge.health()
        console.print(
            f"[green]✓[/green] T2Helix @ {helix_bridge.resolve_data_dir()}"
        )
        if h.get("ok") and isinstance(h.get("result"), dict):
            console.print(f"  session: {h['result'].get('session')}")
    else:
        console.print(
            "[yellow]![/yellow] T2Helix not available "
            "(clone github.com/templetwo/t2helix → ~/t2helix)"
        )

    console.print("\n[bold]Daily path[/bold]")
    console.print("  cosmic-cli doctor")
    console.print("  cosmic-cli helix status         # local memory substrate")
    console.print("  cosmic-cli init                 # COSMIC.md in this project")
    console.print("  cosmic-cli do '…'               # boots Helix memory")
    console.print("  cosmic-cli do --review '…'")
    console.print("  cosmic-cli review")
    console.print("  cosmic-cli sessions")


@cli.command("helix")
@click.argument(
    "action",
    type=click.Choice(
        [
            "status",
            "boot",
            "recall",
            "state",
            "thread",
            "record",
            "confirm",
            "pending",
            "show-pause-token",
            "accept-pause",
        ],
        case_sensitive=False,
    ),
    default="status",
)
@click.argument("query", required=False, default="")
@click.option("--domain", default="cosmic-cli", show_default=True)
def helix_cmd(action: str, query: str, domain: str) -> None:
    """T2Helix local memory + compass substrate (shared chronicle with Claude).

    Compass honesty: WITNESS hard-denies. PAUSE soft-denies with a token —
    approve via `helix confirm <token>`, then retry the action. OPEN proceeds.

    Local PAUSE tokens (gate seam) are L2-only (privilege ranking):
      show-pause-token  — display last minted token (interactive TTY only)
      accept-pause      — stage token for one approved retry via the wrapper
                          (writes ~/.cosmic-cli/operator_approval_token)
    L0 agent shells cannot invoke these. No env break-glass (that would be an
    L2 credential L0 can set). Kernel floor: sandbox.toml deny ~/.cosmic-cli.
    See COSMIC.md § Ranking.
    """
    action = action.lower()
    if action in ("show-pause-token", "accept-pause", "confirm"):
        from cosmic_cli.ranking import require_l2_tty

        blocked = require_l2_tty(f"helix {action}")
        if blocked:
            console.print(f"[red]{blocked}[/red]")
            sys.exit(4)  # same exit class as PAUSE/WITNESS block
    if action == "show-pause-token":
        tok_path = Path.home() / ".cosmic-cli" / "last_pause_token.json"
        if not tok_path.is_file():
            console.print("[yellow]no local PAUSE token on file[/yellow]")
            sys.exit(1)
        try:
            data = json.loads(tok_path.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[red]cannot read token file: {e}[/red]")
            sys.exit(1)
        tok = data.get("token") or ""
        console.print(f"[dim]channel[/dim] {data.get('channel')}")
        console.print(f"[dim]minted[/dim]  {data.get('minted_at')}")
        console.print(f"[green]token[/green]   {tok}")
        console.print(
            f"[dim]export COSMIC_APPROVAL_TOKEN={tok}[/dim]\n"
            f"[dim]# or: cosmic-cli helix accept-pause   # L2 TTY file channel[/dim]\n"
            f"[dim]# or: cosmic-cli helix confirm {tok}[/dim]"
        )
        return
    if action == "accept-pause":
        # Stage L2 operator approval for one gate retry (file channel).
        src = Path.home() / ".cosmic-cli" / "last_pause_token.json"
        dst = Path.home() / ".cosmic-cli" / "operator_approval_token"
        if not src.is_file():
            console.print("[yellow]no last_pause_token.json — nothing to accept[/yellow]")
            sys.exit(1)
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            tok = (data.get("token") or "").strip()
            if not tok:
                console.print("[red]empty token in last_pause_token.json[/red]")
                sys.exit(1)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(tok + "\n", encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(dst)
            os.chmod(dst, 0o600)
        except Exception as e:
            console.print(f"[red]accept-pause failed: {e}[/red]")
            sys.exit(1)
        console.print(
            f"[green]staged[/green] operator_approval_token for one retry "
            f"(channel={data.get('channel')}). Re-run the blocked action."
        )
        return
    if action == "status":
        console.print(f"[dim]T2HELIX_ROOT[/dim] {helix_bridge.resolve_t2helix_root()}")
        console.print(f"[dim]T2HELIX_DATA_DIR[/dim] {helix_bridge.resolve_data_dir()}")
        console.print(f"[dim]available[/dim] {helix_bridge.available()}")
        console.print(
            "[dim]compass[/dim] WITNESS=block · PAUSE=token gate · OPEN=allow "
            "(PAUSE enforced in Cosmic + grok-adapter as of 2026-07-14)"
        )
        h = helix_bridge.health()
        console.print(h)
        return
    if action == "boot":
        r = helix_bridge.boot(query or "current context")
        console.print(helix_bridge.format_boot_context(r) if r.get("ok") else r)
        return
    if action == "recall":
        if not query:
            console.print("[red]usage: cosmic-cli helix recall <query>[/red]")
            sys.exit(2)
        r = helix_bridge.recall(query)
        console.print(r)
        return
    if action == "state":
        r = helix_bridge.get_state()
        console.print(r)
        txt = helix_bridge.format_state_context(r)
        if txt:
            console.print(Panel(txt, title="state", border_style="cyan"))
        return
    if action == "thread":
        if not query:
            console.print(
                "[red]usage: cosmic-cli helix thread 'unresolved question'[/red]"
            )
            sys.exit(2)
        r = helix_bridge.open_thread(query, domain=domain, context="cosmic-cli helix")
        console.print(r)
        return
    if action == "record":
        if not query:
            console.print("[red]usage: cosmic-cli helix record 'insight text'[/red]")
            sys.exit(2)
        r = helix_bridge.record(query, domain=domain)
        console.print(r)
        return
    if action == "confirm":
        if not query:
            console.print("[red]usage: cosmic-cli helix confirm <token>[/red]")
            sys.exit(2)
        r = helix_bridge.confirm_pending(query.strip())
        console.print(r)
        if r.get("ok") and (r.get("result") or {}).get("ok"):
            console.print(
                "[green]approved[/green] — re-run the blocked SHELL/action once "
                "(token is single-use)."
            )
        return
    if action == "pending":
        r = helix_bridge.list_pending(limit=20)
        console.print(r)
        return


@cli.command("init")
@click.argument("directory", required=False, default=".")
@click.option("--force", is_flag=True, help="Overwrite existing COSMIC.md")
@click.option("--name", default="COSMIC.md", show_default=True)
@click.option(
    "--grok",
    is_flag=True,
    help="Install global Grok Build PreToolUse wrapper + boot canary assets (box 2).",
)
def init_cmd(directory: str, force: bool, name: str, grok: bool) -> None:
    """Drop COSMIC.md agent notes into a project (no stack).

    With ``--grok``: install the always-trusted global PreToolUse hook that
    calls ``cosmic-cli gate --hook grok`` under the positive COSMIC-ALLOW protocol.
    """
    msg, ok = write_init(Path(directory), force=force, name=name)
    console.print(f"[{'green' if ok else 'red'}]{msg}[/]")
    if not ok:
        sys.exit(1)
    if grok:
        from cosmic_cli.install_grok_hooks import install_grok_hooks

        imsg, iok = install_grok_hooks(force=force)
        console.print(f"[{'green' if iok else 'red'}]{imsg}[/]")
        if not iok:
            sys.exit(1)


@cli.command("sessions")
@click.option("-n", "--limit", default=10, show_default=True, type=int)
def sessions_cmd(limit: int) -> None:
    """List recent mission sessions."""
    if not SESSION_DIR.is_dir():
        console.print("[dim]No sessions yet.[/dim]")
        return
    files = sorted(SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        console.print("[dim]No sessions yet.[/dim]")
        return
    for path in files[:limit]:
        directive = ""
        status = "?"
        edited: list = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev.get("event") == "start":
                    directive = (ev.get("directive") or "")[:60]
                if ev.get("event") == "end":
                    status = ev.get("status") or status
                    edited = ev.get("edited") or edited
        except (OSError, json.JSONDecodeError):
            pass
        rev_path = SESSION_DIR / f"{path.stem}.review.json"
        rev = " · reviewed" if rev_path.exists() else ""
        ed = f" · edited {','.join(edited)}" if edited else ""
        console.print(
            f"[cyan]{path.stem}[/cyan]  [{status}]{ed}{rev}\n"
            f"  [dim]{directive or '(no directive)'}[/dim]"
        )


@click.group()
def stargazer() -> None:
    """Stargazer multi-step agents."""
    pass


@stargazer.command("deploy")
@click.argument("directive")
@click.option(
    "--mode",
    type=click.Choice(["safe", "interactive", "full"]),
    default="safe",
)
@click.option("--max-steps", default=20, show_default=True, type=int)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("-q", "--quiet", is_flag=True)
@click.option("--verify/--no-verify", default=True)
def deploy(
    directive: str, mode: str, max_steps: int, model: str, quiet: bool, verify: bool
) -> None:
    """Deploy Grok Stargazer (same as `cosmic-cli do`)."""
    result = _run_stargazer(
        directive,
        mode=mode,
        max_steps=max_steps,
        model=model,
        quiet=quiet,
        auto_verify=verify,
    )
    if result.get("status") != "complete":
        sys.exit(1)


cli.add_command(stargazer)


@cli.command()
def tui() -> None:
    """Launch Textual TUI."""
    DirectivesUI().run()


@cli.command()
@click.argument("tasks", nargs=-1)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def workflow(tasks, model: str) -> None:
    """Legacy task pairs: analyze <file> [hack <mode>] …"""
    if not tasks:
        console.print("[red]usage: cosmic-cli workflow analyze <file> …[/red]")
        sys.exit(2)
    parts = []
    i = 0
    while i < len(tasks):
        task = tasks[i]
        arg = tasks[i + 1] if i + 1 < len(tasks) else ""
        if task == "analyze":
            parts.append(f"Analyze file {arg}")
            i += 2
        elif task == "hack":
            parts.append(f"Generate a terminal hack in mode {arg}")
            i += 2
        else:
            console.print(f"[red]unknown task: {task}[/red]")
            sys.exit(2)
    result = _run_stargazer(" and ".join(parts) + ".", model=model)
    if result.get("status") != "complete":
        sys.exit(1)


@cli.command("gate")
@click.option(
    "--hook",
    type=click.Choice(["grok", "claude"]),
    default="grok",
    show_default=True,
    help="Cockpit hook flavor (PreToolUse envelope on stdin).",
)
@click.option("--verb-check", is_flag=True, help="Existence probe for the wrapper; exit 0.")
@click.option(
    "--boot-canary",
    is_flag=True,
    help="Box 3: deny+allow canaries against the real gate; exit 1 on fail.",
)
@click.option(
    "--floor-canary",
    is_flag=True,
    help="Temple floor: sandbox.toml + default profile + live Seatbelt deny; exit 1 on fail.",
)
@click.option(
    "--mode",
    type=click.Choice(["safe", "interactive", "full"]),
    default="safe",
    show_default=True,
)
def gate_cmd(
    hook: str, verb_check: bool, boot_canary: bool, floor_canary: bool, mode: str
) -> None:
    """COSMIC-ALLOW sentinel gate (RFC v1.1).

    Reads a cockpit PreToolUse envelope on stdin and emits
    `COSMIC-ALLOW v1 <nonce>` on stdout ONLY for a genuine OPEN decision. Deny is
    signaled by an empty stdout; all diagnostics go to stderr. The nonce comes
    from COSMIC_GATE_NONCE only, never the payload.

    ``--boot-canary`` runs the launcher pre-flight (box 3) and exits 0/1.
    ``--floor-canary`` proves the temple sandbox floor is provisioned and binds.
    """
    if boot_canary:
        from cosmic_cli.boot_canary import run_boot_canary

        raise SystemExit(run_boot_canary())
    if floor_canary:
        from cosmic_cli.temple_floor import run_floor_canary

        raise SystemExit(run_floor_canary())
    from cosmic_cli.gate import run_gate

    raise SystemExit(run_gate(hook=hook, verb_check=verb_check, exec_mode=mode))

def main() -> None:
    cli()


if __name__ == "__main__":
    cli()
