"""Cosmic CLI entrypoint — Grok chat + Stargazer agent."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import pyfiglet
from click.core import Command, Group
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from xai_sdk import Client
from xai_sdk.chat import assistant, system, user

from cosmic_cli.agents import DEFAULT_MODEL, SESSION_DIR, StargazerAgent
from cosmic_cli import helix_bridge
from cosmic_cli.init_cmd import write_init
from cosmic_cli.principles import DIFFERENTIATORS
from cosmic_cli.review import (
    build_bundle,
    latest_session_id,
    run_review,
)
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


class CosmicHelpFormatter(click.HelpFormatter):
    def write_usage(self, usage: str, prefix: Optional[str] = None) -> None:
        console.print(
            Panel(
                Text(
                    pyfiglet.figlet_format("Cosmic CLI", font="small"),
                    justify="center",
                    style="bold magenta",
                ),
                border_style="blue",
                subtitle=f"model default: {DEFAULT_MODEL}",
            )
        )
        super().write_usage(usage, prefix)

    def write_heading(self, heading: str) -> None:
        console.print(f"[bold yellow]{heading}[/bold yellow]")

    def write_text(self, text: str) -> None:
        console.print(text)

    def write_dl(
        self, rows: List[tuple], col_max: int = 30, col_spacing: int = 2
    ) -> None:
        for term, definition in rows:
            console.print(
                f"  [bold cyan]{term.ljust(col_max)}[/bold cyan] {definition}"
            )


class CosmicCommand(Command):
    def get_help(self, ctx):
        formatter = CosmicHelpFormatter(width=ctx.terminal_width)
        self.format_help(ctx, formatter)
        return formatter.getvalue().rstrip("\n")


class CosmicGroup(Group):
    command_class = CosmicCommand
    group_class = type


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


@cli.command("chat")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def chat_cmd(model: str) -> None:
    """Interactive chat with conversation memory."""
    chat_instance = cosmic_cli.initialize_chat(load_history=True, model=model)
    if chat_instance is None:
        return
    console.print(f"[green]Chat · {model} · quit/exit to leave, reset clears history[/green]\n")
    while True:
        user_input = click.prompt("You", type=str, prompt_suffix="> ")
        if user_input.lower() in ("quit", "exit"):
            cosmic_cli.save_memory(getattr(chat_instance, "messages", []))
            console.print("[yellow]bye[/yellow]")
            break
        if user_input.lower() == "reset":
            chat_instance = cosmic_cli.initialize_chat(load_history=False, model=model)
            cosmic_cli.save_memory([])
            console.print("[yellow]history cleared[/yellow]")
            continue
        chat_instance.append(user(user_input))
        try:
            response = chat_instance.sample()
            prefix = random.choice(COSMIC_QUOTES)
            console.print(f"[blue]{prefix}[/blue] {response.content}")
            chat_instance.append(assistant(response.content))
            cosmic_cli.save_memory(getattr(chat_instance, "messages", []))
        except Exception as e:
            console.print(f"[red]{e}[/red]")


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
def run_cmd(command_to_run: str) -> None:
    """Run a shell command after confirmation."""
    console.print(f"[red]run:[/red] {command_to_run}")
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
    """
    action = action.lower()
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
def init_cmd(directory: str, force: bool, name: str) -> None:
    """Drop COSMIC.md agent notes into a project (no stack)."""
    msg, ok = write_init(Path(directory), force=force, name=name)
    console.print(f"[{'green' if ok else 'red'}]{msg}[/]")
    if not ok:
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


def main() -> None:
    cli()


if __name__ == "__main__":
    cli()
