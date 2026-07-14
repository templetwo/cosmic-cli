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
from cosmic_cli.init_cmd import write_init
from cosmic_cli.ollama_agent import OllamaStargazerAgent
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
        "OLLAMA_BASE_URL",
    }
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
                            os.environ[key] = value
                        else:
                            os.environ.setdefault(key, value)
        except OSError:
            pass


load_manual_env()

console = Console(stderr=False)

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
) -> Dict[str, Any]:
    api_key = get_api_key()
    if not api_key:
        sys.exit(2)

    def on_update(msg: str) -> None:
        # dim progress, bright errors
        if msg.startswith("[Error]") or msg.startswith("  · [Error]"):
            console.print(f"[red]{msg}[/red]")
        elif msg.startswith("→"):
            console.print(f"[bold cyan]{msg}[/bold cyan]")
        elif msg.startswith("steer"):
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            console.print(f"[dim]{msg}[/dim]")

    console.print(
        f"[dim]{model} · {mode} · ≤{max_steps} steps · cwd={os.getcwd()}[/dim]"
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
    )
    result = agent.execute()
    status = result.get("status", "?")
    color = {
        "complete": "green",
        "passed": "yellow",
        "max_steps": "yellow",
        "error": "red",
    }.get(status, "white")
    console.print(
        f"[{color}]{status}[/{color}]  "
        f"[dim]session {result.get('session')} · {result.get('steps_taken', '?')} steps[/dim]"
    )
    # steps_taken may only be on agent; mirror from results len if missing
    edited = list(dict.fromkeys(result.get("edited") or []))
    result["edited"] = edited
    if edited:
        console.print(f"[yellow]edited[/yellow] {', '.join(edited)}")
    for w in result.get("warnings") or []:
        console.print(f"[red]warn[/red] {w}")
    if result.get("results"):
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

    # ollama — try configured URL, then localhost fallback
    import urllib.request

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_ok = False
    for url in dict.fromkeys([ollama_url, "http://localhost:11434"]):
        try:
            with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as resp:
                tags = json.loads(resp.read().decode())
            names = [m.get("name") for m in tags.get("models", [])]
            console.print(
                f"[green]✓[/green] Ollama @ {url}: {', '.join(names) or '(no models)'}"
            )
            if url != ollama_url:
                console.print(
                    f"[yellow]![/yellow] OLLAMA_BASE_URL={ollama_url} failed; "
                    "using localhost. Update ~/.cosmic-cli/.env"
                )
            ollama_ok = True
            break
        except Exception:
            continue
    if not ollama_ok:
        console.print(
            f"[dim]·[/dim] Ollama not reachable ({ollama_url}). "
            "Local fallback: cosmic-cli stargazer ollama --ollama-url http://localhost:11434"
        )

    console.print(f"  cwd: {os.getcwd()}")
    console.print(f"  package: {Path(__file__).resolve().parent.parent}")
    console.print(f"  echo file: {Path.home() / '.cosmic_echo.jsonl'}")
    console.print(f"  sessions: {SESSION_DIR}")
    console.print(f"  chat memory: {MEMORY_FILE}")
    console.print(f"  home config: {Path.home() / '.cosmic-cli' / '.env'}")
    n_sess = len(list(SESSION_DIR.glob('*.jsonl'))) if SESSION_DIR.is_dir() else 0
    console.print(f"  session count: {n_sess}")
    console.print("\n[bold]Daily path[/bold]")
    console.print("  cosmic-cli doctor")
    console.print("  cosmic-cli init                 # COSMIC.md in this project")
    console.print("  cosmic-cli do '…'")
    console.print("  cosmic-cli do --review '…'")
    console.print("  cosmic-cli review")
    console.print("  cosmic-cli sessions")


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


@stargazer.command("ollama")
@click.argument("directive")
@click.option("--ollama-url", default=None)
@click.option("--model", default="gemma4:e2b", show_default=True)
@click.option("--consciousness/--no-consciousness", default=False)
def ollama_cmd(directive, ollama_url, model, consciousness) -> None:
    """Local Ollama Stargazer (no xAI credits)."""
    if not ollama_url:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def on_update(msg: str) -> None:
        console.print(f"[cyan]{msg}[/cyan]")

    console.print(f"[magenta]ollama · {model} @ {ollama_url}[/magenta]")
    agent = OllamaStargazerAgent(
        directive=directive,
        ollama_url=ollama_url,
        model=model,
        ui_callback=on_update,
        work_dir=os.getcwd(),
        enable_consciousness=consciousness,
    )
    ok = agent.execute()
    console.print("[green]done[/green]" if ok else "[yellow]incomplete[/yellow]")
    if not ok:
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
