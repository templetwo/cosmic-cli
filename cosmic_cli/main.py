import os
import click
from xai_sdk import Client
from xai_sdk.chat import user, system, assistant # Import user, system, and assistant functions
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
import pyfiglet
import random
import subprocess
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from cosmic_cli.ui import DirectivesUI
from cosmic_cli.agents import StargazerAgent
import time
from rich.progress import Progress
from click.core import Command, Group

# --- Manual .env loading ---
def load_manual_env() -> None:
    """Load environment variables from .env file with flexible path detection."""
    # Try multiple possible .env locations
    possible_paths = [
        Path.cwd() / '.env',  # Current directory
        Path(__file__).parent / '.env',  # Same directory as script
        Path.home() / '.cosmic-cli' / '.env',  # User's home directory
    ]
    
    for dotenv_path in possible_paths:
        if dotenv_path.exists():
            try:
                with open(dotenv_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ.setdefault(key.strip(), value.strip())
                break
            except Exception as e:
                console.print(f"[bold yellow]Warning: Could not load .env from {dotenv_path}: {e}[/bold yellow]")

load_manual_env()
# --- End of manual loading ---

console = Console()

COSMIC_QUOTES = [
    "From the edge of the universe:",
    "The stars whisper from the void:",
    "A cosmic thought emerges:",
    "The stars align for this response:",
    "Behold, a message from the cosmos:",
]

# Define the memory file path in the user's home directory
MEMORY_FILE = Path.home() / ".cosmic_cli_memory.json"

class CosmicCLI:
    """Main CLI class to manage xAI interactions and state."""
    
    def __init__(self):
        self.chat_instance: Optional[Any] = None
        self.client_instance: Optional[Any] = None
        self.console = Console()
    
    def save_memory(self, messages: List[Dict[str, str]]) -> None:
        """Save conversation history to JSON file."""
        try:
            # Convert xai_sdk message objects back to dictionaries for saving
            serializable_messages = []
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    content = msg.content if isinstance(msg.content, str) else "<recursive agent memory>"
                    serializable_messages.append({'role': msg.role, 'content': content})

            with open(MEMORY_FILE, 'w') as f:
                json.dump(serializable_messages, f, indent=4)
        except Exception as e:
            self.console.print(f"[bold red]Error saving conversation history: {e}[/bold red]")

    def load_memory(self) -> List[Dict[str, str]]:
        """Load conversation history from JSON file."""
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.console.print("[bold yellow]Warning: Memory file is corrupted. Starting fresh.[/bold yellow]")
                return []
            except Exception as e:
                self.console.print(f"[bold red]Error loading memory: {e}[/bold red]")
                return []
        return []

    def append_to_memory(self, role: str, content: str) -> None:
        """Append a single message to memory."""
        current_history = self.load_memory()
        current_history.append({"role": role, "content": content})
        self.save_memory(current_history)

    def initialize_chat(self, load_history: bool = True) -> Optional[Any]:
        """Initialize the xAI chat instance."""
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not api_key:
            api_key = click.prompt("[bold yellow]Enter your xAI API key[/bold yellow]", type=str, hide_input=True)
            if api_key:
                os.environ["XAI_API_KEY"] = api_key
            else:
                self.console.print("[bold red]Error: API key is required.[/bold red]")
                return None
        
        self.client_instance = Client(api_key=api_key)
        self.chat_instance = self.client_instance.chat.create(model="grok-4")
        
        # Always add the system message first
        self.chat_instance.append(system("You are a helpful and friendly AI assistant named Cosmic CLI. You are designed to provide concise and accurate answers. Your responses should sometimes include a touch of cosmic or space-related humor or imagery."))

        if load_history:
            history = self.load_memory()
            # Append loaded history, converting dictionaries to xai_sdk message objects
            for msg in history:
                # Ensure we don't duplicate the system message if it's already added by initialize_chat
                if msg['role'] == 'system':
                    continue
                elif msg['role'] == 'user':
                    self.chat_instance.append(user(msg['content']))
                elif msg['role'] == 'assistant':
                    self.chat_instance.append(assistant(msg['content']))
                # Add more roles if xai_sdk supports them (e.g., 'tool', 'function')

        return self.chat_instance

# Global CLI instance
cosmic_cli = CosmicCLI()

class CosmicHelpFormatter(click.HelpFormatter):
    """Custom help formatter with cosmic styling."""
    
    def write_usage(self, usage: str, prefix: Optional[str] = None) -> None:
        console.print(Panel(Text(pyfiglet.figlet_format("Cosmic CLI"), justify="center", style="bold magenta"), border_style="blue"))
        console.print("[bold green]Built for xAI – Empowering humanity with cosmic AI![/bold green]\n")
        super().write_usage(usage, prefix)

    def write_heading(self, heading: str) -> None:
        console.print(f"[bold yellow]{heading}[/bold yellow]")

    def write_text(self, text: str) -> None:
        console.print(text)

    def write_dl(self, rows: List[tuple], col_max: int = 30, col_spacing: int = 2) -> None:
        for term, definition in rows:
            console.print(f"  [bold cyan]{term.ljust(col_max)}[/bold cyan] {definition}")

class CosmicCommand(Command):
    def get_help(self, ctx):
        formatter = CosmicHelpFormatter(width=ctx.terminal_width)
        self.format_help(ctx, formatter)
        return formatter.getvalue().rstrip('\n')

class CosmicGroup(Group):
    command_class = CosmicCommand
    group_class = type  # Allow subgroups to inherit

@click.group(cls=CosmicGroup, context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--ask', type=str, help='Ask xAI a single question')
@click.option('--analyze', type=click.Path(exists=True), help='Analyze a file')
@click.option('--hack', type=str, help='Get AI-powered terminal hacks (productivity, fun, learn)')
@click.option('--run-command', type=str, help='Run a terminal command')
def cli(ask, analyze, hack, run_command):
    """
    Cosmic CLI: Your cosmic companion.
    """
    # Let click handle subcommands directly
    pass

# Rename original commands to functions
def chat_command():
    # Original chat command code here
    chat_instance = cosmic_cli.initialize_chat(load_history=True)
    if chat_instance is None:
        return

    console.print(Panel(Text(pyfiglet.figlet_format("Cosmic CLI"), justify="center", style="bold magenta"), border_style="blue"))
    console.print("[bold green]Welcome to Cosmic CLI! Type 'quit' or 'exit' to end the conversation. Type 'reset' to clear the conversation history.[/bold green]\n")

    while True:
        user_input = click.prompt("[bold cyan]You[/bold cyan]", type=str, prompt_suffix="> ")

        if user_input.lower() in ['quit', 'exit']:
            console.print("[bold yellow]Goodbye, cosmic traveler![/bold yellow]")
            cosmic_cli.save_memory(chat_instance.messages)
            break
        elif user_input.lower() == 'reset':
            chat_instance = cosmic_cli.initialize_chat(load_history=False) # Start fresh, don't load old history
            cosmic_cli.save_memory([]) # Save an empty history
            if chat_instance is not None:
                console.print("[bold yellow]Wiping the cosmic slate clean! Ready for new adventures![/bold yellow]")
            continue

        chat_instance.append(user(user_input))

        try:
            response = chat_instance.sample()
            cosmic_prefix = random.choice(COSMIC_QUOTES)
            console.print(f"[bold blue]{cosmic_prefix}[/bold blue] [white]{response.content}[/white]")
            chat_instance.append(assistant(response.content))
            cosmic_cli.save_memory(chat_instance.messages)
        except Exception as e:
            console.print(f"[bold red]An error occurred during response generation: {e}[/bold red]")
            # If an error occurs, remove the last user message to avoid sending bad context
            if chat_instance and chat_instance.messages:
                chat_instance.messages.pop()

def ask_command(prompt: str):
    # Original ask command code here
    chat_instance = cosmic_cli.initialize_chat(load_history=False)
    if chat_instance is None:
        return

    try:
        chat_instance.append(user(prompt))
        response = chat_instance.sample()
        cosmic_prefix = random.choice(COSMIC_QUOTES)
        console.print(f"[bold blue]{cosmic_prefix}[/bold blue] [white]{response.content}[/white]")
        cosmic_cli.append_to_memory("user", prompt)
        cosmic_cli.append_to_memory("assistant", response.content)
    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")

def analyze_command(file_path: str):
    chat_instance = cosmic_cli.initialize_chat(load_history=False)
    if chat_instance is None:
        return

    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        console.print(f"[bold yellow]Analyzing file: {file_path}[/bold yellow]")
        
        analysis_prompt = f"Please analyze the following content from a file and provide insights or suggestions:\n\n```\n{content}\n```"
        
        chat_instance.append(user(analysis_prompt))
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Analyzing...", total=100)
            for _ in range(100):
                time.sleep(0.01)  # Simulate progress
                progress.update(task, advance=1)
        
        response = chat_instance.sample()
        
        cosmic_prefix = random.choice(COSMIC_QUOTES)
        console.print(f"[bold blue]{cosmic_prefix}[/bold blue] [white]{response.content}[/white]")

        cosmic_cli.append_to_memory("user", analysis_prompt)
        cosmic_cli.append_to_memory("assistant", response.content)

    except UnicodeDecodeError:
        console.print(f"[bold red]Error: Could not decode file {file_path}. It might be a binary file. Cosmic CLI currently supports text-based files for analysis.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An error occurred during file analysis: {e}[/bold red]")

def hack_command(mode: str):
    chat_instance = cosmic_cli.initialize_chat(load_history=False)
    if chat_instance is None:
        return

    if mode.lower() == "productivity":
        hack_prompt = "Suggest a simple terminal command for a 25-minute Pomodoro timer with a 5-minute break. Provide only the command, no extra text."
    elif mode.lower() == "fun":
        hack_prompt = "Suggest a fun, simple terminal command or ASCII art generator. Provide only the command, no extra text."
    elif mode.lower() == "learn":
        hack_prompt = "Suggest a terminal command to learn about a random Linux command. Provide only the command, no extra text."
    else:
        console.print("[bold red]Invalid hack mode. Choose from: productivity, fun, learn.[/bold red]")
        return

    console.print(f"[bold yellow]xAI is thinking of a {mode} hack...[/bold yellow]")
    chat_instance.append(user(hack_prompt))

    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Generating hack...", total=100)
            for _ in range(100):
                time.sleep(0.01)  # Simulate progress
                progress.update(task, advance=1)
        
        response = chat_instance.sample()
        suggested_command = response.content.strip()
        console.print(f"[bold green]xAI suggests:[/bold green] [yellow]{suggested_command}[/yellow]")
        console.print("[bold yellow]To run this command, use: cosmic-cli --run-command \"{suggested_command}\"[/bold yellow]")
        cosmic_cli.append_to_memory("user", hack_prompt)
        cosmic_cli.append_to_memory("assistant", suggested_command)
    except Exception as e:
        console.print(f"[bold red]An error occurred while generating hack: {e}[/bold red]")

def run_command_command(command_to_run: str):
    # Original run_command code here
    console.print(f"[bold red]WARNING: Executing command:[/bold red] [yellow]{command_to_run}[/yellow]")
    confirm = click.confirm("Are you sure you want to run this command?", default=False)
    if confirm:
        try:
            result = subprocess.run(command_to_run, shell=True, capture_output=True, text=True, check=True)
            console.print("[bold green]Command Output:[/bold green]")
            console.print(result.stdout)
            if result.stderr:
                console.print("[bold yellow]Command Errors:[/bold yellow]")
                console.print(result.stderr)
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Command failed with error code {e.returncode}:[/bold red]")
            console.print(e.stderr)
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred while running command: {e}[/bold red]")
    else:
        console.print("[bold yellow]Command execution cancelled.[/bold yellow]")

@cli.command()
@click.argument('tasks', nargs=-1)
def workflow(tasks):
    """
    Chain tasks in a workflow using StargazerAgent.
    """
    if not tasks:
        console.print("[bold red]Usage: cosmic-cli workflow <task1> <arg1> <task2> <arg2> ...[/bold red]")
        return

    # Formulate directive from tasks
    directive_parts = []
    for i in range(0, len(tasks), 2):
        task = tasks[i]
        arg = tasks[i+1] if i+1 < len(tasks) else ''
        if task == 'analyze':
            directive_parts.append(f"Analyze file {arg}")
        elif task == 'hack':
            directive_parts.append(f"Generate hack in mode {arg}")
        else:
            console.print(f"[bold red]Unknown task: {task}[/bold red]")
            return

    directive = " and ".join(directive_parts) + "."

    # Invoke Stargazer
    def on_update(msg):
        console.print(f"[cyan]{msg}[/cyan]")

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        console.print("[red]XAI_API_KEY not set.[/red]")
        return

    agent = StargazerAgent(
        directive,
        api_key,
        on_update,
        work_dir=os.getcwd()
    )
    result = agent.execute()
    console.print(f"[green]Workflow completed via Stargazer: {result}[/green]")

@click.group()
def stargazer():
    """Cosmic Stargazer agents for directive execution."""
    pass

@stargazer.command()
@click.argument('directive')
def deploy(directive):
    def on_update(msg):
        console.print(f"[cyan]{msg}[/cyan]")

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        console.print("[red]XAI_API_KEY not set.[/red]")
        return

    agent = StargazerAgent(
        directive,
        api_key,
        on_update,
        work_dir=os.getcwd()
    )
    result = agent.execute()
    console.print(f"[green]Stargazer deployed for '{directive}'—engaging warp drive![/green]")

# Add to main CLI
cli.add_command(stargazer)

def main():
    DirectivesUI().run()

if __name__ == "__main__":
    main()