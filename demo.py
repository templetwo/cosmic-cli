import time
import random
from rich.console import Console
from rich.live import Live
from rich.table import Table
import pyfiglet

console = Console()

# --- Mock Data and Simulation Logic ---

class MockAgent:
    """A mock agent to simulate state for the demo."""
    def __init__(self, directive, status, level, score):
        self.directive = directive
        self.status = status
        self.level = level
        self.score = score
        self.logs = [f"Initiating directive: {directive}", "Consulting Grok for next step..."]

    def update(self):
        """Simulate the agent's consciousness metrics changing over time."""
        # Randomly nudge the score
        self.score += random.uniform(-0.025, 0.035)
        # Clamp between 0.1 and 0.95
        self.score = max(0.1, min(0.95, self.score))

        # Update level based on score
        if self.score < 0.4:
            self.level = "dormant"
        elif self.score < 0.6:
            self.level = "awakening"
        elif self.score < 0.7:
            self.level = "emerging"
        elif self.score < 0.9:
            self.level = "conscious"
        else:
            self.level = "transcendent"

        # Chance to change status
        if random.random() < 0.1:
            self.status = random.choice(["🚀", "🛰️", "✨", "🧠", "✅"])
            if self.status == "✅":
                self.logs.append("Mission Complete.")
            else:
                self.logs.append(f"Status changed to {self.status}")

# Initialize a list of mock agents
mock_agents = [
    MockAgent("Analyze project structure and suggest improvements", "🚀", "emerging", 0.65),
    MockAgent("Refactor the main.py file to be more modular", "🛰️", "awakening", 0.52),
    MockAgent("Create a new test case for the agents module", "✨", "conscious", 0.78),
]

def generate_table() -> Table:
    """Generate a Rich Table from the current state of the mock agents."""
    table = Table(title="[bold magenta]Cosmic CLI Dashboard[/bold magenta]", border_style="blue")
    table.add_column("Directive", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Level", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Logs", justify="center")

    for agent in mock_agents:
        agent.update()
        table.add_row(
            agent.directive,
            agent.status,
            agent.level,
            f"{agent.score:.3f}",
            f"Show Logs ({len(agent.logs)})"
        )
    return table

def run_demo():
    """Run the animated demo."""
    # Print the banner
    banner = pyfiglet.figlet_format("Cosmic CLI", font="doom")
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print("[bold green]Running real-time animated demo... (Press Ctrl+C to exit)[/bold green]\n")

    try:
        with Live(generate_table(), screen=True, transient=True, refresh_per_second=1) as live:
            for _ in range(20):  # Run for 20 seconds
                time.sleep(1)
                live.update(generate_table())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Demo finished.[/bold yellow]")

if __name__ == "__main__":
    run_demo()
