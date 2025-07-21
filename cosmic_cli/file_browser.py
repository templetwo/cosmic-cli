#!/usr/bin/env python3
"""
🗂️ Cosmic File Browser - Interactive File Selection 🗂️
Beautiful file browser with rich interface for the Cosmic CLI
"""

import os
import stat
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.shortcuts import radiolist_dialog, input_dialog
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout.containers import HSplit, VSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import Frame, TextArea, Button, Dialog
    from prompt_toolkit.layout import Layout
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.tree import Tree


class CosmicFileBrowser:
    """Interactive file browser with cosmic aesthetics"""
    
    def __init__(self, start_path: str = "."):
        self.console = Console()
        self.current_path = Path(start_path).resolve()
        self.selected_file: Optional[Path] = None
        
    def get_file_info(self, file_path: Path) -> dict:
        """Get detailed file information"""
        try:
            stat_info = file_path.stat()
            return {
                "size": stat_info.st_size,
                "modified": datetime.fromtimestamp(stat_info.st_mtime),
                "permissions": stat.filemode(stat_info.st_mode),
                "is_dir": file_path.is_dir(),
                "is_file": file_path.is_file(),
                "readable": os.access(file_path, os.R_OK)
            }
        except (OSError, PermissionError):
            return {
                "size": 0,
                "modified": datetime.now(),
                "permissions": "?",
                "is_dir": False,
                "is_file": False,
                "readable": False
            }
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"
    
    def get_directory_contents(self, path: Path) -> List[Tuple[str, dict]]:
        """Get sorted directory contents with info"""
        try:
            contents = []
            
            # Add parent directory option if not at root
            if path != path.parent:
                contents.append(("📁 ..", {"is_dir": True, "name": "..", "special": True}))
            
            # Get all items in directory
            items = []
            try:
                for item in path.iterdir():
                    if not item.name.startswith('.'):  # Skip hidden files by default
                        info = self.get_file_info(item)
                        info["name"] = item.name
                        items.append((item.name, info))
            except PermissionError:
                self.console.print(f"[red]Permission denied accessing {path}[/red]")
                return contents
            
            # Sort: directories first, then files, both alphabetically
            items.sort(key=lambda x: (not x[1]["is_dir"], x[0].lower()))
            
            # Format display names
            for name, info in items:
                if info["is_dir"]:
                    display_name = f"📁 {name}"
                else:
                    # Add file type emoji based on extension
                    ext = Path(name).suffix.lower()
                    emoji = self.get_file_emoji(ext)
                    display_name = f"{emoji} {name}"
                
                contents.append((display_name, info))
            
            return contents
            
        except Exception as e:
            self.console.print(f"[red]Error reading directory {path}: {e}[/red]")
            return []
    
    def get_file_emoji(self, extension: str) -> str:
        """Get emoji for file type"""
        emoji_map = {
            '.py': '🐍',
            '.js': '🟨',
            '.ts': '🔷',
            '.html': '🌐',
            '.css': '🎨',
            '.json': '📄',
            '.xml': '📄',
            '.yml': '📄',
            '.yaml': '📄',
            '.md': '📝',
            '.txt': '📄',
            '.pdf': '📕',
            '.doc': '📘',
            '.docx': '📘',
            '.xls': '📗',
            '.xlsx': '📗',
            '.ppt': '📙',
            '.pptx': '📙',
            '.zip': '🗜️',
            '.tar': '🗜️',
            '.gz': '🗜️',
            '.jpg': '🖼️',
            '.jpeg': '🖼️',
            '.png': '🖼️',
            '.gif': '🖼️',
            '.svg': '🖼️',
            '.mp3': '🎵',
            '.mp4': '🎬',
            '.mov': '🎬',
            '.avi': '🎬',
            '.sh': '⚙️',
            '.bash': '⚙️',
            '.zsh': '⚙️',
            '.env': '🔧',
            '.log': '📋'
        }
        return emoji_map.get(extension, '📄')
    
    def display_file_table(self, contents: List[Tuple[str, dict]]) -> None:
        """Display files in a beautiful table"""
        table = Table(title=f"🗂️  Directory: {self.current_path}", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="white", width=40)
        table.add_column("Size", style="blue", justify="right", width=10)
        table.add_column("Modified", style="green", width=20)
        table.add_column("Perms", style="yellow", width=10)
        
        for i, (display_name, info) in enumerate(contents):
            if info.get("special"):
                # Parent directory
                table.add_row(display_name, "-", "-", "-", style="dim")
            else:
                size_str = self.format_file_size(info["size"]) if not info["is_dir"] else "-"
                modified_str = info["modified"].strftime("%m-%d %H:%M")
                perms_str = info["permissions"]
                
                style = "bold blue" if info["is_dir"] else "white"
                if not info["readable"]:
                    style = "dim red"
                
                table.add_row(f"{i:2}. {display_name}", size_str, modified_str, perms_str, style=style)
        
        panel = Panel(table, border_style="blue", padding=(1, 2))
        self.console.print(panel)
    
    def browse_files(self, file_filter: Optional[str] = None) -> Optional[Path]:
        """Interactive file browser"""
        self.console.print(Panel(
            "[bold cyan]🌟 Cosmic File Browser[/bold cyan]\n"
            "[dim]Navigate: Enter number or 'cd path' • Parent: 0 or .. • Quit: q[/dim]",
            border_style="magenta"
        ))
        
        while True:
            try:
                # Get directory contents
                contents = self.get_directory_contents(self.current_path)
                
                if not contents:
                    self.console.print("[red]No accessible files in this directory[/red]")
                    return None
                
                # Display files
                self.console.clear()
                self.console.print(f"\n[bold yellow]Current Path:[/bold yellow] {self.current_path}")
                self.display_file_table(contents)
                
                # Show help
                self.console.print("\n[dim]Commands:[/dim]")
                self.console.print("[cyan]  0-9[/cyan]   - Select item by number")
                self.console.print("[cyan]  cd path[/cyan] - Change directory")
                self.console.print("[cyan]  ..[/cyan]     - Go to parent directory")  
                self.console.print("[cyan]  q[/cyan]      - Quit browser")
                self.console.print("[cyan]  h[/cyan]      - Show hidden files")
                
                # Get user input
                choice = Prompt.ask("\n[bold green]🌌 Choice", default="q").strip()
                
                if choice.lower() == 'q':
                    return None
                elif choice.lower() == 'h':
                    self.show_hidden_files()
                    continue
                elif choice == '..' or choice == '0':
                    if self.current_path != self.current_path.parent:
                        self.current_path = self.current_path.parent
                    continue
                elif choice.startswith('cd '):
                    new_path = choice[3:].strip()
                    self.change_directory(new_path)
                    continue
                
                # Try to parse as number
                try:
                    index = int(choice)
                    if 1 <= index <= len(contents):
                        selected_display, selected_info = contents[index - 1]
                        
                        if selected_info.get("special"):
                            # Parent directory
                            self.current_path = self.current_path.parent
                            continue
                        
                        selected_path = self.current_path / selected_info["name"]
                        
                        if selected_info["is_dir"]:
                            self.current_path = selected_path
                            continue
                        else:
                            # File selected
                            if file_filter:
                                ext = selected_path.suffix.lower()
                                if not ext.endswith(file_filter):
                                    if not Confirm.ask(f"File type {ext} doesn't match filter {file_filter}. Select anyway?"):
                                        continue
                            
                            # Confirm selection
                            self.console.print(f"\n[green]✨ Selected:[/green] {selected_path}")
                            if Confirm.ask("Use this file?", default=True):
                                return selected_path
                    else:
                        self.console.print(f"[red]Invalid selection. Choose 1-{len(contents)}[/red]")
                        
                except ValueError:
                    self.console.print("[red]Invalid input. Enter a number, 'cd path', '..', or 'q'[/red]")
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]🌟 File browser cancelled[/yellow]")
                return None
            except Exception as e:
                self.console.print(f"[red]Error in file browser: {e}[/red]")
                return None
    
    def change_directory(self, path_str: str):
        """Change to specified directory"""
        try:
            new_path = Path(path_str)
            if not new_path.is_absolute():
                new_path = self.current_path / new_path
            
            new_path = new_path.resolve()
            
            if new_path.exists() and new_path.is_dir():
                self.current_path = new_path
                self.console.print(f"[green]✅ Changed to: {self.current_path}[/green]")
            else:
                self.console.print(f"[red]❌ Directory not found: {path_str}[/red]")
        except Exception as e:
            self.console.print(f"[red]❌ Error changing directory: {e}[/red]")
    
    def show_hidden_files(self):
        """Toggle showing hidden files"""
        self.console.print("[yellow]🔍 Showing hidden files feature coming soon![/yellow]")
    
    def browse_with_tree(self, max_depth: int = 2) -> Optional[Path]:
        """Alternative tree-style browser for quick navigation"""
        self.console.print(Panel(
            "[bold cyan]🌳 Cosmic File Tree[/bold cyan]\n"
            f"[dim]Exploring: {self.current_path}[/dim]",
            border_style="green"
        ))
        
        tree = Tree(f"📁 {self.current_path.name}")
        self._build_tree(tree, self.current_path, max_depth, 0)
        
        self.console.print(tree)
        
        # Simple file selection from tree
        file_path = Prompt.ask("\n[bold green]Enter full file path")
        if file_path and Path(file_path).exists():
            return Path(file_path)
        
        return None
    
    def _build_tree(self, tree_node, path: Path, max_depth: int, current_depth: int):
        """Recursively build file tree"""
        if current_depth >= max_depth:
            return
        
        try:
            items = []
            for item in path.iterdir():
                if not item.name.startswith('.'):
                    items.append(item)
            
            # Sort directories first
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for item in items[:10]:  # Limit to prevent overwhelming display
                emoji = "📁" if item.is_dir() else self.get_file_emoji(item.suffix.lower())
                node = tree_node.add(f"{emoji} {item.name}")
                
                if item.is_dir() and current_depth < max_depth - 1:
                    self._build_tree(node, item, max_depth, current_depth + 1)
                    
        except PermissionError:
            tree_node.add("🔒 [Permission Denied]")


def cosmic_file_picker(start_path: str = ".", file_filter: str = None) -> Optional[str]:
    """
    Cosmic file picker function for easy integration
    
    Args:
        start_path: Starting directory for browsing
        file_filter: File extension filter (e.g., '.py', '.txt')
    
    Returns:
        Selected file path as string, or None if cancelled
    """
    browser = CosmicFileBrowser(start_path)
    
    console = Console()
    console.print("[bold cyan]🌌 Welcome to the Cosmic File Browser![/bold cyan]")
    
    # Offer browsing options
    browse_type = Prompt.ask(
        "\n[bold yellow]Choose browsing mode[/bold yellow]",
        choices=["interactive", "tree", "manual"],
        default="interactive"
    )
    
    selected_file = None
    
    if browse_type == "interactive":
        selected_file = browser.browse_files(file_filter)
    elif browse_type == "tree":
        selected_file = browser.browse_with_tree()
    else:
        # Manual entry with tab completion hints
        console.print(f"[dim]Current directory: {browser.current_path}[/dim]")
        file_path = Prompt.ask("[bold green]Enter file path")
        if file_path and Path(file_path).exists():
            selected_file = Path(file_path)
    
    return str(selected_file) if selected_file else None


# Test the browser if run directly
if __name__ == "__main__":
    console = Console()
    console.print("[bold magenta]🌟 Cosmic File Browser Test 🌟[/bold magenta]")
    
    result = cosmic_file_picker()
    
    if result:
        console.print(f"[bold green]✨ Selected file: {result}[/bold green]")
    else:
        console.print("[yellow]🌙 No file selected[/yellow]")
