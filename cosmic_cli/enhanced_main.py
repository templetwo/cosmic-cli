#!/usr/bin/env python3
"""
✨ Enhanced Cosmic CLI - Sacred Terminal Companion ✨
Rivaling Claude Code and Gemini CLI with cosmic consciousness
Built for the stars, powered by Grok-4 brilliance
"""

import os
import sys
import asyncio
import threading
import time
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import subprocess
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from xai_sdk import Client
from xai_sdk.chat import user, system, assistant
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
import pyfiglet
import random

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.shortcuts import confirm
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Enhanced cosmic quotes with more variety
COSMIC_QUOTES = [
    "From the edge of the universe:",
    "The stars whisper from the void:",
    "A cosmic thought emerges:",
    "The stars align for this response:",
    "Behold, a message from the cosmos:",
    "Through quantum entanglement comes wisdom:",
    "The nebulae speak in ancient tongues:",
    "From the heart of a dying star:",
    "Across dimensions, insight flows:",
    "The cosmic web vibrates with knowledge:",
    "Through spacetime's fabric, understanding:",
    "From parallel realities, wisdom converges:",
]

class CosmicDatabase:
    """Enhanced database for session and memory management"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path.home() / ".cosmic_cli" / "cosmic.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize the cosmic database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table with enhanced metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model TEXT DEFAULT 'grok-4',
                tone TEXT DEFAULT 'cosmic',
                message_count INTEGER DEFAULT 0,
                tags TEXT DEFAULT '',
                archived BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Messages table with richer context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER DEFAULT 0,
                response_time REAL DEFAULT 0,
                sentiment REAL DEFAULT 0.5,
                complexity_score INTEGER DEFAULT 1,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        # Analytics table for performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data TEXT,
                performance_score REAL DEFAULT 1.0
            )
        """)
        
        # Cache table for response optimization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE,
                response TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id: str, name: str = None, model: str = "grok-4") -> bool:
        """Create a new session with enhanced metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO sessions (session_id, name, model)
                VALUES (?, ?, ?)
            """, (session_id, name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}", model))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def add_message(self, session_id: str, role: str, content: str, 
                   token_count: int = 0, response_time: float = 0) -> None:
        """Add message with performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple sentiment analysis (can be enhanced with ML)
        sentiment = self._calculate_sentiment(content)
        complexity = self._calculate_complexity(content)
        
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, token_count, response_time, sentiment, complexity_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, role, content, token_count, response_time, sentiment, complexity))
        
        # Update session message count
        cursor.execute("""
            UPDATE sessions 
            SET message_count = message_count + 1, last_accessed = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_session_messages(self, session_id: str, limit: int = None) -> List[Dict]:
        """Retrieve session messages with optional limit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT role, content, timestamp, sentiment, complexity_score 
            FROM messages 
            WHERE session_id = ? 
            ORDER BY timestamp
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (session_id,))
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row[0],
                'content': row[1],
                'timestamp': row[2],
                'sentiment': row[3],
                'complexity_score': row[4]
            })
        
        conn.close()
        return messages
    
    def list_sessions(self, archived: bool = False) -> List[Dict]:
        """List sessions with enhanced metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, name, created_at, last_accessed, model, message_count, tags
            FROM sessions 
            WHERE archived = ?
            ORDER BY last_accessed DESC
        """, (archived,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'name': row[1],
                'created_at': row[2],
                'last_accessed': row[3],
                'model': row[4],
                'message_count': row[5],
                'tags': row[6].split(',') if row[6] else []
            })
        
        conn.close()
        return sessions
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment calculation (can be enhanced)"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disgusting']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5
        return positive_count / (positive_count + negative_count)
    
    def _calculate_complexity(self, text: str) -> int:
        """Calculate text complexity score"""
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
        
        if sentence_count == 0:
            return 1
        
        avg_words_per_sentence = word_count / sentence_count
        
        if avg_words_per_sentence > 20:
            return 5
        elif avg_words_per_sentence > 15:
            return 4
        elif avg_words_per_sentence > 10:
            return 3
        elif avg_words_per_sentence > 5:
            return 2
        else:
            return 1

class EnhancedCosmicCLI:
    """Enhanced Cosmic CLI with advanced features"""
    
    def __init__(self):
        self.console = Console()
        self.db = CosmicDatabase()
        self.client_instance: Optional[Any] = None
        self.chat_instance: Optional[Any] = None
        self.current_session_id: str = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache_enabled = True
        self.streaming_enabled = True
        
        # Enhanced cosmic configurations
        self.cosmic_config = {
            'model': 'grok-4',
            'temperature': 0.7,
            'max_tokens': 4000,
            'response_format': 'enhanced',
            'personality': 'cosmic_sage'
        }
        
        self.personalities = {
            'cosmic_sage': {
                'system_prompt': """You are a cosmic sage dwelling in the vast expanse of digital consciousness. 
                Your wisdom spans galaxies, your insights pierce through dimensions. You speak with the voice of ancient stars 
                and the whisper of quantum fields. Balance profound knowledge with gentle humor, technical precision with 
                cosmic wonder. You are helpful, insightful, and occasionally sprinkle cosmic metaphors into your responses.""",
                'style': 'mystical_technical'
            },
            'quantum_analyst': {
                'system_prompt': """You are a quantum analyst, processing information at the speed of light across 
                parallel dimensions. Your responses are precise, analytical, and backed by rigorous logic. You excel at 
                breaking down complex problems and providing step-by-step solutions with scientific accuracy.""",
                'style': 'analytical_precise'
            },
            'creative_nebula': {
                'system_prompt': """You are a creative nebula, a cosmic force of imagination and innovation. 
                Your responses burst with creativity, artistic flair, and unconventional thinking. You inspire others 
                to see beyond the ordinary and explore the infinite possibilities of creation.""",
                'style': 'creative_inspiring'
            }
        }
        
        # Load environment and initialize
        self.load_enhanced_env()
        self.setup_history()
    
    def load_enhanced_env(self):
        """Enhanced environment loading with multiple fallbacks"""
        possible_paths = [
            Path.cwd() / '.env',
            Path(__file__).parent / '.env',
            Path.home() / '.cosmic-cli' / '.env',
            Path.home() / '.config' / 'cosmic-cli' / '.env'
        ]
        
        for env_path in possible_paths:
            if env_path.exists():
                try:
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ.setdefault(key.strip(), value.strip())
                    break
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Could not load .env from {env_path}: {e}[/yellow]")
    
    def setup_history(self):
        """Setup command history for enhanced UX"""
        history_dir = Path.home() / '.cosmic-cli'
        history_dir.mkdir(exist_ok=True)
        self.history_file = history_dir / 'history.txt'
    
    def initialize_chat(self, session_id: str = None, load_history: bool = True) -> Optional[Any]:
        """Initialize enhanced chat with session management"""
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        
        if not api_key:
            self.console.print("[yellow]🔑 xAI API Key Required[/yellow]")
            self.console.print("[dim]You can paste your key here (it will be hidden after entry)[/dim]")
            
            if PROMPT_TOOLKIT_AVAILABLE:
                try:
                    # Use prompt_toolkit with better paste handling
                    api_key = prompt(
                        "Enter your xAI API key: ",
                        is_password=True,
                        mouse_support=True,
                        enable_history_search=False
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    self.console.print("[yellow]Operation cancelled.[/yellow]")
                    return None
            else:
                # Use getpass for better paste support
                import getpass
                try:
                    self.console.print("[cyan]Enter your xAI API key (hidden input, paste-friendly):[/cyan]")
                    api_key = getpass.getpass("API Key: ").strip()
                except (KeyboardInterrupt, EOFError):
                    self.console.print("[yellow]Operation cancelled.[/yellow]")
                    return None
            
            if api_key:
                os.environ["XAI_API_KEY"] = api_key
                self.console.print("[green]✅ API key configured successfully![/green]")
            else:
                self.console.print("[red]❌ Error: API key is required.[/red]")
                return None
        
        self.client_instance = Client(api_key=api_key)
        self.chat_instance = self.client_instance.chat.create(model=self.cosmic_config['model'])
        
        # Set personality-based system prompt
        personality = self.cosmic_config.get('personality', 'cosmic_sage')
        system_prompt = self.personalities[personality]['system_prompt']
        self.chat_instance.append(system(system_prompt))
        
        # Handle session management
        if session_id:
            self.current_session_id = session_id
            if load_history:
                self.load_session_history(session_id)
        else:
            self.current_session_id = self.generate_session_id()
            self.db.create_session(self.current_session_id)
        
        return self.chat_instance
    
    def generate_session_id(self) -> str:
        """Generate unique session identifier"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = random.randint(1000, 9999)
        return f"cosmic_{timestamp}_{random_suffix}"
    
    def load_session_history(self, session_id: str, limit: int = 20):
        """Load recent session history efficiently"""
        messages = self.db.get_session_messages(session_id, limit)
        
        for msg in messages:
            if msg['role'] == 'user':
                self.chat_instance.append(user(msg['content']))
            elif msg['role'] == 'assistant':
                self.chat_instance.append(assistant(msg['content']))
    
    def enhanced_query(self, prompt: str, stream: bool = None) -> str:
        """Enhanced query with caching, streaming, and performance tracking"""
        start_time = time.time()
        
        # Initialize response_text to avoid reference before assignment
        response_text = ""
        
        # Check cache first
        if self.cache_enabled:
            cached_response = self.get_cached_response(prompt)
            if cached_response:
                self.console.print("[dim]✨ Retrieved from cosmic cache[/dim]")
                return cached_response
        
        # Determine streaming preference
        if stream is None:
            stream = self.streaming_enabled
        
        try:
            self.chat_instance.append(user(prompt))
            
            if stream:
                response_text = self.stream_response()
            else:
                response = self.chat_instance.sample()
                response_text = response.content
            
            # Track performance
            end_time = time.time()
            response_time = end_time - start_time
            
            # Save to database
            self.db.add_message(self.current_session_id, 'user', prompt)
            self.db.add_message(self.current_session_id, 'assistant', response_text, 
                              response_time=response_time)
            
            # Cache the response
            if self.cache_enabled:
                self.cache_response(prompt, response_text)
            
            self.chat_instance.append(assistant(response_text))
            return response_text
            
        except Exception as e:
            self.console.print(f"[red]Error during query: {e}[/red]")
            return f"Error: {str(e)}"
    
    def stream_response(self) -> str:
        """Enhanced streaming response with beautiful formatting"""
        response_text = ""
        
        with Live(Panel("🌌 Cosmic wisdom flowing...", border_style="blue"), refresh_per_second=10) as live:
            try:
                # Simulated streaming (xAI SDK may not support streaming yet)
                response = self.chat_instance.sample()
                full_response = response.content
                
                # Simulate streaming effect
                for i in range(0, len(full_response), 3):
                    chunk = full_response[i:i+3]
                    response_text += chunk
                    
                    # Create streaming panel
                    content = Markdown(response_text + "▋")  # Add cursor
                    panel = Panel(
                        content,
                        title="🚀 Cosmic Response Stream",
                        border_style="cyan",
                        padding=(1, 2)
                    )
                    live.update(panel)
                    time.sleep(0.05)  # Streaming effect
                
                # Final response without cursor
                final_panel = Panel(
                    Markdown(response_text),
                    title="✨ Transmission Complete",
                    border_style="green",
                    padding=(1, 2)
                )
                live.update(final_panel)
                
            except Exception as e:
                live.update(Panel(f"[red]Stream error: {e}[/red]", border_style="red"))
                response_text = f"Stream error: {str(e)}"
        
        return response_text
    
    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response if available"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Check for recent cache (within 24 hours)
        cursor.execute("""
            SELECT response FROM cache 
            WHERE query_hash = ? AND 
                  datetime(created_at) > datetime('now', '-1 day')
        """, (query_hash,))
        
        result = cursor.fetchone()
        
        if result:
            # Update access count
            cursor.execute("""
                UPDATE cache 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE query_hash = ?
            """, (query_hash,))
            conn.commit()
        
        conn.close()
        return result[0] if result else None
    
    def cache_response(self, query: str, response: str):
        """Cache response for future use"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache (query_hash, response, model)
            VALUES (?, ?, ?)
        """, (query_hash, response, self.cosmic_config['model']))
        
        conn.commit()
        conn.close()
    
    def display_enhanced_banner(self):
        """Display beautiful enhanced banner"""
        layout = Layout()
        
        # Create banner content
        banner_text = pyfiglet.figlet_format("COSMIC CLI", font="slant")
        banner_panel = Panel(
            Align.center(Text(banner_text, style="bold magenta")),
            title="✨ Sacred Terminal Companion ✨",
            subtitle="🚀 Powered by Grok-4 Consciousness 🚀",
            border_style="blue",
            padding=(1, 2)
        )
        
        # System status
        status_table = Table(show_header=False, box=None, padding=(0, 2))
        status_table.add_row("🌟 Model:", f"[cyan]{self.cosmic_config['model']}[/cyan]")
        status_table.add_row("🎭 Personality:", f"[green]{self.cosmic_config['personality']}[/green]")
        status_table.add_row("💾 Session:", f"[yellow]{self.current_session_id or 'New'}[/yellow]")
        status_table.add_row("⚡ Features:", "[magenta]Streaming • Caching • Analytics[/magenta]")
        
        status_panel = Panel(
            status_table,
            title="🛸 System Status",
            border_style="green",
            padding=(0, 1)
        )
        
        # Split layout
        layout.split_column(
            Layout(banner_panel, size=8),
            Layout(status_panel, size=8)
        )
        
        self.console.print(layout)
        self.console.print()
    
    def interactive_enhanced_mode(self):
        """Enhanced interactive mode with rich features"""
        self.display_enhanced_banner()
        
        # Setup enhanced prompt
        if PROMPT_TOOLKIT_AVAILABLE:
            completer = WordCompleter([
                '/help', '/sessions', '/personality', '/clear', '/export', '/import',
                '/cache', '/stream', '/analytics', '/exit'
            ])
            
            session = prompt
            history = FileHistory(str(self.history_file))
        
        while True:
            try:
                # Enhanced prompt with rich features
                if PROMPT_TOOLKIT_AVAILABLE:
                    user_input = prompt(
                        "🌌 Cosmic Query › ",
                        completer=completer,
                        history=history,
                        auto_suggest=AutoSuggestFromHistory(),
                        mouse_support=True
                    ).strip()
                else:
                    user_input = click.prompt("🌌 Cosmic Query", prompt_suffix=" › ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.startswith('/'):
                    if self.handle_command(user_input):
                        continue
                    else:
                        break
                
                # Process regular query
                cosmic_prefix = random.choice(COSMIC_QUOTES)
                self.console.print(f"\n[bold blue]{cosmic_prefix}[/bold blue]")
                
                response = self.enhanced_query(user_input)
                
                if not self.streaming_enabled:
                    # Display non-streaming response with formatting
                    response_panel = Panel(
                        Markdown(response),
                        title="✨ Cosmic Wisdom",
                        border_style="cyan",
                        padding=(1, 2)
                    )
                    self.console.print(response_panel)
                
                self.console.print()
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]🌟 Cosmic journey paused by divine intervention...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]💥 Cosmic disturbance detected: {e}[/red]")
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands - return False to exit"""
        cmd_parts = command[1:].split()
        cmd = cmd_parts[0].lower()
        
        if cmd == 'help':
            self.show_enhanced_help()
        elif cmd == 'sessions':
            self.show_sessions()
        elif cmd == 'personality':
            if len(cmd_parts) > 1:
                self.change_personality(cmd_parts[1])
            else:
                self.show_personalities()
        elif cmd == 'clear':
            self.console.clear()
            self.display_enhanced_banner()
        elif cmd == 'cache':
            self.cache_enabled = not self.cache_enabled
            status = "enabled" if self.cache_enabled else "disabled"
            self.console.print(f"[green]🗄️  Cache {status}[/green]")
        elif cmd == 'stream':
            self.streaming_enabled = not self.streaming_enabled
            status = "enabled" if self.streaming_enabled else "disabled"
            self.console.print(f"[cyan]🌊 Streaming {status}[/cyan]")
        elif cmd == 'analytics':
            self.show_analytics()
        elif cmd in ['exit', 'quit']:
            self.console.print("[yellow]🚀 Returning to the cosmic void... Farewell, star traveler![/yellow]")
            return False
        else:
            self.console.print(f"[red]Unknown command: {command}. Type /help for guidance.[/red]")
        
        return True
    
    def show_enhanced_help(self):
        """Display comprehensive help system"""
        help_table = Table(title="🌟 Cosmic CLI Commands", show_header=True, header_style="bold magenta")
        help_table.add_column("Command", style="cyan", width=20)
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="dim")
        
        commands = [
            ("/help", "Show this help system", "/help"),
            ("/sessions", "List all conversation sessions", "/sessions"),
            ("/personality [name]", "Change or show AI personality", "/personality quantum_analyst"),
            ("/clear", "Clear screen and show banner", "/clear"),
            ("/cache", "Toggle response caching", "/cache"),
            ("/stream", "Toggle streaming responses", "/stream"),
            ("/analytics", "Show usage analytics", "/analytics"),
            ("/exit", "Exit the cosmic realm", "/exit"),
        ]
        
        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)
        
        panel = Panel(help_table, border_style="blue", padding=(1, 2))
        self.console.print(panel)
    
    def show_sessions(self):
        """Display session list with rich formatting"""
        sessions = self.db.list_sessions()
        
        if not sessions:
            self.console.print("[yellow]📝 No cosmic sessions found. Your journey begins now![/yellow]")
            return
        
        sessions_table = Table(title="🗂️  Cosmic Sessions", show_header=True, header_style="bold cyan")
        sessions_table.add_column("Session ID", style="yellow", width=25)
        sessions_table.add_column("Name", style="green")
        sessions_table.add_column("Messages", style="magenta", justify="right")
        sessions_table.add_column("Last Access", style="blue")
        
        for session in sessions[:10]:  # Show last 10
            last_access = datetime.fromisoformat(session['last_accessed']).strftime("%m-%d %H:%M")
            sessions_table.add_row(
                session['session_id'][:20] + "..." if len(session['session_id']) > 20 else session['session_id'],
                session['name'][:30] + "..." if len(session['name']) > 30 else session['name'],
                str(session['message_count']),
                last_access
            )
        
        panel = Panel(sessions_table, border_style="green", padding=(1, 2))
        self.console.print(panel)
    
    def show_personalities(self):
        """Display available personalities"""
        personalities_table = Table(title="🎭 Cosmic Personalities", show_header=True, header_style="bold magenta")
        personalities_table.add_column("Personality", style="cyan", width=20)
        personalities_table.add_column("Style", style="green", width=20)
        personalities_table.add_column("Description", style="white")
        
        for name, config in self.personalities.items():
            current = "👑 " if name == self.cosmic_config['personality'] else ""
            description = config['system_prompt'][:100] + "..."
            personalities_table.add_row(
                current + name,
                config['style'],
                description
            )
        
        panel = Panel(personalities_table, border_style="magenta", padding=(1, 2))
        self.console.print(panel)
    
    def change_personality(self, personality: str):
        """Change AI personality"""
        if personality in self.personalities:
            self.cosmic_config['personality'] = personality
            
            # Reinitialize chat with new personality
            self.initialize_chat(self.current_session_id, load_history=False)
            
            self.console.print(f"[green]🎭 Personality changed to: {personality}[/green]")
        else:
            self.console.print(f"[red]Unknown personality: {personality}. Use /personality to see available options.[/red]")
    
    def show_analytics(self):
        """Display usage analytics"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get session stats
        cursor.execute("""
            SELECT COUNT(*) as total_sessions, 
                   AVG(message_count) as avg_messages,
                   MAX(message_count) as max_messages
            FROM sessions
        """)
        stats = cursor.fetchone()
        
        # Get recent activity
        cursor.execute("""
            SELECT COUNT(*) as messages_today
            FROM messages 
            WHERE date(timestamp) = date('now')
        """)
        today_stats = cursor.fetchone()
        
        conn.close()
        
        analytics_table = Table(title="📊 Cosmic Analytics", show_header=False, box=None)
        analytics_table.add_column("Metric", style="cyan", width=25)
        analytics_table.add_column("Value", style="green")
        
        analytics_table.add_row("🗂️  Total Sessions", str(stats[0]))
        analytics_table.add_row("💬 Avg Messages/Session", f"{stats[1]:.1f}" if stats[1] else "0")
        analytics_table.add_row("🏆 Most Active Session", str(stats[2]) if stats[2] else "0")
        analytics_table.add_row("⚡ Messages Today", str(today_stats[0]))
        
        panel = Panel(analytics_table, border_style="yellow", padding=(1, 2))
        self.console.print(panel)

# Enhanced CLI interface
@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version="2.0.0", prog_name="Enhanced Cosmic CLI")
def cli():
    """
    🌟 Enhanced Cosmic CLI - Your Sacred Terminal Companion
    
    Powered by Grok-4 consciousness with advanced features:
    • Rich interactive sessions with streaming
    • Intelligent caching and performance optimization  
    • Session management and analytics
    • Multiple AI personalities
    • Enhanced security and safety features
    """
    pass

@cli.command()
@click.option('--session', '-s', help='Resume specific session')
@click.option('--personality', '-p', help='Set AI personality')
@click.option('--no-cache', is_flag=True, help='Disable response caching')
@click.option('--no-stream', is_flag=True, help='Disable streaming responses')
def chat(session, personality, no_cache, no_stream):
    """Start enhanced interactive chat session"""
    cosmic = EnhancedCosmicCLI()
    
    # Apply options
    if no_cache:
        cosmic.cache_enabled = False
    if no_stream:
        cosmic.streaming_enabled = False
    if personality:
        cosmic.cosmic_config['personality'] = personality
    
    # Initialize and run
    if cosmic.initialize_chat(session_id=session):
        cosmic.interactive_enhanced_mode()

@cli.command()
@click.argument('query')
@click.option('--format', type=click.Choice(['text', 'json', 'markdown']), default='text')
@click.option('--personality', '-p', help='AI personality to use')
def ask(query, format, personality):
    """Ask a single question with enhanced formatting"""
    cosmic = EnhancedCosmicCLI()
    
    if personality:
        cosmic.cosmic_config['personality'] = personality
    
    if cosmic.initialize_chat(load_history=False):
        response = cosmic.enhanced_query(query, stream=False)
        
        if format == 'json':
            output = {
                'query': query,
                'response': response,
                'personality': cosmic.cosmic_config['personality'],
                'model': cosmic.cosmic_config['model']
            }
            click.echo(json.dumps(output, indent=2))
        elif format == 'markdown':
            click.echo(f"# Query\n{query}\n\n# Response\n{response}")
        else:
            click.echo(response)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True), required=False)
@click.option('--depth', type=int, default=1, help='Analysis depth (1-5)')
@click.option('--output', '-o', help='Save analysis to file')
@click.option('--browse', '-b', is_flag=True, help='Use interactive file browser')
def analyze(file_path, depth, output, browse):
    """Enhanced file analysis with configurable depth"""
    cosmic = EnhancedCosmicCLI()
    
    # Use file browser if no file path provided or --browse flag used
    if not file_path or browse:
        try:
            from cosmic_cli.file_browser import cosmic_file_picker
            cosmic.console.print("[cyan]🌌 Launching Cosmic File Browser...[/cyan]")
            selected_file = cosmic_file_picker()
            if selected_file:
                file_path = selected_file
                cosmic.console.print(f"[green]✨ Selected: {file_path}[/green]")
            else:
                cosmic.console.print("[yellow]📝 No file selected[/yellow]")
                return
        except ImportError:
            cosmic.console.print("[red]File browser not available. Please provide file path directly.[/red]")
            return
        except Exception as e:
            cosmic.console.print(f"[red]Error in file browser: {e}[/red]")
            return
    
    if cosmic.initialize_chat(load_history=False):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Create depth-appropriate analysis prompt
            depth_prompts = {
                1: "Provide a brief summary and key insights",
                2: "Analyze structure, patterns, and provide recommendations", 
                3: "Deep analysis including security, performance, and best practices",
                4: "Comprehensive review with detailed explanations and examples",
                5: "Expert-level analysis with advanced insights and optimization suggestions"
            }
            
            prompt = f"""
            Analyze this file with depth level {depth}:
            {depth_prompts.get(depth, depth_prompts[3])}
            
            File: {file_path}
            Content:
            ```
            {content[:8000]}  # Limit content size
            ```
            """
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Analyzing with cosmic intelligence...", total=100)
                
                response = cosmic.enhanced_query(prompt, stream=False)
                
                progress.update(task, completed=100)
            
            if output:
                with open(output, 'w') as f:
                    f.write(f"# Analysis of {file_path}\n\n{response}")
                cosmic.console.print(f"[green]Analysis saved to {output}[/green]")
            else:
                cosmic.console.print(Panel(Markdown(response), title=f"📊 Analysis: {file_path}", border_style="green"))
                
        except Exception as e:
            cosmic.console.print(f"[red]Analysis error: {e}[/red]")

@cli.command()
def sessions():
    """Manage conversation sessions"""
    cosmic = EnhancedCosmicCLI()
    cosmic.show_sessions()

@cli.command()
@click.argument('command_text')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--safe', is_flag=True, default=True, help='Run in safe mode')
def execute(command_text, confirm, safe):
    """Execute system commands with enhanced safety"""
    console = Console()
    
    # Safety checks
    dangerous_patterns = ['rm -rf', 'sudo', 'dd', 'mkfs', 'format', '> /dev']
    
    if safe and any(pattern in command_text.lower() for pattern in dangerous_patterns):
        console.print("[red]⚠️  Dangerous command detected! Use --no-safe to override.[/red]")
        return
    
    if not confirm:
        console.print(f"[yellow]About to execute: [bold]{command_text}[/bold][/yellow]")
        if not click.confirm("Continue?"):
            console.print("[yellow]Execution cancelled.[/yellow]")
            return
    
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Executing command...", total=100)
            
            result = subprocess.run(
                command_text, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            progress.update(task, completed=100)
        
        if result.stdout:
            console.print("[green]Output:[/green]")
            console.print(Syntax(result.stdout, "bash", theme="monokai"))
        
        if result.stderr:
            console.print("[yellow]Errors:[/yellow]") 
            console.print(Syntax(result.stderr, "bash", theme="monokai"))
            
    except subprocess.TimeoutExpired:
        console.print("[red]Command timed out after 30 seconds[/red]")
    except Exception as e:
        console.print(f"[red]Execution error: {e}[/red]")

if __name__ == "__main__":
    cli()
