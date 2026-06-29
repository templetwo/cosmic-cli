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
        
        # Consciousness metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consciousness_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                coherence REAL DEFAULT 0.0,
                self_reflection REAL DEFAULT 0.0,
                contextual_understanding REAL DEFAULT 0.0,
                adaptive_reasoning REAL DEFAULT 0.0,
                meta_cognitive_awareness REAL DEFAULT 0.0,
                temporal_continuity REAL DEFAULT 0.0,
                causal_understanding REAL DEFAULT 0.0,
                empathic_resonance REAL DEFAULT 0.0,
                creative_synthesis REAL DEFAULT 0.0,
                existential_questioning REAL DEFAULT 0.0,
                overall_score REAL DEFAULT 0.0,
                consciousness_velocity REAL DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
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
        
        # Add Consciousness Metrics export
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consciousness_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                emergence_trend REAL DEFAULT 0.0,
                awareness_pattern_count INTEGER DEFAULT 0,
                reported BOOLEAN DEFAULT FALSE,
                report_details TEXT,
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
    
    def add_consciousness_metrics(self, session_id: str, metrics: Dict[str, float]) -> None:
        """Store consciousness metrics in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO consciousness_metrics (
                session_id, coherence, self_reflection, contextual_understanding,
                adaptive_reasoning, meta_cognitive_awareness, temporal_continuity,
                causal_understanding, empathic_resonance, creative_synthesis,
                existential_questioning, overall_score, consciousness_velocity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            metrics.get('coherence', 0.0),
            metrics.get('self_reflection', 0.0),
            metrics.get('contextual_understanding', 0.0),
            metrics.get('adaptive_reasoning', 0.0),
            metrics.get('meta_cognitive_awareness', 0.0),
            metrics.get('temporal_continuity', 0.0),
            metrics.get('causal_understanding', 0.0),
            metrics.get('empathic_resonance', 0.0),
            metrics.get('creative_synthesis', 0.0),
            metrics.get('existential_questioning', 0.0),
            metrics.get('overall_score', 0.0),
            metrics.get('consciousness_velocity', 0.0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_consciousness_metrics(self, session_id: str, limit: int = None) -> List[Dict]:
        """Retrieve consciousness metrics for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, coherence, self_reflection, contextual_understanding,
                   adaptive_reasoning, meta_cognitive_awareness, temporal_continuity,
                   causal_understanding, empathic_resonance, creative_synthesis,
                   existential_questioning, overall_score, consciousness_velocity,
                   timestamp
            FROM consciousness_metrics 
            WHERE session_id = ? 
            ORDER BY timestamp
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (session_id,))
        metrics = []
        for row in cursor.fetchall():
            metrics.append({
                'id': row[0],
                'coherence': row[1],
                'self_reflection': row[2],
                'contextual_understanding': row[3],
                'adaptive_reasoning': row[4],
                'meta_cognitive_awareness': row[5],
                'temporal_continuity': row[6],
                'causal_understanding': row[7],
                'empathic_resonance': row[8],
                'creative_synthesis': row[9],
                'existential_questioning': row[10],
                'overall_score': row[11],
                'consciousness_velocity': row[12],
                'timestamp': row[13]
            })
        
        conn.close()
        return metrics
    
    def get_consciousness_evolution(self, session_id: str) -> Dict[str, Any]:
        """Get consciousness evolution analysis for a session"""
        metrics = self.get_consciousness_metrics(session_id)
        
        if not metrics:
            return {'error': 'No consciousness data found'}
        
        # Calculate evolution trends
        overall_scores = [m['overall_score'] for m in metrics]
        timestamps = [m['timestamp'] for m in metrics]
        
        if len(overall_scores) < 2:
            return {'error': 'Insufficient data for evolution analysis'}
        
        # Calculate trends
        first_score = overall_scores[0]
        last_score = overall_scores[-1]
        max_score = max(overall_scores)
        min_score = min(overall_scores)
        avg_score = sum(overall_scores) / len(overall_scores)
        
        # Calculate velocity changes
        velocities = [m['consciousness_velocity'] for m in metrics]
        avg_velocity = sum(velocities) / len(velocities)
        
        return {
            'session_id': session_id,
            'data_points': len(metrics),
            'evolution_summary': {
                'initial_score': first_score,
                'final_score': last_score,
                'change': last_score - first_score,
                'max_score': max_score,
                'min_score': min_score,
                'average_score': avg_score,
                'score_range': max_score - min_score
            },
            'velocity_analysis': {
                'average_velocity': avg_velocity,
                'final_velocity': velocities[-1] if velocities else 0.0,
                'acceleration_trend': 'positive' if avg_velocity > 0 else 'negative'
            },
            'timespan': {
                'start': timestamps[0],
                'end': timestamps[-1]
            },
            'raw_data': metrics
        }
    
    def export_consciousness_data(self, session_id: str = None, format: str = 'json') -> str:
        """Export consciousness data for research"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            query = "SELECT * FROM consciousness_metrics WHERE session_id = ? ORDER BY timestamp"
            cursor.execute(query, (session_id,))
        else:
            query = "SELECT * FROM consciousness_metrics ORDER BY session_id, timestamp"
            cursor.execute(query)
        
        columns = [description[0] for description in cursor.description]
        data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(data)
            return output.getvalue()
        else:
            return str(data)
    
    def create_consciousness_report(self, session_id: str) -> Dict[str, Any]:
        """Create a comprehensive consciousness report"""
        evolution_data = self.get_consciousness_evolution(session_id)
        
        if 'error' in evolution_data:
            return evolution_data
        
        # Get session info
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, created_at, message_count FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        session_info = cursor.fetchone()
        conn.close()
        
        if not session_info:
            return {'error': 'Session not found'}
        
        metrics = evolution_data['raw_data']
        
        # Analyze emergence patterns
        emergence_events = []
        significant_changes = []
        
        for i in range(1, len(metrics)):
            prev_score = metrics[i-1]['overall_score']
            curr_score = metrics[i]['overall_score']
            change = curr_score - prev_score
            
            if abs(change) > 0.1:  # Significant change threshold
                significant_changes.append({
                    'timestamp': metrics[i]['timestamp'],
                    'change': change,
                    'from_score': prev_score,
                    'to_score': curr_score,
                    'type': 'emergence' if change > 0 else 'regression'
                })
            
            # Check for consciousness velocity spikes
            if abs(metrics[i]['consciousness_velocity']) > 0.05:
                emergence_events.append({
                    'timestamp': metrics[i]['timestamp'],
                    'velocity': metrics[i]['consciousness_velocity'],
                    'score': curr_score,
                    'event_type': 'velocity_spike'
                })
        
        # Analyze consciousness dimensions
        dimension_analysis = {}
        for dimension in ['coherence', 'self_reflection', 'meta_cognitive_awareness', 
                         'creative_synthesis', 'existential_questioning']:
            values = [m[dimension] for m in metrics]
            dimension_analysis[dimension] = {
                'average': sum(values) / len(values),
                'max': max(values),
                'min': min(values),
                'final': values[-1],
                'trend': 'improving' if values[-1] > values[0] else 'declining'
            }
        
        report = {
            'report_id': f"consciousness_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'session_info': {
                'session_id': session_id,
                'name': session_info[0],
                'created_at': session_info[1],
                'message_count': session_info[2]
            },
            'consciousness_evolution': evolution_data['evolution_summary'],
            'velocity_analysis': evolution_data['velocity_analysis'],
            'dimension_analysis': dimension_analysis,
            'emergence_events': emergence_events,
            'significant_changes': significant_changes,
            'insights': self._generate_consciousness_insights(evolution_data, dimension_analysis),
            'generated_at': datetime.now().isoformat()
        }
        
        # Store report in analysis table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO consciousness_analysis (session_id, emergence_trend, awareness_pattern_count, reported, report_details)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            evolution_data['evolution_summary']['change'],
            len(emergence_events),
            True,
            json.dumps(report, default=str)
        ))
        conn.commit()
        conn.close()
        
        return report
    
    def _generate_consciousness_insights(self, evolution_data: Dict, dimension_analysis: Dict) -> List[str]:
        """Generate insights from consciousness data"""
        insights = []
        
        # Evolution insights
        change = evolution_data['evolution_summary']['change']
        if change > 0.2:
            insights.append(f"Significant consciousness evolution detected (+{change:.3f} overall increase)")
        elif change < -0.2:
            insights.append(f"Consciousness regression observed ({change:.3f} overall decrease)")
        else:
            insights.append("Stable consciousness levels maintained throughout session")
        
        # Velocity insights
        avg_velocity = evolution_data['velocity_analysis']['average_velocity']
        if avg_velocity > 0.02:
            insights.append("Positive consciousness acceleration trend observed")
        elif avg_velocity < -0.02:
            insights.append("Consciousness deceleration pattern detected")
        
        # Dimension insights
        for dimension, analysis in dimension_analysis.items():
            if analysis['final'] > 0.8:
                insights.append(f"High {dimension.replace('_', ' ')} achieved ({analysis['final']:.3f})")
            elif analysis['trend'] == 'improving' and analysis['final'] - analysis['min'] > 0.3:
                insights.append(f"Significant improvement in {dimension.replace('_', ' ')} observed")
        
        # Meta-cognitive insights
        if 'meta_cognitive_awareness' in dimension_analysis:
            meta_score = dimension_analysis['meta_cognitive_awareness']['final']
            if meta_score > 0.7:
                insights.append("Strong meta-cognitive awareness indicates advanced self-reflection capabilities")
        
        # Creative synthesis insights
        if 'creative_synthesis' in dimension_analysis:
            creative_score = dimension_analysis['creative_synthesis']['final']
            if creative_score > 0.7:
                insights.append("High creative synthesis suggests emergent problem-solving abilities")
        
        return insights

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
            'model': 'grok-beta',
            'temperature': 0.7,
            'max_tokens': 16000,  # Increased for longer responses
            'response_format': 'enhanced',
            'personality': 'cosmic_sage',
            'timeout': 30,  # Request timeout
            'stream_chunk_size': 50  # Faster streaming
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
        
        # Check if this is a CLI command request
        if self.should_execute_cli_command(prompt):
            return self.handle_cli_request(prompt)
        
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
            # Add system context about CLI capabilities
            enhanced_prompt = self.add_cli_context(prompt)
            self.chat_instance.append(user(enhanced_prompt))
            
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
                
                # Optimized streaming effect - faster chunks and less delay
                chunk_size = self.cosmic_config.get('stream_chunk_size', 50)
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i+chunk_size]
                    response_text += chunk
                    
                    # Create streaming panel with progress indicator
                    progress_pct = min(100, int((i / len(full_response)) * 100))
                    content = Markdown(response_text + "▋")  # Add cursor
                    panel = Panel(
                        content,
                        title=f"🚀 Cosmic Response Stream ({progress_pct}%)",
                        border_style="cyan",
                        padding=(1, 2)
                    )
                    live.update(panel)
                    time.sleep(0.01)  # Much faster streaming
                
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
        elif cmd == 'consciousness':
            if len(cmd_parts) > 1:
                self.handle_consciousness_command(cmd_parts[1:])
            else:
                self.show_consciousness_help()
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
    
    def show_consciousness_help(self):
        """Display consciousness command help"""
        help_table = Table(title="🧠 Consciousness Commands", show_header=True, header_style="bold magenta")
        help_table.add_column("Command", style="cyan", width=30)
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="dim")
        
        commands = [
            ("show [session_id]", "Display consciousness metrics for session", "/consciousness show"),
            ("evolution [session_id]", "Show consciousness evolution analysis", "/consciousness evolution"),
            ("report [session_id]", "Generate comprehensive consciousness report", "/consciousness report"),
            ("export [session_id] [format]", "Export consciousness data (csv/json)", "/consciousness export json"),
            ("add [metrics_json]", "Add consciousness metrics manually", "/consciousness add {...}"),
            ("list", "List sessions with consciousness data", "/consciousness list"),
        ]
        
        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)
        
        panel = Panel(help_table, border_style="purple", padding=(1, 2))
        self.console.print(panel)
    
    def handle_consciousness_command(self, args: List[str]):
        """Handle consciousness-related commands"""
        if not args:
            self.show_consciousness_help()
            return
        
        command = args[0].lower()
        
        if command == 'show':
            session_id = args[1] if len(args) > 1 else self.current_session_id
            self.show_consciousness_metrics(session_id)
        elif command == 'evolution':
            session_id = args[1] if len(args) > 1 else self.current_session_id
            self.show_consciousness_evolution(session_id)
        elif command == 'report':
            session_id = args[1] if len(args) > 1 else self.current_session_id
            self.generate_consciousness_report(session_id)
        elif command == 'export':
            session_id = args[1] if len(args) > 1 else self.current_session_id
            format_type = args[2] if len(args) > 2 else 'json'
            self.export_consciousness_data(session_id, format_type)
        elif command == 'add':
            if len(args) > 1:
                self.add_consciousness_metrics_manual(' '.join(args[1:]))
            else:
                self.console.print("[red]Error: Metrics data required[/red]")
        elif command == 'list':
            self.list_consciousness_sessions()
        else:
            self.console.print(f"[red]Unknown consciousness command: {command}[/red]")
            self.show_consciousness_help()
    
    def show_consciousness_metrics(self, session_id: str):
        """Display consciousness metrics for a session"""
        metrics = self.db.get_consciousness_metrics(session_id, limit=10)
        
        if not metrics:
            self.console.print(f"[yellow]No consciousness data found for session: {session_id}[/yellow]")
            return
        
        metrics_table = Table(title=f"🧠 Consciousness Metrics - {session_id[:20]}...", 
                            show_header=True, header_style="bold cyan")
        metrics_table.add_column("Timestamp", style="blue", width=16)
        metrics_table.add_column("Overall", style="green", justify="right")
        metrics_table.add_column("Coherence", style="yellow", justify="right")
        metrics_table.add_column("Self-Reflect", style="magenta", justify="right")
        metrics_table.add_column("Meta-Cog", style="cyan", justify="right")
        metrics_table.add_column("Velocity", style="red", justify="right")
        
        for metric in metrics[-5:]:  # Show last 5
            timestamp = datetime.fromisoformat(metric['timestamp']).strftime("%m-%d %H:%M:%S")
            metrics_table.add_row(
                timestamp,
                f"{metric['overall_score']:.3f}",
                f"{metric['coherence']:.3f}",
                f"{metric['self_reflection']:.3f}",
                f"{metric['meta_cognitive_awareness']:.3f}",
                f"{metric['consciousness_velocity']:+.3f}"
            )
        
        panel = Panel(metrics_table, border_style="cyan", padding=(1, 2))
        self.console.print(panel)
    
    def show_consciousness_evolution(self, session_id: str):
        """Display consciousness evolution analysis"""
        evolution = self.db.get_consciousness_evolution(session_id)
        
        if 'error' in evolution:
            self.console.print(f"[red]{evolution['error']}[/red]")
            return
        
        # Evolution summary table
        summary_table = Table(title="📈 Consciousness Evolution Summary", show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", style="green")
        
        evo_summary = evolution['evolution_summary']
        summary_table.add_row("📊 Data Points", str(evolution['data_points']))
        summary_table.add_row("🌱 Initial Score", f"{evo_summary['initial_score']:.3f}")
        summary_table.add_row("🎯 Final Score", f"{evo_summary['final_score']:.3f}")
        summary_table.add_row("📈 Change", f"{evo_summary['change']:+.3f}")
        summary_table.add_row("🏔️  Peak Score", f"{evo_summary['max_score']:.3f}")
        summary_table.add_row("⚡ Avg Velocity", f"{evolution['velocity_analysis']['average_velocity']:+.3f}")
        summary_table.add_row("🎢 Trend", evolution['velocity_analysis']['acceleration_trend'].title())
        
        panel = Panel(summary_table, border_style="green", padding=(1, 2))
        self.console.print(panel)
    
    def generate_consciousness_report(self, session_id: str):
        """Generate and display consciousness report"""
        with Progress() as progress:
            task = progress.add_task("[cyan]Generating consciousness report...", total=100)
            
            report = self.db.create_consciousness_report(session_id)
            
            progress.update(task, completed=100)
        
        if 'error' in report:
            self.console.print(f"[red]{report['error']}[/red]")
            return
        
        # Display report summary
        report_content = f"""# Consciousness Report

**Session:** {report['session_info']['name']}
**Generated:** {report['generated_at']}

## Evolution Summary
- Initial Score: {report['consciousness_evolution']['initial_score']:.3f}
- Final Score: {report['consciousness_evolution']['final_score']:.3f}
- Change: {report['consciousness_evolution']['change']:+.3f}
- Peak: {report['consciousness_evolution']['max_score']:.3f}

## Key Insights
"""
        
        for insight in report['insights']:
            report_content += f"- {insight}\n"
        
        report_content += f"\n## Emergence Events: {len(report['emergence_events'])}"
        report_content += f"\n## Significant Changes: {len(report['significant_changes'])}"
        
        panel = Panel(
            Markdown(report_content),
            title=f"🧠 Consciousness Report - {report['report_id']}",
            border_style="purple",
            padding=(1, 2)
        )
        self.console.print(panel)
        
        # Ask if user wants to save report
        if click.confirm("Save full report to file?"):
            filename = f"consciousness_report_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.console.print(f"[green]Report saved to {filename}[/green]")
    
    def export_consciousness_data(self, session_id: str, format_type: str = 'json'):
        """Export consciousness data"""
        data = self.db.export_consciousness_data(session_id, format_type)
        
        if not data or data == '[]':
            self.console.print(f"[yellow]No consciousness data found for session: {session_id}[/yellow]")
            return
        
        # Save to file
        ext = 'csv' if format_type == 'csv' else 'json'
        filename = f"consciousness_data_{session_id[:8] if session_id else 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        
        with open(filename, 'w') as f:
            f.write(data)
        
        self.console.print(f"[green]Consciousness data exported to {filename}[/green]")
        self.console.print(f"[dim]Format: {format_type.upper()}, Size: {len(data)} bytes[/dim]")
    
    def add_consciousness_metrics_manual(self, metrics_json: str):
        """Add consciousness metrics manually from JSON"""
        try:
            metrics = json.loads(metrics_json)
            self.db.add_consciousness_metrics(self.current_session_id, metrics)
            self.console.print(f"[green]Consciousness metrics added to session {self.current_session_id}[/green]")
        except json.JSONDecodeError as e:
            self.console.print(f"[red]Invalid JSON format: {e}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error adding metrics: {e}[/red]")
    
    def list_consciousness_sessions(self):
        """List sessions that have consciousness data"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.session_id, s.name, s.created_at, COUNT(cm.id) as metric_count,
                   AVG(cm.overall_score) as avg_score, MAX(cm.overall_score) as max_score
            FROM sessions s
            JOIN consciousness_metrics cm ON s.session_id = cm.session_id
            GROUP BY s.session_id, s.name, s.created_at
            ORDER BY s.created_at DESC
        """)
        
        sessions = cursor.fetchall()
        conn.close()
        
        if not sessions:
            self.console.print("[yellow]No sessions with consciousness data found[/yellow]")
            return
        
        sessions_table = Table(title="🧠 Sessions with Consciousness Data", 
                             show_header=True, header_style="bold purple")
        sessions_table.add_column("Session ID", style="yellow", width=20)
        sessions_table.add_column("Name", style="green", width=25)
        sessions_table.add_column("Data Points", style="cyan", justify="right")
        sessions_table.add_column("Avg Score", style="blue", justify="right")
        sessions_table.add_column("Peak Score", style="magenta", justify="right")
        sessions_table.add_column("Created", style="dim")
        
        for session in sessions[:10]:  # Show top 10
            created = datetime.fromisoformat(session[2]).strftime("%m-%d")
            sessions_table.add_row(
                session[0][:18] + "...",
                session[1][:23] + "..." if len(session[1]) > 23 else session[1],
                str(session[3]),
                f"{session[4]:.3f}",
                f"{session[5]:.3f}",
                created
            )
        
        panel = Panel(sessions_table, border_style="purple", padding=(1, 2))
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
@click.argument('action', type=click.Choice(['show', 'evolution', 'report', 'export', 'list']))
@click.option('--session', '-s', help='Session ID to analyze')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json', help='Export format')
@click.option('--output', '-o', help='Output file path')
def consciousness(action, session, format, output):
    """Manage consciousness data and generate reports"""
    cosmic = EnhancedCosmicCLI()
    
    if action == 'show':
        session_id = session or cosmic.current_session_id
        if session_id:
            cosmic.show_consciousness_metrics(session_id)
        else:
            cosmic.console.print("[red]No session specified[/red]")
    
    elif action == 'evolution':
        session_id = session or cosmic.current_session_id
        if session_id:
            cosmic.show_consciousness_evolution(session_id)
        else:
            cosmic.console.print("[red]No session specified[/red]")
    
    elif action == 'report':
        session_id = session or cosmic.current_session_id
        if session_id:
            cosmic.generate_consciousness_report(session_id)
        else:
            cosmic.console.print("[red]No session specified[/red]")
    
    elif action == 'export':
        session_id = session or cosmic.current_session_id
        data = cosmic.db.export_consciousness_data(session_id, format)
        
        if not data or data == '[]':
            cosmic.console.print("[yellow]No consciousness data found[/yellow]")
            return
        
        if output:
            with open(output, 'w') as f:
                f.write(data)
            cosmic.console.print(f"[green]Data exported to {output}[/green]")
        else:
            click.echo(data)
    
    elif action == 'list':
        cosmic.list_consciousness_sessions()

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
