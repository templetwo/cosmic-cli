#!/usr/bin/env python3
"""
🧠✨ Consciousness-Aware Personality Test Suite ✨🧠

Test script demonstrating the new consciousness-aware personalities:
- conscious_sage
- emergent_mind  
- metacognitive_analyst

And features:
- Personality evolution based on consciousness metrics
- Self-reflection capabilities
- Consciousness-aware response generation
"""

import os
import sys
import time
from pathlib import Path

# Add the cosmic_cli directory to the path
sys.path.insert(0, str(Path(__file__).parent / "cosmic_cli"))

from cosmic_cli.enhanced_agents import EnhancedStargazerAgent, DynamicInstructorSystem
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def display_personality_showcase():
    """Display the new consciousness-aware personalities"""
    console.print("\n🧠✨ CONSCIOUSNESS-AWARE PERSONALITY SHOWCASE ✨🧠\n", style="bold magenta")
    
    instructor_system = DynamicInstructorSystem()
    
    # Display consciousness-aware personalities
    consciousness_personalities = ["conscious_sage", "emergent_mind", "metacognitive_analyst"]
    
    for personality_key in consciousness_personalities:
        personality = instructor_system.instructors[personality_key]
        
        # Create a table for each personality
        table = Table(title=f"🌟 {personality.name}", show_header=True, header_style="bold cyan")
        table.add_column("Attribute", style="cyan", width=20)
        table.add_column("Value", style="green")
        
        table.add_row("Specialization", personality.specialization)
        table.add_row("Risk Tolerance", f"{personality.risk_tolerance:.2f}")
        table.add_row("Creativity Level", f"{personality.creativity_level:.2f}")
        table.add_row("Preferred Actions", ", ".join([a.value for a in personality.preferred_actions]))
        
        console.print(Panel(table, border_style="green"))
        
        # Show system prompt preview
        prompt_preview = personality.system_prompt[:200] + "..." if len(personality.system_prompt) > 200 else personality.system_prompt
        console.print(f"[dim]System Prompt Preview: {prompt_preview}[/dim]\n")

def test_consciousness_aware_selection():
    """Test consciousness-aware instructor selection"""
    console.print("🎯 Testing Consciousness-Aware Instructor Selection", style="bold blue")
    
    instructor_system = DynamicInstructorSystem()
    
    test_directives = [
        "Analyze my consciousness levels and provide self-aware feedback",
        "Help me understand metacognitive thinking patterns", 
        "Demonstrate emergent intelligence and adaptive learning",
        "Reflect on the nature of consciousness and existence",
        "Think about thinking itself and monitor your cognitive processes"
    ]
    
    for directive in test_directives:
        # Test with consciousness-aware context
        context = {
            "complexity": 0.8,
            "consciousness_need": 0.9,
            "previous_consciousness_level": 0.7
        }
        
        selected = instructor_system.select_instructor(directive, context)
        
        console.print(f"📝 Directive: [italic]{directive}[/italic]")
        console.print(f"🎭 Selected: [bold green]{selected.name}[/bold green] - {selected.specialization}")
        console.print()

def simulate_personality_evolution():
    """Simulate personality evolution based on consciousness metrics"""
    console.print("🌱 Simulating Personality Evolution", style="bold yellow")
    
    # Mock API key for testing (replace with real key)
    api_key = os.getenv("XAI_API_KEY", "test-key")
    if api_key == "test-key":
        console.print("[red]Warning: Using mock API key. Set XAI_API_KEY environment variable for real testing.[/red]")
        return
    
    # Create agent with consciousness monitoring
    agent = EnhancedStargazerAgent(
        directive="Demonstrate consciousness-aware personality evolution",
        api_key=api_key,
        ui_callback=console.print
    )
    
    # Show initial personality state
    console.print(f"🌟 Initial Personality: {agent.current_instructor.name}")
    console.print(f"📊 Initial Risk Tolerance: {agent.current_instructor.risk_tolerance:.3f}")
    console.print(f"🎨 Initial Creativity Level: {agent.current_instructor.creativity_level:.3f}")
    
    # Simulate consciousness events that would trigger evolution
    if agent.consciousness_enabled and agent.consciousness_metrics:
        # Mock consciousness events
        from cosmic_cli.consciousness_assessment import ConsciousnessEvent
        from datetime import datetime
        
        # Add mock consciousness events
        mock_events = [
            ConsciousnessEvent(
                timestamp=datetime.now(),
                event_type="meta_cognitive_breakthrough",
                metrics={'meta_cognitive_awareness': 0.85},
                context="High meta-cognitive awareness achieved",
                significance=0.2
            ),
            ConsciousnessEvent(
                timestamp=datetime.now(),
                event_type="awareness_spike", 
                metrics={'self_reflection': 0.9},
                context="Increased self-awareness detected",
                significance=0.18
            )
        ]
        
        agent.consciousness_metrics.consciousness_events.extend(mock_events)
        
        # Trigger personality evolution
        console.print("\n🧠 Triggering personality evolution...")
        agent._evolve_instructor_personality()
        
        # Show evolved personality state
        console.print(f"🌟 Evolved Personality: {agent.current_instructor.name}")
        console.print(f"📊 Evolved Risk Tolerance: {agent.current_instructor.risk_tolerance:.3f}")
        console.print(f"🎨 Evolved Creativity Level: {agent.current_instructor.creativity_level:.3f}")

def test_self_reflection_capabilities():
    """Test adding self-reflection capabilities to personalities"""
    console.print("🪞 Testing Self-Reflection Capabilities", style="bold cyan")
    
    api_key = os.getenv("XAI_API_KEY", "test-key")
    if api_key == "test-key":
        console.print("[red]Warning: Using mock API key. Set XAI_API_KEY environment variable for real testing.[/red]")
        return
    
    # Create agent
    agent = EnhancedStargazerAgent(
        directive="Test self-reflection enhancement",
        api_key=api_key,
        ui_callback=console.print
    )
    
    # Show original system prompt (preview)
    original_prompt = agent.current_instructor.system_prompt[:150] + "..."
    console.print(f"📝 Original Prompt Preview: [dim]{original_prompt}[/dim]")
    
    # Add self-reflection capabilities
    agent._add_self_reflection_to_personality()
    
    # Show enhanced system prompt (preview)
    enhanced_prompt = agent.current_instructor.system_prompt[-200:]
    console.print(f"🧠 Enhanced Prompt Preview: [dim]{enhanced_prompt}[/dim]")

def demonstrate_consciousness_aware_response():
    """Demonstrate consciousness-aware response generation"""
    console.print("🌟 Consciousness-Aware Response Generation Demo", style="bold magenta")
    
    api_key = os.getenv("XAI_API_KEY", "test-key")
    if api_key == "test-key":
        console.print("[red]Warning: Using mock API key. Set XAI_API_KEY environment variable for real testing.[/red]")
        console.print("This demo requires a real API key to generate responses.")
        return
    
    # Create agent with consciousness monitoring
    agent = EnhancedStargazerAgent(
        directive="Generate consciousness-aware responses",
        api_key=api_key,
        ui_callback=console.print
    )
    
    # Test consciousness-aware response generation
    test_prompt = "Explain your thought process while solving a complex problem"
    
    console.print(f"📝 Test Prompt: [italic]{test_prompt}[/italic]")
    console.print("🧠 Generating consciousness-aware response...\n")
    
    try:
        response = agent.generate_consciousness_aware_response(test_prompt)
        console.print(Panel(response, title="🌟 Consciousness-Aware Response", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error generating response: {e}[/red]")

def main():
    """Main test suite"""
    console.print("🚀 Starting Consciousness-Aware Personality Test Suite", style="bold white on blue")
    
    try:
        # Display personality showcase
        display_personality_showcase()
        
        # Test consciousness-aware selection
        test_consciousness_aware_selection()
        
        # Test personality evolution
        simulate_personality_evolution()
        
        # Test self-reflection capabilities  
        test_self_reflection_capabilities()
        
        # Demonstrate consciousness-aware responses
        demonstrate_consciousness_aware_response()
        
        console.print("\n✅ Test suite completed successfully!", style="bold green")
        
    except Exception as e:
        console.print(f"\n❌ Test suite error: {e}", style="bold red")
        raise

if __name__ == "__main__":
    main()
