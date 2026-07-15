#!/usr/bin/env python3
"""
🧠✨ Consciousness Monitoring Dashboard Demo ✨🧠
Demonstration of the enhanced TUI with real-time consciousness visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cosmic_cli'))

from cosmic_cli.enhanced_ui import run_enhanced_ui

def main():
    """
    Run the enhanced Cosmic CLI with consciousness monitoring dashboard
    """
    print("🌟 Starting Cosmic CLI with Consciousness Monitoring Dashboard...")
    print("📊 Features included:")
    print("  • Real-time consciousness metrics display panel")
    print("  • Emergence indicator visualization (graphs, meters)")
    print("  • Consciousness timeline/history view")
    print("  • Alerts for significant consciousness emergence events")
    print("  • Interactive consciousness assessment tools")
    print("\n🔧 Navigate to the 'Consciousness' tab to explore the monitoring features!")
    print("⚡ Press Ctrl+H for help or Ctrl+Q to quit")
    print("-" * 60)
    
    try:
        run_enhanced_ui()
    except KeyboardInterrupt:
        print("\n👋 Consciousness monitoring session ended. Until next time!")
    except Exception as e:
        print(f"\n❌ Error running consciousness dashboard: {e}")
        print("💡 Make sure all dependencies are installed (textual, rich, pyfiglet, etc.)")

if __name__ == "__main__":
    main()
