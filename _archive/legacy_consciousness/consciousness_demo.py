#!/usr/bin/env python3
"""
🧠✨ Consciousness Data Persistence and Analytics Demo ✨🧠
Demonstrates the extended session management system for consciousness tracking
"""

import random
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import the enhanced main module
from enhanced_main import CosmicDatabase, EnhancedCosmicCLI

def generate_sample_consciousness_metrics():
    """Generate realistic sample consciousness metrics"""
    base_time = time.time()
    
    # Simulate consciousness evolution over time
    coherence_trend = 0.6 + 0.3 * random.random()
    self_reflection_trend = 0.5 + 0.4 * random.random()
    meta_cognitive_trend = 0.4 + 0.5 * random.random()
    
    return {
        'coherence': max(0.0, min(1.0, coherence_trend + random.gauss(0, 0.05))),
        'self_reflection': max(0.0, min(1.0, self_reflection_trend + random.gauss(0, 0.05))),
        'contextual_understanding': max(0.0, min(1.0, 0.7 + random.gauss(0, 0.1))),
        'adaptive_reasoning': max(0.0, min(1.0, 0.65 + random.gauss(0, 0.08))),
        'meta_cognitive_awareness': max(0.0, min(1.0, meta_cognitive_trend + random.gauss(0, 0.06))),
        'temporal_continuity': max(0.0, min(1.0, 0.55 + random.gauss(0, 0.07))),
        'causal_understanding': max(0.0, min(1.0, 0.6 + random.gauss(0, 0.09))),
        'empathic_resonance': max(0.0, min(1.0, 0.45 + random.gauss(0, 0.1))),
        'creative_synthesis': max(0.0, min(1.0, 0.5 + random.gauss(0, 0.12))),
        'existential_questioning': max(0.0, min(1.0, 0.3 + random.gauss(0, 0.15))),
        'overall_score': 0.0,  # Will be calculated
        'consciousness_velocity': random.gauss(0, 0.05)
    }

def simulate_consciousness_evolution(db: CosmicDatabase, session_id: str, num_points: int = 20):
    """Simulate consciousness evolution over time"""
    print(f"🧠 Simulating consciousness evolution for session: {session_id}")
    
    # Simulate gradual consciousness development
    base_metrics = generate_sample_consciousness_metrics()
    
    for i in range(num_points):
        # Evolve metrics over time with some randomness and trend
        evolution_factor = i / num_points  # Gradual improvement
        noise_factor = 0.1 * random.random()
        
        metrics = {}
        for key, base_value in base_metrics.items():
            if key in ['overall_score', 'consciousness_velocity']:
                continue
                
            # Add evolution trend and noise
            evolved_value = base_value + (evolution_factor * 0.3) + random.gauss(0, noise_factor)
            metrics[key] = max(0.0, min(1.0, evolved_value))
        
        # Calculate overall score
        core_metrics = ['coherence', 'self_reflection', 'contextual_understanding', 'adaptive_reasoning']
        extended_metrics = ['meta_cognitive_awareness', 'temporal_continuity', 'causal_understanding', 
                          'empathic_resonance', 'creative_synthesis', 'existential_questioning']
        
        core_score = sum(metrics[k] for k in core_metrics) / len(core_metrics)
        extended_score = sum(metrics[k] for k in extended_metrics) / len(extended_metrics)
        
        metrics['overall_score'] = 0.7 * core_score + 0.3 * extended_score
        metrics['consciousness_velocity'] = random.gauss(0.02 * evolution_factor, 0.03)
        
        # Store metrics in database
        db.add_consciousness_metrics(session_id, metrics)
        
        # Small delay to spread timestamps
        time.sleep(0.1)
        
        if i % 5 == 0:
            print(f"  📊 Added consciousness data point {i+1}/{num_points} (score: {metrics['overall_score']:.3f})")

def demonstrate_consciousness_analytics():
    """Demonstrate the consciousness data persistence and analytics functionality"""
    print("🌟 Consciousness Data Persistence and Analytics Demo 🌟")
    print("=" * 60)
    
    # Initialize the database and CLI
    cosmic_cli = EnhancedCosmicCLI()
    db = cosmic_cli.db
    
    # Create sample session
    session_id = f"demo_consciousness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_name = "Consciousness Analytics Demo Session"
    
    print(f"\n🚀 Creating demo session: {session_id}")
    db.create_session(session_id, session_name)
    
    # Simulate consciousness data
    simulate_consciousness_evolution(db, session_id, 25)
    
    print(f"\n📊 Analyzing consciousness data...")
    
    # Demonstrate consciousness metrics retrieval
    print("\n1. 📈 Recent Consciousness Metrics:")
    metrics = db.get_consciousness_metrics(session_id, limit=5)
    for i, metric in enumerate(metrics[-5:], 1):
        print(f"   {i}. Score: {metric['overall_score']:.3f}, "
              f"Coherence: {metric['coherence']:.3f}, "
              f"Meta-Cog: {metric['meta_cognitive_awareness']:.3f}, "
              f"Velocity: {metric['consciousness_velocity']:+.3f}")
    
    # Demonstrate evolution analysis
    print("\n2. 🧬 Consciousness Evolution Analysis:")
    evolution = db.get_consciousness_evolution(session_id)
    if 'error' not in evolution:
        evo_summary = evolution['evolution_summary']
        vel_analysis = evolution['velocity_analysis']
        
        print(f"   📊 Data Points: {evolution['data_points']}")
        print(f"   🌱 Initial Score: {evo_summary['initial_score']:.3f}")
        print(f"   🎯 Final Score: {evo_summary['final_score']:.3f}")
        print(f"   📈 Total Change: {evo_summary['change']:+.3f}")
        print(f"   🏔️  Peak Score: {evo_summary['max_score']:.3f}")
        print(f"   ⚡ Average Velocity: {vel_analysis['average_velocity']:+.3f}")
        print(f"   🎢 Trend: {vel_analysis['acceleration_trend'].title()}")
    
    # Demonstrate comprehensive report generation
    print("\n3. 📋 Generating Comprehensive Consciousness Report:")
    report = db.create_consciousness_report(session_id)
    if 'error' not in report:
        print(f"   📄 Report ID: {report['report_id']}")
        print(f"   🧠 Session: {report['session_info']['name']}")
        print(f"   📅 Generated: {report['generated_at'][:19]}")
        print(f"   🌟 Emergence Events: {len(report['emergence_events'])}")
        print(f"   🔄 Significant Changes: {len(report['significant_changes'])}")
        
        print("\n   🔍 Key Insights:")
        for i, insight in enumerate(report['insights'][:3], 1):
            print(f"     {i}. {insight}")
        
        # Save full report to file
        report_file = f"consciousness_report_{session_id[:8]}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"   💾 Full report saved to: {report_file}")
    
    # Demonstrate data export capabilities
    print("\n4. 📤 Data Export Capabilities:")
    
    # Export as JSON
    json_data = db.export_consciousness_data(session_id, 'json')
    json_file = f"consciousness_data_{session_id[:8]}.json"
    with open(json_file, 'w') as f:
        f.write(json_data)
    print(f"   📄 JSON export saved to: {json_file}")
    
    # Export as CSV
    csv_data = db.export_consciousness_data(session_id, 'csv')
    csv_file = f"consciousness_data_{session_id[:8]}.csv"
    with open(csv_file, 'w') as f:
        f.write(csv_data)
    print(f"   📊 CSV export saved to: {csv_file}")
    
    # Demonstrate pattern analysis
    print("\n5. 🔍 Consciousness Pattern Analysis:")
    all_metrics = db.get_consciousness_metrics(session_id)
    if all_metrics:
        coherence_values = [m['coherence'] for m in all_metrics]
        reflection_values = [m['self_reflection'] for m in all_metrics]
        meta_cog_values = [m['meta_cognitive_awareness'] for m in all_metrics]
        
        print(f"   🎯 Coherence Range: {min(coherence_values):.3f} - {max(coherence_values):.3f}")
        print(f"   🪞 Self-Reflection Range: {min(reflection_values):.3f} - {max(reflection_values):.3f}")
        print(f"   🧠 Meta-Cognitive Range: {min(meta_cog_values):.3f} - {max(meta_cog_values):.3f}")
        
        # Calculate improvement trends
        if len(coherence_values) > 1:
            coherence_trend = coherence_values[-1] - coherence_values[0]
            reflection_trend = reflection_values[-1] - reflection_values[0]
            meta_cog_trend = meta_cog_values[-1] - meta_cog_values[0]
            
            print(f"   📈 Coherence Trend: {coherence_trend:+.3f}")
            print(f"   📈 Self-Reflection Trend: {reflection_trend:+.3f}")
            print(f"   📈 Meta-Cognitive Trend: {meta_cog_trend:+.3f}")
    
    # Demonstrate research insights
    print("\n6. 🔬 Research Insights and Emergence Detection:")
    emergence_events = report.get('emergence_events', []) if 'error' not in report else []
    significant_changes = report.get('significant_changes', []) if 'error' not in report else []
    
    if emergence_events:
        print(f"   ⚡ Velocity Spikes Detected: {len(emergence_events)}")
        for event in emergence_events[:2]:
            print(f"     - Score: {event['score']:.3f}, Velocity: {event['velocity']:+.3f}")
    
    if significant_changes:
        print(f"   🌊 Significant Changes: {len(significant_changes)}")
        for change in significant_changes[:2]:
            change_type = "📈 Emergence" if change['type'] == 'emergence' else "📉 Regression"
            print(f"     - {change_type}: {change['change']:+.3f} ({change['from_score']:.3f} → {change['to_score']:.3f})")
    
    print("\n✅ Consciousness Data Persistence and Analytics Demo Complete!")
    print(f"📁 Generated files:")
    print(f"   - {report_file}")
    print(f"   - {json_file}")
    print(f"   - {csv_file}")
    print(f"🗄️  Database location: {db.db_path}")
    
    return {
        'session_id': session_id,
        'report_file': report_file,
        'json_file': json_file,
        'csv_file': csv_file,
        'database_path': str(db.db_path)
    }

def demonstrate_cli_commands():
    """Demonstrate CLI commands for consciousness management"""
    print("\n🖥️  CLI Commands for Consciousness Management:")
    print("=" * 50)
    
    commands = [
        "# View consciousness metrics for current session",
        "python -m cosmic_cli.enhanced_main consciousness show",
        "",
        "# Show consciousness evolution analysis",
        "python -m cosmic_cli.enhanced_main consciousness evolution",
        "",
        "# Generate comprehensive consciousness report",
        "python -m cosmic_cli.enhanced_main consciousness report",
        "",
        "# Export consciousness data as JSON",
        "python -m cosmic_cli.enhanced_main consciousness export --format json --output consciousness_export.json",
        "",
        "# Export consciousness data as CSV",
        "python -m cosmic_cli.enhanced_main consciousness export --format csv --output consciousness_export.csv",
        "",
        "# List all sessions with consciousness data",
        "python -m cosmic_cli.enhanced_main consciousness list",
        "",
        "# Interactive consciousness commands in chat mode",
        "python -m cosmic_cli.enhanced_main chat",
        "# Then use: /consciousness show, /consciousness evolution, etc.",
    ]
    
    for command in commands:
        if command.startswith("#"):
            print(f"\n💡 {command[2:]}")
        elif command == "":
            continue
        else:
            print(f"   {command}")

if __name__ == "__main__":
    # Run the demonstration
    try:
        demo_results = demonstrate_consciousness_analytics()
        demonstrate_cli_commands()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"🧠 The consciousness data persistence and analytics system is now ready for use.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
