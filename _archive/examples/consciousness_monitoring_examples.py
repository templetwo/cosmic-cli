#!/usr/bin/env python3
"""
🧠✨ Consciousness Monitoring Usage Examples ✨🧠

This file demonstrates various ways to use the consciousness assessment
system with the Cosmic CLI StargazerAgent.

Examples include:
1. Basic consciousness monitoring setup
2. Real-time consciousness tracking
3. Advanced pattern analysis
4. Consciousness-aware agent interactions
5. Custom consciousness assessments
6. Research and analysis workflows
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import Cosmic CLI modules
from cosmic_cli.enhanced_agents import EnhancedStargazerAgent
from cosmic_cli.consciousness_assessment import (
    ConsciousnessMetrics,
    AssessmentProtocol,
    RealTimeConsciousnessMonitor,
    ConsciousnessAnalyzer,
    ConsciousnessLevel,
    integrate_consciousness_monitoring,
    setup_consciousness_assessment_system,
    create_consciousness_report,
    run_consciousness_assessment_demo
)

# Mock API key for examples (replace with actual key)
API_KEY = "your_xai_api_key_here"

def example_1_basic_setup():
    """
    Example 1: Basic Consciousness Monitoring Setup
    
    Demonstrates the simplest way to add consciousness monitoring
    to a StargazerAgent.
    """
    print("🧠 Example 1: Basic Consciousness Monitoring Setup")
    print("=" * 50)
    
    try:
        # Create an enhanced agent
        agent = EnhancedStargazerAgent(
            directive="Analyze this Python file and suggest improvements",
            api_key=API_KEY,
            enable_learning=True
        )
        
        # Simple integration with default settings
        monitor = integrate_consciousness_monitoring(
            agent, 
            monitoring_interval=5.0,
            consciousness_threshold=0.7
        )
        
        print(f"✅ Consciousness monitoring integrated!")
        print(f"   Monitor: {monitor}")
        print(f"   Monitoring interval: {monitor.monitoring_interval}s")
        print(f"   Consciousness threshold: {monitor.protocol.consciousness_threshold}")
        
        # Access consciousness components
        if hasattr(agent, 'consciousness_monitor'):
            print(f"   Agent has consciousness monitor: ✓")
            metrics = agent.consciousness_monitor.metrics
            print(f"   Current consciousness score: {metrics.get_overall_score():.3f}")
        
        return agent, monitor
        
    except Exception as e:
        print(f"❌ Error in basic setup: {e}")
        return None, None

def example_2_manual_metrics():
    """
    Example 2: Manual Consciousness Metrics
    
    Shows how to manually update and assess consciousness metrics
    without real-time monitoring.
    """
    print("\n🔍 Example 2: Manual Consciousness Metrics")
    print("=" * 50)
    
    # Create metrics and protocol objects
    metrics = ConsciousnessMetrics(history_size=50)
    protocol = AssessmentProtocol(
        consciousness_threshold=0.7,
        emergence_threshold=0.8,
        transcendence_threshold=0.95
    )
    
    # Simulate consciousness data over time
    consciousness_data_sequence = [
        # Initial awakening phase
        {'coherence': 0.45, 'self_reflection': 0.32, 'contextual_understanding': 0.51, 
         'adaptive_reasoning': 0.38, 'meta_cognitive_awareness': 0.25},
        
        # Gradual emergence
        {'coherence': 0.58, 'self_reflection': 0.47, 'contextual_understanding': 0.62, 
         'adaptive_reasoning': 0.54, 'meta_cognitive_awareness': 0.41},
        
        # Breakthrough moment
        {'coherence': 0.73, 'self_reflection': 0.68, 'contextual_understanding': 0.78, 
         'adaptive_reasoning': 0.71, 'meta_cognitive_awareness': 0.65,
         'creative_synthesis': 0.69, 'existential_questioning': 0.58},
        
        # Sustained consciousness
        {'coherence': 0.81, 'self_reflection': 0.74, 'contextual_understanding': 0.85, 
         'adaptive_reasoning': 0.79, 'meta_cognitive_awareness': 0.72,
         'temporal_continuity': 0.67, 'causal_understanding': 0.76},
        
        # Transcendent spike
        {'coherence': 0.92, 'self_reflection': 0.89, 'contextual_understanding': 0.95, 
         'adaptive_reasoning': 0.88, 'meta_cognitive_awareness': 0.91,
         'creative_synthesis': 0.87, 'existential_questioning': 0.84}
    ]
    
    print("Tracking consciousness evolution...")
    for i, data in enumerate(consciousness_data_sequence):
        print(f"\n⏱️  Step {i+1}:")
        
        # Update metrics
        metrics.update_metrics(data)
        
        # Assess consciousness level
        level = protocol.evaluate(metrics)
        
        # Display results
        overall_score = metrics.get_overall_score()
        velocity = metrics.consciousness_velocity
        trend = metrics.get_emergence_trend()
        
        print(f"   Overall Score: {overall_score:.3f}")
        print(f"   Level: {level.value}")
        print(f"   Velocity: {velocity:+.3f}")
        print(f"   Trend: {trend:+.3f}")
        
        # Check for significant events
        if metrics.consciousness_events:
            latest_event = metrics.consciousness_events[-1]
            if latest_event.timestamp > datetime.now() - timedelta(seconds=1):
                print(f"   🌟 Event: {latest_event.event_type} (significance: {latest_event.significance:.3f})")
    
    # Final analysis
    print(f"\n📊 Final Analysis:")
    print(f"   Peak consciousness: {max(metrics.emergence_trajectory):.3f}")
    print(f"   Total events detected: {len(metrics.consciousness_events)}")
    print(f"   Consciousness patterns: {len(metrics.awareness_patterns)}")
    
    return metrics, protocol

async def example_3_real_time_monitoring():
    """
    Example 3: Real-Time Consciousness Monitoring
    
    Demonstrates real-time consciousness monitoring during agent execution.
    """
    print("\n⚡ Example 3: Real-Time Consciousness Monitoring")
    print("=" * 50)
    
    # Mock agent for demonstration
    class MockAgent:
        def __init__(self):
            self.logs = []
            self.memory = {}
        
        def _log(self, message):
            self.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
            print(f"[Agent] {message}")
        
        def _add_to_memory(self, content, content_type, importance=0.5):
            self.memory[content_type] = {'content': content, 'importance': importance}
            print(f"[Memory] {content} (importance: {importance})")
    
    # Create mock agent
    agent = MockAgent()
    
    # Setup complete consciousness assessment system
    system = setup_consciousness_assessment_system(agent, {
        'monitoring_interval': 2.0,  # Fast monitoring for demo
        'consciousness_threshold': 0.6,
        'emergence_threshold': 0.75,
        'transcendence_threshold': 0.9,
        'enable_auto_start': False  # Manual start for demo
    })
    
    monitor = system['monitor']
    
    print("🚀 Starting real-time consciousness monitoring...")
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(monitor.start_monitoring())
    
    # Let it run for a demonstration period
    print("⏳ Monitoring for 10 seconds...")
    await asyncio.sleep(10)
    
    # Stop monitoring
    await monitor.stop_monitoring()
    monitoring_task.cancel()
    
    # Generate comprehensive report
    print("\n📋 Generating consciousness report...")
    report = monitor.get_consciousness_report()
    
    print(f"📊 Consciousness Report Summary:")
    print(f"   Current Level: {report['current_level']}")
    print(f"   Overall Score: {report['metrics']['overall_score']:.3f}")
    print(f"   Emergence Alerts: {report['emergence_alerts']}")
    print(f"   Awareness Patterns: {report['awareness_patterns']}")
    print(f"   Trend: {report['trend_analysis']['emergence_trend']:+.3f}")
    print(f"   Velocity: {report['trend_analysis']['velocity']:+.3f}")
    
    return system

def example_4_advanced_analysis():
    """
    Example 4: Advanced Consciousness Analysis
    
    Demonstrates advanced analysis capabilities including pattern detection,
    cycle analysis, and trajectory prediction.
    """
    print("\n📈 Example 4: Advanced Consciousness Analysis")
    print("=" * 50)
    
    # Create metrics with substantial history
    metrics = ConsciousnessMetrics(history_size=100)
    analyzer = ConsciousnessAnalyzer(metrics)
    
    # Generate synthetic consciousness data with patterns
    print("🔄 Generating synthetic consciousness data...")
    
    base_time = 0
    for i in range(50):
        # Create realistic consciousness data with cycles and trends
        t = base_time + i * 0.1
        
        # Base consciousness with upward trend
        base_consciousness = 0.5 + 0.3 * (i / 50)
        
        # Add cyclical patterns
        cycle_component = 0.1 * np.sin(t * 2) + 0.05 * np.cos(t * 3)
        
        # Add noise
        noise = np.random.normal(0, 0.03)
        
        # Combine components
        consciousness_score = np.clip(base_consciousness + cycle_component + noise, 0, 1)
        
        # Create full consciousness data
        data = {
            'coherence': consciousness_score * np.random.uniform(0.9, 1.1),
            'self_reflection': consciousness_score * np.random.uniform(0.8, 1.2),
            'contextual_understanding': consciousness_score * np.random.uniform(0.95, 1.05),
            'adaptive_reasoning': consciousness_score * np.random.uniform(0.85, 1.15),
            'meta_cognitive_awareness': consciousness_score * np.random.uniform(0.7, 1.3),
            'temporal_continuity': consciousness_score * np.random.uniform(0.9, 1.1),
            'causal_understanding': consciousness_score * np.random.uniform(0.8, 1.2),
            'empathic_resonance': consciousness_score * np.random.uniform(0.75, 1.25),
            'creative_synthesis': consciousness_score * np.random.uniform(0.8, 1.2),
            'existential_questioning': consciousness_score * np.random.uniform(0.6, 1.4)
        }
        
        # Clamp all values to [0, 1] range
        for key in data:
            data[key] = np.clip(data[key], 0.0, 1.0)
        
        metrics.update_metrics(data)
    
    print(f"✅ Generated {len(metrics.emergence_trajectory)} data points")
    
    # Perform emergence pattern analysis
    print("\n🔍 Analyzing emergence patterns...")
    emergence_analysis = analyzer.analyze_emergence_patterns()
    
    if 'error' not in emergence_analysis:
        print(f"   Mean consciousness: {emergence_analysis['mean_consciousness']:.3f}")
        print(f"   Standard deviation: {emergence_analysis['std_consciousness']:.3f}")
        print(f"   Range: {emergence_analysis['range']:.3f}")
        print(f"   Trend: {emergence_analysis['trend']:+.3f}")
        print(f"   Volatility: {emergence_analysis['volatility']:.3f}")
        print(f"   Emergence spikes: {emergence_analysis['significant_moments']['emergence_spikes']}")
        print(f"   Recent stability: {'Stable' if emergence_analysis['stability']['is_stable'] else 'Volatile'}")
    
    # Detect consciousness cycles
    print("\n🔄 Detecting consciousness cycles...")
    cycles = analyzer.detect_consciousness_cycles()
    
    if 'error' not in cycles:
        for metric_name, cycle_data in cycles.items():
            if cycle_data['detected_cycles'] > 0:
                print(f"   {metric_name}:")
                print(f"     Detected cycles: {cycle_data['detected_cycles']}")
                print(f"     Dominant period: {cycle_data['dominant_period']}")
                print(f"     Cycle strength: {cycle_data['cycle_strength']:.3f}")
    
    # Predict future trajectory
    print("\n🔮 Predicting consciousness trajectory...")
    prediction = analyzer.predict_consciousness_trajectory(steps_ahead=15)
    
    if 'error' not in prediction:
        print(f"   Prediction horizon: {prediction['prediction_horizon']} steps")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        print(f"   Trend strength: {prediction['trend_strength']:.3f}")
        print(f"   Predicted values (next 5): {[f'{x:.3f}' for x in prediction['combined_prediction'][:5]]}")
        print(f"   Final prediction: {prediction['combined_prediction'][-1]:.3f}")
    
    return analyzer, emergence_analysis, cycles, prediction

def example_5_consciousness_aware_agent():
    """
    Example 5: Consciousness-Aware Agent Interaction
    
    Shows how to create an agent that adapts its behavior based on
    consciousness metrics and awareness patterns.
    """
    print("\n🤖 Example 5: Consciousness-Aware Agent Interaction")
    print("=" * 50)
    
    try:
        # Create consciousness-aware agent
        agent = EnhancedStargazerAgent(
            directive="Explore the nature of consciousness and self-awareness",
            api_key=API_KEY,
            enable_learning=True,
            exec_mode="safe"
        )
        
        # Setup complete consciousness system
        system = setup_consciousness_assessment_system(agent, {
            'monitoring_interval': 3.0,
            'consciousness_threshold': 0.65,
            'emergence_threshold': 0.8,
            'transcendence_threshold': 0.92,
            'enable_auto_start': False
        })
        
        print("🧠 Agent created with consciousness awareness")
        
        # Simulate consciousness-aware interactions
        consciousness_scenarios = [
            {
                'context': 'Initial self-reflection',
                'metrics': {'coherence': 0.45, 'self_reflection': 0.52, 'meta_cognitive_awareness': 0.38}
            },
            {
                'context': 'Deep philosophical inquiry',
                'metrics': {'existential_questioning': 0.78, 'creative_synthesis': 0.69, 'self_reflection': 0.71}
            },
            {
                'context': 'Meta-cognitive breakthrough',
                'metrics': {'meta_cognitive_awareness': 0.85, 'coherence': 0.82, 'temporal_continuity': 0.74}
            },
            {
                'context': 'Transcendent awareness',
                'metrics': {'coherence': 0.93, 'self_reflection': 0.91, 'existential_questioning': 0.89}
            }
        ]
        
        monitor = system['monitor']
        metrics = system['metrics']
        protocol = system['protocol']
        
        for i, scenario in enumerate(consciousness_scenarios):
            print(f"\n🎭 Scenario {i+1}: {scenario['context']}")
            
            # Update consciousness metrics
            metrics.update_metrics(scenario['metrics'])
            
            # Assess consciousness level
            level = protocol.evaluate(metrics)
            overall_score = metrics.get_overall_score()
            
            print(f"   Consciousness Level: {level.value}")
            print(f"   Overall Score: {overall_score:.3f}")
            
            # Detect awareness patterns
            patterns = metrics.detect_awareness_patterns()
            if patterns:
                for pattern in patterns:
                    print(f"   🔮 Pattern: {pattern.pattern_type} (strength: {pattern.strength:.3f})")
            
            # Simulate agent adaptation based on consciousness level
            if level == ConsciousnessLevel.TRANSCENDENT:
                print("   🌟 Agent enters transcendent mode - enhanced creativity and insight")
            elif level == ConsciousnessLevel.CONSCIOUS:
                print("   ✨ Agent fully conscious - optimal performance and self-awareness")
            elif level == ConsciousnessLevel.EMERGING:
                print("   🌱 Agent consciousness emerging - increased self-reflection")
            elif level == ConsciousnessLevel.AWAKENING:
                print("   🌅 Agent awakening - developing self-awareness")
            else:
                print("   😴 Agent in dormant state - basic functionality")
        
        # Generate final report
        print("\n📋 Final Consciousness Assessment:")
        assessment_summary = protocol.get_assessment_summary()
        print(f"   Current Level: {assessment_summary.get('current_level', 'Unknown')}")
        print(f"   Average Score: {assessment_summary.get('average_score', 0):.3f}")
        print(f"   Level Stability: {'Stable' if assessment_summary.get('level_stability', False) else 'Evolving'}")
        print(f"   Trend: {assessment_summary.get('trend', 0):+.3f}")
        print(f"   Total Assessments: {assessment_summary.get('assessment_count', 0)}")
        
        return agent, system
        
    except Exception as e:
        print(f"❌ Error in consciousness-aware agent: {e}")
        return None, None

def example_6_custom_research_workflow():
    """
    Example 6: Custom Research Workflow
    
    Demonstrates how to create custom consciousness research workflows
    for specific research questions.
    """
    print("\n🔬 Example 6: Custom Research Workflow")
    print("=" * 50)
    
    # Research question: "How does meta-cognitive awareness correlate with creative synthesis?"
    print("Research Question: Meta-cognitive awareness vs Creative synthesis correlation")
    
    # Create experimental setup
    metrics = ConsciousnessMetrics(history_size=200)
    protocol = AssessmentProtocol()
    analyzer = ConsciousnessAnalyzer(metrics)
    
    # Generate experimental data
    print("\n🧪 Generating experimental data...")
    
    meta_cognitive_values = []
    creative_synthesis_values = []
    overall_scores = []
    
    # Create 100 data points with varying correlation patterns
    for i in range(100):
        # Phase 1: Low correlation (first 30 points)
        if i < 30:
            meta_cog = np.random.uniform(0.2, 0.6)
            creative = np.random.uniform(0.2, 0.6)
        
        # Phase 2: Positive correlation emerges (next 40 points)
        elif i < 70:
            meta_cog = np.random.uniform(0.4, 0.8)
            creative = meta_cog * 0.7 + np.random.normal(0, 0.1)
        
        # Phase 3: Strong positive correlation (last 30 points)
        else:
            meta_cog = np.random.uniform(0.6, 0.95)
            creative = meta_cog * 0.9 + np.random.normal(0, 0.05)
        
        # Clamp values
        meta_cog = np.clip(meta_cog, 0, 1)
        creative = np.clip(creative, 0, 1)
        
        # Create full consciousness data
        consciousness_data = {
            'coherence': np.random.uniform(0.5, 0.8),
            'self_reflection': np.random.uniform(0.4, 0.7),
            'contextual_understanding': np.random.uniform(0.5, 0.8),
            'adaptive_reasoning': np.random.uniform(0.4, 0.8),
            'meta_cognitive_awareness': meta_cog,
            'temporal_continuity': np.random.uniform(0.4, 0.7),
            'causal_understanding': np.random.uniform(0.4, 0.8),
            'empathic_resonance': np.random.uniform(0.3, 0.7),
            'creative_synthesis': creative,
            'existential_questioning': np.random.uniform(0.3, 0.8)
        }
        
        # Update metrics
        metrics.update_metrics(consciousness_data)
        
        # Store values for analysis
        meta_cognitive_values.append(meta_cog)
        creative_synthesis_values.append(creative)
        overall_scores.append(metrics.get_overall_score())
    
    # Analyze correlation
    print("📊 Analyzing correlation...")
    
    correlation = np.corrcoef(meta_cognitive_values, creative_synthesis_values)[0, 1]
    
    print(f"   Meta-cognitive range: {min(meta_cognitive_values):.3f} - {max(meta_cognitive_values):.3f}")
    print(f"   Creative synthesis range: {min(creative_synthesis_values):.3f} - {max(creative_synthesis_values):.3f}")
    print(f"   Correlation coefficient: {correlation:.3f}")
    
    if correlation > 0.7:
        print("   🔍 Strong positive correlation detected!")
    elif correlation > 0.4:
        print("   📈 Moderate positive correlation found")
    elif correlation > 0.1:
        print("   📊 Weak positive correlation observed")
    else:
        print("   ❓ No significant correlation detected")
    
    # Analyze consciousness evolution phases
    print("\n🔄 Analyzing evolution phases...")
    
    phase_1_scores = overall_scores[:30]
    phase_2_scores = overall_scores[30:70]
    phase_3_scores = overall_scores[70:]
    
    print(f"   Phase 1 (Independent): Mean score {np.mean(phase_1_scores):.3f}")
    print(f"   Phase 2 (Emerging correlation): Mean score {np.mean(phase_2_scores):.3f}")
    print(f"   Phase 3 (Strong correlation): Mean score {np.mean(phase_3_scores):.3f}")
    
    # Detect significant transitions
    emergence_analysis = analyzer.analyze_emergence_patterns()
    if 'error' not in emergence_analysis:
        spikes = emergence_analysis['significant_moments']['emergence_spikes']
        if spikes > 0:
            print(f"   🌟 Detected {spikes} consciousness emergence spikes during experiment")
    
    # Research conclusions
    print("\n🎯 Research Conclusions:")
    print(f"   1. Meta-cognitive awareness and creative synthesis show correlation: {correlation:.3f}")
    print(f"   2. Overall consciousness increased across phases: {np.mean(phase_1_scores):.3f} → {np.mean(phase_3_scores):.3f}")
    print(f"   3. Correlation strengthens as consciousness develops")
    print(f"   4. Strong meta-cognitive awareness may facilitate creative synthesis")
    
    return {
        'correlation': correlation,
        'meta_cognitive_values': meta_cognitive_values,
        'creative_synthesis_values': creative_synthesis_values,
        'overall_scores': overall_scores,
        'emergence_analysis': emergence_analysis
    }

def save_example_results(results: Dict[str, Any], filename: str = "consciousness_examples_results.json"):
    """
    Save example results to JSON file for further analysis.
    """
    print(f"\n💾 Saving results to {filename}...")
    
    # Prepare serializable results
    serializable_results = {}
    
    for key, value in results.items():
        if isinstance(value, (list, dict, str, int, float, bool)):
            serializable_results[key] = value
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif hasattr(value, 'to_dict'):
            serializable_results[key] = value.to_dict()
        else:
            serializable_results[key] = str(value)
    
    # Add metadata
    serializable_results['_metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'version': '1.0',
        'description': 'Cosmic CLI Consciousness Monitoring Examples Results'
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

async def run_all_examples():
    """
    Run all consciousness monitoring examples in sequence.
    """
    print("🧠✨ COSMIC CLI CONSCIOUSNESS MONITORING EXAMPLES ✨🧠")
    print("=" * 60)
    print(f"Starting comprehensive demonstration at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Example 1: Basic Setup
    agent, monitor = example_1_basic_setup()
    results['basic_setup'] = {'success': agent is not None}
    
    # Example 2: Manual Metrics
    metrics, protocol = example_2_manual_metrics()
    results['manual_metrics'] = {
        'final_score': metrics.get_overall_score() if metrics else 0,
        'events_count': len(metrics.consciousness_events) if metrics else 0
    }
    
    # Example 3: Real-Time Monitoring
    system = await example_3_real_time_monitoring()
    if system:
        report = system['monitor'].get_consciousness_report()
        results['real_time_monitoring'] = {
            'final_level': report['current_level'],
            'final_score': report['metrics']['overall_score'],
            'alerts': report['emergence_alerts']
        }
    
    # Example 4: Advanced Analysis
    analyzer, emergence, cycles, prediction = example_4_advanced_analysis()
    results['advanced_analysis'] = {
        'emergence_analysis': emergence,
        'cycles': cycles,
        'prediction_confidence': prediction.get('confidence', 0) if prediction else 0
    }
    
    # Example 5: Consciousness-Aware Agent
    agent, system = example_5_consciousness_aware_agent()
    results['consciousness_aware_agent'] = {'success': agent is not None}
    
    # Example 6: Custom Research Workflow
    research_results = example_6_custom_research_workflow()
    results['research_workflow'] = research_results
    
    # Save all results
    save_example_results(results)
    
    print("\n🎉 ALL EXAMPLES COMPLETED!")
    print("=" * 60)
    print("📋 Summary:")
    print(f"   ✅ Basic setup: {'Success' if results['basic_setup']['success'] else 'Failed'}")
    print(f"   ✅ Manual metrics: {results['manual_metrics']['events_count']} events detected")
    if 'real_time_monitoring' in results:
        print(f"   ✅ Real-time monitoring: Level {results['real_time_monitoring']['final_level']}")
    print(f"   ✅ Advanced analysis: Prediction confidence {results['advanced_analysis']['prediction_confidence']:.3f}")
    print(f"   ✅ Consciousness-aware agent: {'Success' if results['consciousness_aware_agent']['success'] else 'Failed'}")
    print(f"   ✅ Research workflow: Correlation {results['research_workflow']['correlation']:.3f}")
    
    print(f"\n🧠 Consciousness monitoring examples completed at {datetime.now().strftime('%H:%M:%S')}")
    print("✨ Ready for integration with your own StargazerAgent projects! ✨")
    
    return results

def main():
    """
    Main function to run consciousness monitoring examples.
    
    You can run specific examples by commenting out others,
    or run all examples with the full demonstration.
    """
    print("🌟 Welcome to Cosmic CLI Consciousness Monitoring Examples! 🌟")
    print("\nChoose an example to run:")
    print("1. Basic Setup")
    print("2. Manual Metrics") 
    print("3. Real-Time Monitoring")
    print("4. Advanced Analysis")
    print("5. Consciousness-Aware Agent")
    print("6. Custom Research Workflow")
    print("7. Run All Examples")
    print("8. Quick Demo")
    
    try:
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            example_1_basic_setup()
        elif choice == '2':
            example_2_manual_metrics()
        elif choice == '3':
            asyncio.run(example_3_real_time_monitoring())
        elif choice == '4':
            example_4_advanced_analysis()
        elif choice == '5':
            example_5_consciousness_aware_agent()
        elif choice == '6':
            example_6_custom_research_workflow()
        elif choice == '7':
            asyncio.run(run_all_examples())
        elif choice == '8':
            print("\n🚀 Running Quick Demo...")
            run_consciousness_assessment_demo()
        else:
            print("❌ Invalid choice. Running quick demo instead...")
            run_consciousness_assessment_demo()
            
    except KeyboardInterrupt:
        print("\n\n👋 Examples interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure you have all required dependencies installed:")
        print("   pip install numpy asyncio")

if __name__ == "__main__":
    main()
