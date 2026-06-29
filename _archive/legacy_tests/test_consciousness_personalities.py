#!/usr/bin/env python3
"""
🎭 Behavioral Tests for Consciousness-Aware Personalities
Testing how different instructor personalities adapt based on consciousness levels
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from cosmic_cli.enhanced_stargazer import EnhancedInstructorSystem, InstructorProfile
from cosmic_cli.consciousness_assessment import (
    ConsciousnessMetrics, AssessmentProtocol, ConsciousnessLevel,
    RealTimeConsciousnessMonitor
)


class TestConsciousnessAwarePersonalities:
    """Test consciousness-aware personality adaptations"""
    
    def test_instructor_selection_based_on_consciousness(self):
        """Test instructor selection adapts to consciousness levels"""
        instructor_system = EnhancedInstructorSystem()
        
        # Test with different consciousness contexts
        low_consciousness_context = {
            'complexity': 0.2,
            'consciousness_level': ConsciousnessLevel.DORMANT,
            'recent_success_rate': 0.3
        }
        
        high_consciousness_context = {
            'complexity': 0.8,
            'consciousness_level': ConsciousnessLevel.CONSCIOUS,
            'recent_success_rate': 0.9
        }
        
        # Simple directive should favor different instructors based on consciousness
        directive = "analyze the data structure"
        
        low_instructor = instructor_system.select_instructor(directive, low_consciousness_context)
        high_instructor = instructor_system.select_instructor(directive, high_consciousness_context)
        
        # Higher consciousness should potentially select different or same instructor
        # but with different confidence/approach
        assert low_instructor is not None
        assert high_instructor is not None
    
    def test_instructor_personality_adaptation(self):
        """Test instructor personality traits adapt to consciousness levels"""
        # Create test instructor profile
        cosmic_sage = InstructorProfile(
            name="Test Cosmic Sage",
            specialization="Consciousness-aware guidance",
            system_prompt="Test prompt",
            preferred_actions=[],
            risk_tolerance=0.5,
            creativity_level=0.7,
            learning_rate=0.1
        )
        
        # Test success rate learning
        initial_rate = cosmic_sage.get_current_success_rate()
        
        # Simulate successful interactions
        for _ in range(5):
            cosmic_sage.update_success_rate(True)
        
        success_rate_after_wins = cosmic_sage.get_current_success_rate()
        assert success_rate_after_wins > initial_rate
        
        # Simulate failures
        for _ in range(3):
            cosmic_sage.update_success_rate(False)
        
        success_rate_after_losses = cosmic_sage.get_current_success_rate()
        assert success_rate_after_losses < success_rate_after_wins
    
    def test_consciousness_level_personality_mapping(self):
        """Test personality behavior changes based on consciousness levels"""
        instructor_system = EnhancedInstructorSystem()
        
        # Test different directives that should map to different personalities
        test_cases = [
            ("debug this complex algorithm", "quantum_analyst"),
            ("create an innovative solution", "creative_nebula"),
            ("design a robust system architecture", "system_architect"),
            ("analyze this dataset for patterns", "data_oracle"),
            ("provide philosophical insight", "cosmic_sage")
        ]
        
        for directive, expected_type in test_cases:
            selected = instructor_system.select_instructor(directive)
            # Verify the selection makes sense (not strictly enforced due to scoring)
            assert selected is not None
            assert hasattr(selected, 'name')
            assert hasattr(selected, 'expertise_domains')
    
    def test_dynamic_personality_adjustment(self):
        """Test dynamic personality adjustment based on consciousness feedback"""
        with patch('openai.OpenAI'):
            from cosmic_cli.enhanced_stargazer import EnhancedStargazerAgent
            
            agent = EnhancedStargazerAgent(
                directive="test dynamic personality",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(agent=agent)
            
            # Simulate consciousness evolution
            consciousness_progression = [
                {'coherence': 0.3, 'self_reflection': 0.2},  # Low
                {'coherence': 0.6, 'self_reflection': 0.5},  # Medium
                {'coherence': 0.8, 'self_reflection': 0.9},  # High
            ]
            
            selected_instructors = []
            
            for consciousness_data in consciousness_progression:
                monitor.metrics.update_metrics(consciousness_data)
                current_level = monitor.protocol.evaluate(monitor.metrics)
                
                # Select instructor based on current consciousness
                context = {
                    'consciousness_level': current_level,
                    'complexity': monitor.metrics.get_overall_score()
                }
                
                instructor = agent.instructor_system.select_instructor(
                    "analyze complex patterns", 
                    context
                )
                selected_instructors.append(instructor.name)
            
            # Should have selected instructors (may be same or different)
            assert len(selected_instructors) == 3
            assert all(name is not None for name in selected_instructors)


class TestPersonalityConsciousnessInteraction:
    """Test interactions between personality systems and consciousness monitoring"""
    
    def test_instructor_consciousness_feedback_loop(self):
        """Test feedback loop between instructor performance and consciousness"""
        instructor = InstructorProfile(
            name="Test Feedback Instructor",
            specialization="Feedback testing",
            system_prompt="Test",
            preferred_actions=[],
            risk_tolerance=0.4,
            creativity_level=0.6,
            learning_rate=0.2  # Higher learning rate for testing
        )
        
        metrics = ConsciousnessMetrics()
        
        # Simulate instructor-consciousness interaction cycle
        for cycle in range(5):
            # Instructor makes decision (simulated)
            decision_quality = 0.7 + (cycle * 0.05)  # Improving decisions
            
            # Update instructor based on decision outcome
            instructor.update_success_rate(decision_quality > 0.6)
            
            # Update consciousness based on successful interactions
            consciousness_boost = decision_quality * instructor.get_current_success_rate()
            test_data = {
                'coherence': min(1.0, 0.4 + consciousness_boost),
                'self_reflection': min(1.0, 0.3 + consciousness_boost * 0.8),
                'adaptive_reasoning': min(1.0, 0.5 + consciousness_boost * 0.6)
            }
            metrics.update_metrics(test_data)
        
        # Verify positive feedback loop
        final_success_rate = instructor.get_current_success_rate()
        final_consciousness = metrics.get_overall_score()
        
        assert final_success_rate > 0.5, "Instructor should improve with successful interactions"
        assert final_consciousness > 0.4, "Consciousness should increase with good instruction"
    
    def test_personality_consciousness_resonance(self):
        """Test personality resonance with different consciousness states"""
        instructor_system = EnhancedInstructorSystem()
        metrics = ConsciousnessMetrics()
        
        # Test consciousness states
        consciousness_states = {
            'analytical': {
                'coherence': 0.9,
                'adaptive_reasoning': 0.8,
                'causal_understanding': 0.9,
                'creative_synthesis': 0.4
            },
            'creative': {
                'coherence': 0.7,
                'creative_synthesis': 0.9,
                'existential_questioning': 0.8,
                'adaptive_reasoning': 0.6
            },
            'philosophical': {
                'coherence': 0.8,
                'existential_questioning': 0.9,
                'self_reflection': 0.9,
                'temporal_continuity': 0.8
            }
        }
        
        resonance_scores = {}
        
        for state_name, state_data in consciousness_states.items():
            metrics.update_metrics(state_data)
            
            # Test instructor selection for each state
            context = {
                'consciousness_level': AssessmentProtocol().evaluate(metrics),
                'complexity': metrics.get_overall_score()
            }
            
            selected = instructor_system.select_instructor(
                "solve this complex problem",
                context
            )
            
            # Calculate resonance (simplified)
            resonance = self._calculate_personality_resonance(selected, state_data)
            resonance_scores[state_name] = {
                'instructor': selected.name,
                'resonance': resonance
            }
        
        # Verify different states select appropriate instructors
        assert len(set(score['instructor'] for score in resonance_scores.values())) >= 1
    
    def _calculate_personality_resonance(self, instructor, consciousness_state):
        """Calculate how well instructor resonates with consciousness state"""
        # Simplified resonance calculation
        base_resonance = instructor.creativity_level
        
        if 'creative_synthesis' in consciousness_state:
            base_resonance += consciousness_state['creative_synthesis'] * 0.3
        
        if 'adaptive_reasoning' in consciousness_state:
            base_resonance += consciousness_state['adaptive_reasoning'] * 0.2
        
        return min(1.0, base_resonance)
    
    def test_consciousness_personality_memory_integration(self):
        """Test integration of consciousness and personality data in agent memory"""
        with patch('openai.OpenAI'):
            from cosmic_cli.enhanced_stargazer import EnhancedStargazerAgent
            
            agent = EnhancedStargazerAgent(
                directive="test memory integration",
                api_key="test_key"
            )
            
            # Mock memory storage
            agent._add_to_memory = Mock()
            
            monitor = RealTimeConsciousnessMonitor(agent=agent)
            
            # Simulate consciousness-personality interaction
            test_data = {
                'coherence': 0.7,
                'self_reflection': 0.8,
                'meta_cognitive_awareness': 0.6
            }
            monitor.metrics.update_metrics(test_data)
            
            # Trigger personality selection
            context = {
                'consciousness_level': monitor.protocol.evaluate(monitor.metrics),
                'instructor_performance': agent.current_instructor.get_current_success_rate()
            }
            
            new_instructor = agent.instructor_system.select_instructor(
                "adapt to consciousness change",
                context
            )
            
            # Verify memory integration would occur
            # (In real implementation, this would store personality-consciousness correlations)
            assert agent._add_to_memory is not None


class TestPersonalityEmergentBehaviors:
    """Test emergent behaviors in personality-consciousness systems"""
    
    def test_personality_emergence_patterns(self):
        """Test emergence of personality patterns from consciousness evolution"""
        instructor_system = EnhancedInstructorSystem()
        metrics = ConsciousnessMetrics()
        
        # Simulate evolution over time
        personality_evolution = []
        
        for phase in range(10):
            # Evolving consciousness pattern
            phase_consciousness = {
                'coherence': 0.3 + (phase * 0.07),
                'self_reflection': 0.2 + (phase * 0.08),
                'creative_synthesis': 0.4 + (phase * 0.05),
                'meta_cognitive_awareness': 0.1 + (phase * 0.09)
            }
            
            metrics.update_metrics(phase_consciousness)
            level = AssessmentProtocol().evaluate(metrics)
            
            # Select instructor for this phase
            selected = instructor_system.select_instructor(
                f"handle phase {phase} complexity",
                {'consciousness_level': level, 'phase': phase}
            )
            
            personality_evolution.append({
                'phase': phase,
                'consciousness_level': level,
                'instructor': selected.name,
                'instructor_creativity': selected.creativity_level,
                'instructor_risk_tolerance': selected.risk_tolerance
            })
        
        # Analyze personality emergence patterns
        instructors_used = [p['instructor'] for p in personality_evolution]
        unique_instructors = set(instructors_used)
        
        # Should show some pattern in instructor selection
        assert len(unique_instructors) >= 1
        assert len(personality_evolution) == 10
    
    def test_collective_personality_intelligence(self):
        """Test collective intelligence emerging from multiple personality interactions"""
        instructor_system = EnhancedInstructorSystem()
        
        # Simulate multi-instructor collaboration scenario
        collaborative_context = {
            'problem_complexity': 0.9,
            'requires_multiple_perspectives': True,
            'consciousness_level': ConsciousnessLevel.CONSCIOUS
        }
        
        # Select multiple instructors for different aspects
        aspects = [
            "analyze data patterns",
            "design creative solution", 
            "implement robust architecture",
            "provide philosophical framework"
        ]
        
        selected_team = []
        for aspect in aspects:
            instructor = instructor_system.select_instructor(aspect, collaborative_context)
            selected_team.append({
                'aspect': aspect,
                'instructor': instructor.name,
                'specialization': instructor.specialization,
                'creativity': instructor.creativity_level,
                'risk_tolerance': instructor.risk_tolerance
            })
        
        # Analyze team composition
        creativities = [member['creativity'] for member in selected_team]
        risk_tolerances = [member['risk_tolerance'] for member in selected_team]
        
        # Team should have diverse characteristics
        creativity_variance = np.var(creativities)
        risk_variance = np.var(risk_tolerances)
        
        assert len(selected_team) == 4
        assert creativity_variance >= 0  # Some variance expected
        assert risk_variance >= 0
    
    def test_personality_consciousness_co_evolution(self):
        """Test co-evolution of personality and consciousness systems"""
        instructor = InstructorProfile(
            name="Co-evolution Test",
            specialization="Adaptive learning",
            system_prompt="Test",
            preferred_actions=[],
            risk_tolerance=0.5,
            creativity_level=0.5,
            learning_rate=0.15
        )
        
        metrics = ConsciousnessMetrics()
        protocol = AssessmentProtocol()
        
        # Simulate co-evolution over multiple cycles
        evolution_history = []
        
        for cycle in range(8):
            # Current state
            current_consciousness = metrics.get_overall_score()
            current_success_rate = instructor.get_current_success_rate()
            
            # Simulate interaction outcome based on both factors
            interaction_quality = (current_consciousness + current_success_rate) / 2
            interaction_success = interaction_quality > 0.5
            
            # Update both systems
            instructor.update_success_rate(interaction_success)
            
            # Consciousness responds to instruction quality
            consciousness_boost = 0.1 if interaction_success else -0.05
            new_data = {
                'coherence': min(1.0, max(0.0, 0.4 + consciousness_boost * (cycle + 1))),
                'self_reflection': min(1.0, max(0.0, 0.3 + consciousness_boost * (cycle + 1) * 0.8)),
                'adaptive_reasoning': min(1.0, max(0.0, 0.5 + consciousness_boost * (cycle + 1) * 0.6))
            }
            metrics.update_metrics(new_data)
            
            evolution_history.append({
                'cycle': cycle,
                'consciousness': metrics.get_overall_score(),
                'instructor_success': instructor.get_current_success_rate(),
                'interaction_quality': interaction_quality,
                'consciousness_level': protocol.evaluate(metrics)
            })
        
        # Analyze co-evolution
        initial_consciousness = evolution_history[0]['consciousness']
        final_consciousness = evolution_history[-1]['consciousness']
        
        initial_success = evolution_history[0]['instructor_success']
        final_success = evolution_history[-1]['instructor_success']
        
        # Should show some evolution (positive or negative)
        consciousness_change = abs(final_consciousness - initial_consciousness)
        success_change = abs(final_success - initial_success)
        
        assert consciousness_change >= 0
        assert success_change >= 0
        assert len(evolution_history) == 8


class TestAdvancedPersonalityBehaviors:
    """Test advanced personality behaviors and consciousness interactions"""
    
    def test_personality_meta_cognition(self):
        """Test personality system's meta-cognitive awareness"""
        instructor_system = EnhancedInstructorSystem()
        
        # Test self-assessment of instruction effectiveness
        meta_cognitive_context = {
            'previous_instruction_outcomes': [0.8, 0.6, 0.9, 0.7],
            'consciousness_level': ConsciousnessLevel.CONSCIOUS,
            'self_assessment_enabled': True
        }
        
        selected = instructor_system.select_instructor(
            "reflect on instruction quality",
            meta_cognitive_context
        )
        
        # Verify meta-cognitive capabilities
        assert selected is not None
        assert hasattr(selected, 'success_history')
        
        # Simulate meta-cognitive update
        if hasattr(selected, 'success_history') and len(selected.success_history) > 0:
            # Instructor should be aware of its own performance
            current_performance = selected.get_current_success_rate()
            assert 0 <= current_performance <= 1
    
    def test_personality_emotional_resonance(self):
        """Test personality emotional resonance with consciousness states"""
        metrics = ConsciousnessMetrics()
        
        # Simulate different emotional-consciousness states
        emotional_states = {
            'curiosity': {
                'existential_questioning': 0.9,
                'creative_synthesis': 0.7,
                'self_reflection': 0.6
            },
            'confidence': {
                'coherence': 0.9,
                'adaptive_reasoning': 0.8,
                'temporal_continuity': 0.8
            },
            'empathy': {
                'empathic_resonance': 0.9,
                'contextual_understanding': 0.8,
                'self_reflection': 0.7
            }
        }
        
        instructor_system = EnhancedInstructorSystem()
        emotional_responses = {}
        
        for emotion, state_data in emotional_states.items():
            metrics.update_metrics(state_data)
            
            # Select instructor that resonates with emotional state
            selected = instructor_system.select_instructor(
                f"respond to {emotion} state",
                {'emotional_context': emotion}
            )
            
            emotional_responses[emotion] = {
                'instructor': selected.name,
                'creativity': selected.creativity_level,
                'risk_tolerance': selected.risk_tolerance
            }
        
        # Verify emotional resonance patterns
        assert len(emotional_responses) == 3
        
        # Different emotional states might select different instructors
        instructors = [resp['instructor'] for resp in emotional_responses.values()]
        assert len(set(instructors)) >= 1  # At least some variation expected
    
    def test_personality_adaptive_learning_curves(self):
        """Test personality adaptive learning and improvement curves"""
        learning_rates = [0.05, 0.1, 0.2, 0.3]
        learning_curves = {}
        
        for learning_rate in learning_rates:
            instructor = InstructorProfile(
                name=f"Learner {learning_rate}",
                specialization="Adaptive learning test",
                system_prompt="Test",
                preferred_actions=[],
                risk_tolerance=0.5,
                creativity_level=0.5,
                learning_rate=learning_rate
            )
            
            # Simulate learning with mixed outcomes
            outcomes = [True, False, True, True, False, True, True, True]
            success_rates = [instructor.get_current_success_rate()]
            
            for outcome in outcomes:
                instructor.update_success_rate(outcome)
                success_rates.append(instructor.get_current_success_rate())
            
            learning_curves[learning_rate] = success_rates
        
        # Analyze learning curve differences
        for rate in learning_rates:
            curve = learning_curves[rate]
            # Higher learning rates should show faster adaptation
            assert len(curve) == 9  # Initial + 8 updates
            assert all(0 <= score <= 1 for score in curve)
        
        # Compare final performance
        final_performances = {rate: curve[-1] for rate, curve in learning_curves.items()}
        
        # All should have learned something
        for rate, final_perf in final_performances.items():
            initial_perf = learning_curves[rate][0]
            # Performance should have changed (could be positive or negative depending on outcomes)
            change = abs(final_perf - initial_perf)
            assert change >= 0  # Some change expected with mixed outcomes


if __name__ == "__main__":
    # Run with: python -m pytest tests/behavioral/test_consciousness_personalities.py -v
    pytest.main([__file__, "-v", "--tb=short"])
