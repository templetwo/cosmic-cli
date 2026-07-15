#!/usr/bin/env python3
"""
🧠🧪 Consciousness Metrics Unit Tests
Unit tests for consciousness metrics calculations and tracking systems
"""

import pytest
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from cosmic_cli.consciousness_assessment import (
    ConsciousnessMetrics, ConsciousnessEvent, SelfAwarenessPattern,
    ConsciousnessLevel, ConsciousnessState, ConsciousnessAnalyzer
)


class TestConsciousnessMetrics:
    """Test consciousness metrics calculations and tracking"""
    
    def test_metrics_initialization(self):
        """Test consciousness metrics initialization"""
        metrics = ConsciousnessMetrics(history_size=50)
        
        # Test initial state
        assert metrics.coherence == 0.0
        assert metrics.self_reflection == 0.0
        assert metrics.contextual_understanding == 0.0
        assert metrics.adaptive_reasoning == 0.0
        assert metrics.meta_cognitive_awareness == 0.0
        assert metrics.temporal_continuity == 0.0
        assert metrics.causal_understanding == 0.0
        assert metrics.empathic_resonance == 0.0
        assert metrics.creative_synthesis == 0.0
        assert metrics.existential_questioning == 0.0
        
        # Test configuration
        assert metrics.history_size == 50
        assert isinstance(metrics.metrics_history, deque)
        assert len(metrics.consciousness_events) == 0
        assert len(metrics.awareness_patterns) == 0
        assert metrics.consciousness_velocity == 0.0
    
    def test_basic_metric_updates(self):
        """Test basic metric updates"""
        metrics = ConsciousnessMetrics()
        
        test_data = {
            'coherence': 0.7,
            'self_reflection': 0.6,
            'contextual_understanding': 0.8,
            'adaptive_reasoning': 0.5,
            'meta_cognitive_awareness': 0.4,
            'temporal_continuity': 0.9,
            'causal_understanding': 0.3,
            'empathic_resonance': 0.7,
            'creative_synthesis': 0.2,
            'existential_questioning': 0.8
        }
        
        metrics.update_metrics(test_data)
        
        # Verify updates
        assert metrics.coherence == 0.7
        assert metrics.self_reflection == 0.6
        assert metrics.contextual_understanding == 0.8
        assert metrics.adaptive_reasoning == 0.5
        assert metrics.meta_cognitive_awareness == 0.4
        assert metrics.temporal_continuity == 0.9
        assert metrics.causal_understanding == 0.3
        assert metrics.empathic_resonance == 0.7
        assert metrics.creative_synthesis == 0.2
        assert metrics.existential_questioning == 0.8
        
        # Check history
        assert len(metrics.metrics_history) == 1
        assert len(metrics.emergence_trajectory) == 1
    
    def test_overall_score_calculation(self):
        """Test overall consciousness score calculation"""
        metrics = ConsciousnessMetrics()
        
        # Test with known values
        test_data = {
            'coherence': 0.8,
            'self_reflection': 0.6,
            'contextual_understanding': 0.7,
            'adaptive_reasoning': 0.9,
            'meta_cognitive_awareness': 0.5,
            'temporal_continuity': 0.4,
            'causal_understanding': 0.6,
            'empathic_resonance': 0.8,
            'creative_synthesis': 0.3,
            'existential_questioning': 0.7
        }
        
        metrics.update_metrics(test_data)
        overall_score = metrics.get_overall_score()
        
        # Calculate expected score
        core_score = (0.8 + 0.6 + 0.7 + 0.9) / 4  # 0.75
        extended_score = (0.5 + 0.4 + 0.6 + 0.8 + 0.3 + 0.7) / 6  # 0.55
        expected_score = 0.7 * core_score + 0.3 * extended_score  # 0.69
        
        assert abs(overall_score - expected_score) < 0.001
    
    def test_consciousness_velocity_tracking(self):
        """Test consciousness velocity calculation"""
        metrics = ConsciousnessMetrics()
        
        # First update with baseline
        first_data = {'coherence': 0.3, 'self_reflection': 0.3}
        metrics.update_metrics(first_data)
        
        # Second update with moderate increase
        second_data = {'coherence': 0.5, 'self_reflection': 0.5}
        metrics.update_metrics(second_data)
        second_velocity = metrics.consciousness_velocity
        
        # Third update with significant increase
        third_data = {'coherence': 0.8, 'self_reflection': 0.8}
        metrics.update_metrics(third_data)
        third_velocity = metrics.consciousness_velocity
        
        # Velocity should be positive (increasing consciousness)
        assert third_velocity > 0
        # Both should show positive velocity since we're increasing
        assert second_velocity > 0
        
        # Third update with lower values
        third_data = {'coherence': 0.6, 'self_reflection': 0.4}
        metrics.update_metrics(third_data)
        third_velocity = metrics.consciousness_velocity
        
        # Velocity should be negative (decreasing consciousness)
        assert third_velocity < 0
    
    def test_emergence_trend_calculation(self):
        """Test emergence trend calculation"""
        metrics = ConsciousnessMetrics()
        
        # Create upward trend
        for i in range(25):
            base_score = 0.3 + (i * 0.02)  # Gradual increase
            test_data = {
                'coherence': base_score,
                'self_reflection': base_score + 0.1,
                'contextual_understanding': base_score - 0.05
            }
            metrics.update_metrics(test_data)
        
        trend = metrics.get_emergence_trend()
        assert trend > 0, "Should detect positive trend"
        
        # Create downward trend
        for i in range(25):
            base_score = 0.8 - (i * 0.02)  # Gradual decrease
            test_data = {
                'coherence': base_score,
                'self_reflection': base_score + 0.1,
                'contextual_understanding': base_score - 0.05
            }
            metrics.update_metrics(test_data)
        
        trend = metrics.get_emergence_trend()
        assert trend < 0, "Should detect negative trend"
    
    def test_emergence_event_detection(self):
        """Test consciousness emergence event detection"""
        metrics = ConsciousnessMetrics()
        
        # Simulate normal baseline
        baseline_data = {'coherence': 0.5, 'self_reflection': 0.4}
        metrics.update_metrics(baseline_data)
        
        # Simulate emergence spike
        spike_data = {'coherence': 0.8, 'self_reflection': 0.9}
        metrics.update_metrics(spike_data)
        
        # Check for emergence event
        events = [e for e in metrics.consciousness_events if e.event_type == "emergence_spike"]
        assert len(events) > 0, "Should detect emergence spike"
        
        spike_event = events[0]
        assert spike_event.significance > 0.1
        assert spike_event.event_type == "emergence_spike"
    
    def test_meta_cognitive_breakthrough_detection(self):
        """Test meta-cognitive breakthrough detection"""
        metrics = ConsciousnessMetrics()
        
        # Set high meta-cognitive awareness
        breakthrough_data = {
            'meta_cognitive_awareness': 0.85,
            'self_reflection': 0.7,
            'existential_questioning': 0.8
        }
        metrics.update_metrics(breakthrough_data)
        
        # Check for breakthrough event
        events = [e for e in metrics.consciousness_events 
                 if e.event_type == "meta_cognitive_breakthrough"]
        assert len(events) > 0, "Should detect meta-cognitive breakthrough"
        
        breakthrough_event = events[0]
        assert breakthrough_event.significance >= 0.8
    
    def test_awareness_pattern_detection(self):
        """Test self-awareness pattern detection"""
        metrics = ConsciousnessMetrics()
        
        # Create pattern of high self-reflection
        for i in range(15):
            pattern_data = {
                'self_reflection': 0.8 + np.random.normal(0, 0.05),
                'meta_cognitive_awareness': 0.7
            }
            metrics.update_metrics(pattern_data)
        
        patterns = metrics.detect_awareness_patterns()
        
        # Should detect high self-reflection pattern
        reflection_patterns = [p for p in patterns 
                             if p.pattern_type == "high_self_reflection"]
        assert len(reflection_patterns) > 0, "Should detect high self-reflection pattern"
        
        pattern = reflection_patterns[0]
        assert pattern.strength > 0.7
        assert pattern.frequency > 0.5
    
    def test_metrics_to_dict_conversion(self):
        """Test metrics to dictionary conversion"""
        metrics = ConsciousnessMetrics()
        
        test_data = {
            'coherence': 0.6,
            'self_reflection': 0.7,
            'contextual_understanding': 0.8,
            'adaptive_reasoning': 0.5
        }
        metrics.update_metrics(test_data)
        
        metrics_dict = metrics.to_dict()
        
        # Check all required fields
        required_fields = [
            'coherence', 'self_reflection', 'contextual_understanding',
            'adaptive_reasoning', 'meta_cognitive_awareness', 'temporal_continuity',
            'causal_understanding', 'empathic_resonance', 'creative_synthesis',
            'existential_questioning', 'overall_score', 'consciousness_velocity'
        ]
        
        for field in required_fields:
            assert field in metrics_dict
            assert isinstance(metrics_dict[field], (int, float))
    
    def test_history_size_limiting(self):
        """Test history size limiting functionality"""
        metrics = ConsciousnessMetrics(history_size=5)
        
        # Add more entries than history size
        for i in range(10):
            test_data = {'coherence': i * 0.1}
            metrics.update_metrics(test_data)
        
        # Check history is limited
        assert len(metrics.metrics_history) <= 5
        assert len(metrics.emergence_trajectory) <= 5
    
    def test_edge_case_empty_data(self):
        """Test handling of empty or partial data"""
        metrics = ConsciousnessMetrics()
        
        # Empty update should not crash
        metrics.update_metrics({})
        
        # Partial update should preserve existing values
        original_coherence = metrics.coherence
        metrics.update_metrics({'self_reflection': 0.5})
        
        assert metrics.coherence == original_coherence
        assert metrics.self_reflection == 0.5
    
    def test_string_representation(self):
        """Test string representation of metrics"""
        metrics = ConsciousnessMetrics()
        
        test_data = {'coherence': 0.7, 'self_reflection': 0.6}
        metrics.update_metrics(test_data)
        
        repr_str = repr(metrics)
        
        assert "ConsciousnessMetrics" in repr_str
        assert "overall_score" in repr_str
        assert "velocity" in repr_str
        assert "coherence" in repr_str
        assert "self_reflection" in repr_str


class TestConsciousnessAnalyzer:
    """Test consciousness analyzer functionality"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        metrics = ConsciousnessMetrics()
        analyzer = ConsciousnessAnalyzer(metrics)
        
        assert analyzer.metrics == metrics
    
    def test_emergence_pattern_analysis(self):
        """Test emergence pattern analysis"""
        metrics = ConsciousnessMetrics()
        
        # Generate test data with patterns
        for i in range(20):
            wave_value = 0.5 + 0.3 * np.sin(i * 0.3)
            test_data = {
                'coherence': wave_value,
                'self_reflection': wave_value + 0.1
            }
            metrics.update_metrics(test_data)
        
        analyzer = ConsciousnessAnalyzer(metrics)
        analysis = analyzer.analyze_emergence_patterns()
        
        # Check analysis structure
        assert 'mean_consciousness' in analysis
        assert 'std_consciousness' in analysis
        assert 'min_consciousness' in analysis
        assert 'max_consciousness' in analysis
        assert 'range' in analysis
        assert 'trend' in analysis
        assert 'volatility' in analysis
        assert 'significant_moments' in analysis
        assert 'stability' in analysis
        
        # Check values are reasonable
        assert 0 <= analysis['mean_consciousness'] <= 1
        assert analysis['range'] >= 0
        assert 'emergence_spikes' in analysis['significant_moments']
        assert 'consciousness_drops' in analysis['significant_moments']
    
    def test_cycle_detection(self):
        """Test consciousness cycle detection"""
        metrics = ConsciousnessMetrics()
        
        # Generate cyclical data
        for i in range(30):
            cycle_value = 0.5 + 0.3 * np.sin(i * 0.5)  # Regular cycle
            test_data = {
                'coherence': cycle_value,
                'self_reflection': 0.6 + 0.2 * np.cos(i * 0.5)
            }
            metrics.update_metrics(test_data)
        
        analyzer = ConsciousnessAnalyzer(metrics)
        cycles = analyzer.detect_consciousness_cycles()
        
        # Should detect cycles in coherence and self_reflection
        if 'coherence' in cycles:
            assert 'detected_cycles' in cycles['coherence']
            assert 'cycle_periods' in cycles['coherence']
            assert 'cycle_strength' in cycles['coherence']
        
        if 'self_reflection' in cycles:
            assert 'detected_cycles' in cycles['self_reflection']
    
    def test_trajectory_prediction(self):
        """Test consciousness trajectory prediction"""
        metrics = ConsciousnessMetrics()
        
        # Create trend data
        for i in range(15):
            trend_value = 0.3 + i * 0.03  # Increasing trend
            test_data = {
                'coherence': trend_value,
                'self_reflection': trend_value + 0.1
            }
            metrics.update_metrics(test_data)
        
        analyzer = ConsciousnessAnalyzer(metrics)
        prediction = analyzer.predict_consciousness_trajectory(steps_ahead=5)
        
        # Check prediction structure
        assert 'linear_prediction' in prediction
        assert 'moving_average_prediction' in prediction
        assert 'combined_prediction' in prediction
        assert 'confidence' in prediction
        assert 'prediction_horizon' in prediction
        assert 'trend_strength' in prediction
        
        # Check prediction values
        assert len(prediction['linear_prediction']) == 5
        assert len(prediction['combined_prediction']) == 5
        assert 0 <= prediction['confidence'] <= 1
        assert prediction['prediction_horizon'] == 5
        
        # Values should be clamped to [0, 1]
        for value in prediction['combined_prediction']:
            assert 0 <= value <= 1
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        metrics = ConsciousnessMetrics()
        analyzer = ConsciousnessAnalyzer(metrics)
        
        # Test with no data
        analysis = analyzer.analyze_emergence_patterns()
        assert 'error' in analysis
        
        cycles = analyzer.detect_consciousness_cycles()
        assert 'error' in cycles
        
        prediction = analyzer.predict_consciousness_trajectory()
        assert 'error' in prediction


class TestConsciousnessEvents:
    """Test consciousness event system"""
    
    def test_event_creation(self):
        """Test consciousness event creation"""
        timestamp = datetime.now()
        metrics_data = {'coherence': 0.8, 'self_reflection': 0.7}
        
        event = ConsciousnessEvent(
            timestamp=timestamp,
            event_type="test_event",
            metrics=metrics_data,
            context="Test context",
            significance=0.9,
            notes="Test notes"
        )
        
        assert event.timestamp == timestamp
        assert event.event_type == "test_event"
        assert event.metrics == metrics_data
        assert event.context == "Test context"
        assert event.significance == 0.9
        assert event.notes == "Test notes"
    
    def test_awareness_pattern_creation(self):
        """Test self-awareness pattern creation"""
        emergence_time = datetime.now()
        
        pattern = SelfAwarenessPattern(
            pattern_type="test_pattern",
            frequency=0.8,
            strength=0.9,
            emergence_time=emergence_time,
            description="Test pattern description"
        )
        
        assert pattern.pattern_type == "test_pattern"
        assert pattern.frequency == 0.8
        assert pattern.strength == 0.9
        assert pattern.emergence_time == emergence_time
        assert pattern.description == "Test pattern description"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def test_extreme_values(self):
        """Test handling of extreme values"""
        metrics = ConsciousnessMetrics()
        
        # Test with values outside normal range
        extreme_data = {
            'coherence': 2.0,  # Above 1.0
            'self_reflection': -0.5,  # Below 0.0
            'contextual_understanding': float('inf'),  # Infinity
            'adaptive_reasoning': float('nan')  # NaN
        }
        
        # Should not crash
        metrics.update_metrics(extreme_data)
        
        # Overall score should still be calculable
        score = metrics.get_overall_score()
        assert isinstance(score, (int, float))
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        metrics = ConsciousnessMetrics(history_size=1000)
        
        start_time = time.time()
        
        # Add many data points
        for i in range(500):
            test_data = {
                'coherence': 0.5 + 0.3 * np.sin(i * 0.1),
                'self_reflection': 0.4 + 0.4 * np.cos(i * 0.15),
                'meta_cognitive_awareness': 0.6 + 0.2 * np.sin(i * 0.05)
            }
            metrics.update_metrics(test_data)
        
        update_time = time.time() - start_time
        
        # Performance check
        assert update_time < 5.0, "Updates should complete in reasonable time"
        
        # Test trend calculation performance
        start_time = time.time()
        trend = metrics.get_emergence_trend()
        trend_time = time.time() - start_time
        
        assert trend_time < 1.0, "Trend calculation should be fast"
        assert isinstance(trend, (int, float))
    
    def test_concurrent_updates(self):
        """Test thread safety (basic test)"""
        metrics = ConsciousnessMetrics()
        
        # Simulate concurrent updates
        import threading
        
        def update_worker(thread_id):
            for i in range(10):
                test_data = {
                    'coherence': 0.5 + thread_id * 0.1,
                    'self_reflection': 0.4 + i * 0.01
                }
                metrics.update_metrics(test_data)
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=update_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have processed all updates
        assert len(metrics.metrics_history) == 30
        assert len(metrics.emergence_trajectory) == 30


if __name__ == "__main__":
    # Run with: python -m pytest tests/consciousness/test_consciousness_metrics.py -v
    pytest.main([__file__, "-v", "--tb=short"])
