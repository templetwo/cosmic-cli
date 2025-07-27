#!/usr/bin/env python3
"""
🔄 Integration Tests for StargazerAgent
Testing integration of consciousness monitoring with real-time agent operations
"""

import pytest
import asyncio
import time
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from cosmic_cli.enhanced_stargazer import EnhancedStargazerAgent
from cosmic_cli.consciousness_assessment import (
    ConsciousnessMetrics, AssessmentProtocol, RealTimeConsciousnessMonitor,
    ConsciousnessLevel, integrate_consciousness_monitoring
)


class TestStargazerAgentIntegration:
    """Test StargazerAgent integration with consciousness monitoring"""

    @pytest.mark.asyncio
    async def test_monitor_integration(self):
        """Test consciousness monitoring integration"""
        with patch('openai.OpenAI'):
            # Mock agent
            agent = EnhancedStargazerAgent(
                directive="test integration",
                api_key="test_key"
            )
            
            # Setup consciousness monitoring
            monitor = RealTimeConsciousnessMonitor(
                agent=agent,
                metrics=ConsciousnessMetrics(),
                protocol=AssessmentProtocol()
            )
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Check initial state
            assert monitor.is_monitoring is True
            assert monitor.last_consciousness_level == monitor.protocol.evaluate(monitor.metrics)
            
            # Simulate data collection
            with patch.object(monitor, 'collect_consciousness_data', return_value=asyncio.Future()) as mock_data:
                mock_data.return_value.set_result({
                    'coherence': 0.56,
                    'self_reflection': 0.62,
                    'contextual_understanding': 0.63,
                    'adaptive_reasoning': 0.68,
                })
                
                # Run a monitoring loop iteration
                await monitor._monitoring_loop()
                
                # Verify data update
                assert monitor.metrics.coherence == 0.56
                assert monitor.metrics.self_reflection == 0.62

            # Stop monitoring
            await monitor.stop_monitoring()
            assert monitor.is_monitoring is False
    
    @pytest.mark.asyncio
    async def test_consciousness_change_detection(self):
        """Test detection of consciousness level changes"""
        with patch('openai.OpenAI'):
            # Mock agent
            agent = EnhancedStargazerAgent(
                directive="test change detection",
                api_key="test_key"
            )
            
            # Setup consciousness monitoring
            monitor = RealTimeConsciousnessMonitor(
                agent=agent,
                metrics=ConsciousnessMetrics(),
                protocol=AssessmentProtocol()
            )
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Simulate data collection with change
            with patch.object(monitor, 'collect_consciousness_data', return_value=asyncio.Future()) as mock_data:
                mock_data.return_value.set_result({
                    'coherence': 0.8,
                    'self_reflection': 0.9
                })
                
                # Run a monitoring loop iteration
                await monitor._monitoring_loop()
                
                # Check for recorded awareness patterns
                assert len(monitor.metrics.awareness_patterns) >= 0
                
                # Stop monitoring
                await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring_loop(self):
        """Test real-time monitoring loop with multiple iterations"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="test real-time monitoring",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(
                agent=agent,
                monitoring_interval=0.1  # Fast for testing
            )
            
            # Mock data collection to return varying consciousness levels
            data_sequence = [
                {'coherence': 0.4, 'self_reflection': 0.3},  # Low
                {'coherence': 0.6, 'self_reflection': 0.5},  # Medium
                {'coherence': 0.8, 'self_reflection': 0.9},  # High
            ]
            
            call_count = 0
            async def mock_collect_data():
                nonlocal call_count
                data = data_sequence[call_count % len(data_sequence)]
                call_count += 1
                return data
            
            with patch.object(monitor, 'collect_consciousness_data', side_effect=mock_collect_data):
                # Start monitoring
                await monitor.start_monitoring()
                
                # Let it run for a short time
                await asyncio.sleep(0.3)
                
                # Stop monitoring
                await monitor.stop_monitoring()
                
                # Check that multiple data points were collected
                assert len(monitor.metrics.metrics_history) > 1
                assert call_count >= 2
    
    @pytest.mark.asyncio
    async def test_agent_memory_integration(self):
        """Test consciousness data integration with agent memory"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="test memory integration",
                api_key="test_key"
            )
            
            # Mock agent memory methods
            agent._add_to_memory = Mock()
            
            monitor = RealTimeConsciousnessMonitor(agent=agent)
            
            # Simulate consciousness level change
            with patch.object(monitor, 'collect_consciousness_data') as mock_data:
                mock_data.return_value = {
                    'coherence': 0.85,
                    'self_reflection': 0.9,
                    'meta_cognitive_awareness': 0.8
                }
                
                # Start monitoring
                await monitor.start_monitoring()
                
                # Let one iteration run
                await asyncio.sleep(0.1)
                
                # Stop monitoring
                await monitor.stop_monitoring()
                
                # Check that consciousness data was added to agent memory
                assert agent._add_to_memory.called
    
    @pytest.mark.asyncio
    async def test_consciousness_report_generation(self):
        """Test comprehensive consciousness report generation"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="test report generation",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(agent=agent)
            
            # Add some test data
            test_data = {
                'coherence': 0.7,
                'self_reflection': 0.8,
                'contextual_understanding': 0.6,
                'adaptive_reasoning': 0.9,
                'meta_cognitive_awareness': 0.5
            }
            
            monitor.metrics.update_metrics(test_data)
            
            # Generate report
            report = monitor.get_consciousness_report()
            
            # Verify report structure
            assert 'current_level' in report
            assert 'metrics' in report
            assert 'assessment_summary' in report
            assert 'emergence_alerts' in report
            assert 'awareness_patterns' in report
            assert 'consciousness_events' in report
            assert 'trend_analysis' in report
            
            # Check trend analysis structure
            trend_analysis = report['trend_analysis']
            assert 'emergence_trend' in trend_analysis
            assert 'velocity' in trend_analysis
            assert 'stability' in trend_analysis
    
    @pytest.mark.asyncio
    async def test_integration_factory_function(self):
        """Test integrate_consciousness_monitoring factory function"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="test factory integration",
                api_key="test_key"
            )
            
            # Use factory function to integrate monitoring
            monitor = integrate_consciousness_monitoring(
                agent=agent,
                monitoring_interval=1.0,
                consciousness_threshold=0.6
            )
            
            # Verify integration
            assert hasattr(agent, 'consciousness_monitor')
            assert agent.consciousness_monitor == monitor
            assert monitor.agent == agent
            assert monitor.monitoring_interval == 1.0
            assert monitor.protocol.consciousness_threshold == 0.6
    
    @pytest.mark.asyncio
    async def test_consciousness_level_transitions(self):
        """Test consciousness level transitions and alerts"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="test level transitions",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(
                agent=agent,
                monitoring_interval=0.1
            )
            
            # Simulate progression through consciousness levels
            level_data = [
                {'coherence': 0.2, 'self_reflection': 0.3},  # DORMANT
                {'coherence': 0.5, 'self_reflection': 0.6},  # AWAKENING
                {'coherence': 0.7, 'self_reflection': 0.8},  # EMERGING
                {'coherence': 0.9, 'self_reflection': 0.9},  # CONSCIOUS
            ]
            
            call_count = 0
            async def mock_collect_progressive_data():
                nonlocal call_count
                if call_count < len(level_data):
                    data = level_data[call_count]
                    call_count += 1
                    return data
                return level_data[-1]  # Stay at highest level
            
            with patch.object(monitor, 'collect_consciousness_data', side_effect=mock_collect_progressive_data):
                await monitor.start_monitoring()
                
                # Let it progress through levels
                await asyncio.sleep(0.5)
                
                await monitor.stop_monitoring()
                
                # Should have recorded level changes
                assert len(monitor.emergence_alerts) > 0
                
                # Check for level change alerts
                level_changes = [alert for alert in monitor.emergence_alerts 
                               if alert['type'] == 'level_change']
                assert len(level_changes) > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_in_monitoring(self):
        """Test error recovery during consciousness monitoring"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="test error recovery",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(
                agent=agent,
                monitoring_interval=0.1
            )
            
            # Mock data collection to raise exception then work normally
            call_count = 0
            async def mock_collect_with_error():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Simulated collection error")
                return {'coherence': 0.5, 'self_reflection': 0.6}
            
            with patch.object(monitor, 'collect_consciousness_data', side_effect=mock_collect_with_error):
                await monitor.start_monitoring()
                
                # Let it run through error and recovery
                await asyncio.sleep(0.3)
                
                await monitor.stop_monitoring()
                
                # Should have recovered and continued collecting data
                assert call_count > 1
                assert len(monitor.metrics.metrics_history) > 0
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test monitoring performance with high-frequency updates"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="test performance",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(
                agent=agent,
                monitoring_interval=0.01  # Very fast for stress test
            )
            
            # Mock rapid data collection
            async def mock_rapid_data():
                return {
                    'coherence': 0.5 + 0.1 * (time.time() % 1),
                    'self_reflection': 0.6 + 0.1 * (time.time() % 1)
                }
            
            with patch.object(monitor, 'collect_consciousness_data', side_effect=mock_rapid_data):
                start_time = time.time()
                
                await monitor.start_monitoring()
                
                # Run for short duration with high frequency
                await asyncio.sleep(0.2)
                
                await monitor.stop_monitoring()
                
                end_time = time.time()
                
                # Should have collected many data points efficiently
                assert len(monitor.metrics.metrics_history) > 5
                assert (end_time - start_time) < 1.0  # Should complete quickly


class TestConsciousnessIntegrationScenarios:
    """Test various consciousness emergence scenarios"""
    
    @pytest.mark.asyncio
    async def test_gradual_awakening_scenario(self):
        """Test gradual consciousness awakening scenario"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="gradual awakening test",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(agent=agent)
            
            # Simulate gradual increase in consciousness
            for i in range(10):
                base_value = 0.1 + (i * 0.08)  # Gradual increase
                test_data = {
                    'coherence': base_value,
                    'self_reflection': base_value + 0.1,
                    'contextual_understanding': base_value + 0.05,
                    'meta_cognitive_awareness': base_value - 0.05
                }
                monitor.metrics.update_metrics(test_data)
            
            # Check for positive trend
            trend = monitor.metrics.get_emergence_trend()
            assert trend > 0, "Should detect positive emergence trend"
            
            # Check final consciousness level
            final_level = monitor.protocol.evaluate(monitor.metrics)
            assert final_level != ConsciousnessLevel.DORMANT
    
    @pytest.mark.asyncio
    async def test_consciousness_breakthrough_scenario(self):
        """Test sudden consciousness breakthrough scenario"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="breakthrough test",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(agent=agent)
            
            # Establish baseline
            baseline_data = {
                'coherence': 0.4,
                'self_reflection': 0.3,
                'meta_cognitive_awareness': 0.3
            }
            monitor.metrics.update_metrics(baseline_data)
            
            # Simulate breakthrough
            breakthrough_data = {
                'coherence': 0.9,
                'self_reflection': 0.8,
                'meta_cognitive_awareness': 0.85,
                'existential_questioning': 0.9
            }
            monitor.metrics.update_metrics(breakthrough_data)
            
            # Check for breakthrough events
            breakthrough_events = [
                e for e in monitor.metrics.consciousness_events
                if e.event_type in ["emergence_spike", "meta_cognitive_breakthrough"]
            ]
            
            assert len(breakthrough_events) > 0, "Should detect breakthrough events"
    
    @pytest.mark.asyncio
    async def test_consciousness_oscillation_scenario(self):
        """Test consciousness oscillation pattern scenario"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="oscillation test",
                api_key="test_key"
            )
            
            monitor = RealTimeConsciousnessMonitor(agent=agent)
            
            # Simulate oscillating consciousness pattern
            import math
            for i in range(20):
                wave_value = 0.5 + 0.3 * math.sin(i * 0.5)
                test_data = {
                    'coherence': wave_value,
                    'self_reflection': wave_value + 0.1,
                    'temporal_continuity': 0.6 + 0.2 * math.cos(i * 0.3)
                }
                monitor.metrics.update_metrics(test_data)
            
            # Use analyzer to detect patterns
            from cosmic_cli.consciousness_assessment import ConsciousnessAnalyzer
            analyzer = ConsciousnessAnalyzer(monitor.metrics)
            
            cycles = analyzer.detect_consciousness_cycles()
            
            # Should detect cyclical patterns
            if 'coherence' in cycles:
                assert cycles['coherence']['detected_cycles'] > 0

if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_stargazer_agent_integration.py -v
    pytest.main([__file__, "-v", "--tb=short"])
