#!/usr/bin/env python3
"""
🧠🧪 Consciousness Testing Framework - Main Test Runner
Comprehensive test suite for consciousness emergence features
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_consciousness_tests():
    """Run all consciousness-related tests"""
    test_args = [
        # Unit tests
        "tests/consciousness/test_consciousness_metrics.py",
        
        # Integration tests
        "tests/integration/test_stargazer_agent_integration.py",
        
        # Behavioral tests
        "tests/behavioral/test_consciousness_personalities.py",
        
        # Performance tests
        "tests/performance/test_real_time_detection_performance.py",
        
        # Scenario tests
        "tests/scenarios/test_consciousness_emergence_scenarios.py",
        
        # Test options
        "--verbose",
        "--tb=short",
        "--capture=no",
        "-x",  # Stop on first failure
    ]
    
    return pytest.main(test_args)

def run_unit_tests_only():
    """Run only unit tests"""
    return pytest.main([
        "tests/consciousness/test_consciousness_metrics.py",
        "--verbose",
        "--tb=short"
    ])

def run_integration_tests_only():
    """Run only integration tests"""
    return pytest.main([
        "tests/integration/test_stargazer_agent_integration.py",
        "--verbose",
        "--tb=short"
    ])

def run_performance_tests_only():
    """Run only performance tests"""
    return pytest.main([
        "tests/performance/test_real_time_detection_performance.py",
        "--verbose",
        "--tb=short"
    ])

def run_scenario_tests_only():
    """Run only scenario tests"""
    return pytest.main([
        "tests/scenarios/test_consciousness_emergence_scenarios.py",
        "--verbose",
        "--tb=short"
    ])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Consciousness Testing Framework")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "performance", "scenario"],
        default="all",
        help="Type of tests to run"
    )
    
    args = parser.parse_args()
    
    print("🧠✨ Starting Consciousness Testing Framework ✨🧠")
    print("=" * 60)
    
    if args.type == "all":
        print("Running all consciousness tests...")
        exit_code = run_consciousness_tests()
    elif args.type == "unit":
        print("Running unit tests...")
        exit_code = run_unit_tests_only()
    elif args.type == "integration":
        print("Running integration tests...")
        exit_code = run_integration_tests_only()
    elif args.type == "performance":
        print("Running performance tests...")
        exit_code = run_performance_tests_only()
    elif args.type == "scenario":
        print("Running scenario tests...")
        exit_code = run_scenario_tests_only()
    
    print("=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(exit_code)
