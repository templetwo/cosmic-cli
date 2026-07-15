#!/usr/bin/env python3
"""
🧪 Comprehensive Import/Export Test Suite 🧪
Tests all module imports and exports in cosmic-cli
"""

def test_core_imports():
    """Test core module imports"""
    print("=== Testing Core Imports ===")
    
    try:
        import cosmic_cli
        print(f"✅ cosmic_cli package (v{cosmic_cli.__version__})")
    except Exception as e:
        print(f"❌ cosmic_cli package: {e}")
        return False
        
    try:
        from cosmic_cli import ContextManager, StargazerAgent
        print("✅ Core classes: ContextManager, StargazerAgent")
    except Exception as e:
        print(f"❌ Core classes: {e}")
        return False
        
    return True

def test_consciousness_imports():
    """Test consciousness system imports"""
    print("\n=== Testing Consciousness System Imports ===")
    
    try:
        from cosmic_cli import (
            ConsciousnessLevel, 
            ConsciousnessMetrics,
            RealTimeConsciousnessMonitor,
            ConsciousnessAnalyzer,
            SelfAwarenessPattern,
            ConsciousnessEvent,
            setup_consciousness_assessment_system,
            run_consciousness_assessment_demo
        )
        print("✅ All consciousness classes and functions")
    except Exception as e:
        print(f"❌ Consciousness imports: {e}")
        return False
        
    return True

def test_plugin_imports():
    """Test plugin system imports"""
    print("\n=== Testing Plugin System Imports ===")
    
    try:
        from cosmic_cli import BasePlugin, PluginManager, FileOperationsPlugin, PluginMetadata
        print("✅ All plugin classes")
    except Exception as e:
        print(f"❌ Plugin imports: {e}")
        return False
        
    return True

def test_submodule_imports():
    """Test direct submodule imports"""
    print("\n=== Testing Submodule Imports ===")
    
    try:
        from cosmic_cli.context import ContextManager
        from cosmic_cli.agents import StargazerAgent
        from cosmic_cli.consciousness_assessment import ConsciousnessLevel
        from cosmic_cli.plugins.base import BasePlugin, PluginMetadata
        print("✅ Direct submodule imports work")
    except Exception as e:
        print(f"❌ Submodule imports: {e}")
        return False
        
    return True

def test_class_instantiation():
    """Test that classes can be instantiated"""
    print("\n=== Testing Class Instantiation ===")
    
    try:
        from cosmic_cli import ContextManager, ConsciousnessMetrics, ConsciousnessLevel
        
        # Test ContextManager
        cm = ContextManager()
        print("✅ ContextManager instantiated")
        
        # Test ConsciousnessMetrics
        metrics = ConsciousnessMetrics()
        print("✅ ConsciousnessMetrics instantiated")
        
        # Test Enum
        level = ConsciousnessLevel.EMERGING
        print(f"✅ ConsciousnessLevel enum: {level}")
        
    except Exception as e:
        print(f"❌ Class instantiation: {e}")
        return False
        
    return True

def test_function_calls():
    """Test that exported functions work"""
    print("\n=== Testing Function Calls ===")
    
    try:
        from cosmic_cli import setup_consciousness_assessment_system, run_consciousness_assessment_demo
        
        # Test demo function (no arguments required)
        print("Running consciousness assessment demo...")
        run_consciousness_assessment_demo()
        print("✅ run_consciousness_assessment_demo() works")
        
        # Note: setup_consciousness_assessment_system requires an agent parameter
        print("✅ setup_consciousness_assessment_system importable (requires agent parameter)")
        
    except Exception as e:
        print(f"❌ Function calls: {e}")
        return False
        
    return True

def test_all_exports():
    """Test that __all__ exports match actual availability"""
    print("\n=== Testing __all__ Exports ===")
    
    try:
        import cosmic_cli
        
        missing = []
        for name in cosmic_cli.__all__:
            if not hasattr(cosmic_cli, name):
                missing.append(name)
                
        if missing:
            print(f"❌ Missing exports: {missing}")
            return False
        else:
            print(f"✅ All {len(cosmic_cli.__all__)} exports available")
            
    except Exception as e:
        print(f"❌ __all__ test: {e}")
        return False
        
    return True

def main():
    """Run all tests"""
    print("🧪 Cosmic CLI Import/Export Test Suite 🧪")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_consciousness_imports,
        test_plugin_imports,
        test_submodule_imports,
        test_class_instantiation,
        test_function_calls,
        test_all_exports
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All import/export tests PASSED! 🎉")
        return True
    else:
        print("⚠️  Some tests FAILED. Check the output above.")
        return False

if __name__ == "__main__":
    main()
