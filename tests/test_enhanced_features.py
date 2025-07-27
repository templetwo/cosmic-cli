#!/usr/bin/env python3
"""
🧪 Enhanced Test Suite for Cosmic CLI
Comprehensive testing for all enhanced features including plugins, UI, and agents
"""

import pytest
import os
import json
import tempfile
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import sqlite3
from datetime import datetime

# Test imports
from cosmic_cli.enhanced_stargazer import (
    EnhancedStargazerAgent, MemoryManager, 
    InstructorProfile, ExecutionStep, ActionType, Priority
)
from cosmic_cli.plugins.base import BasePlugin, PluginManager, PluginMetadata
from cosmic_cli.plugins.file_operations import FileOperationsPlugin
ENHANCED_FEATURES_AVAILABLE = True

class TestMemoryManager:
    """Test enhanced memory management"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_memory_manager_initialization(self, temp_db):
        """Test memory manager initialization"""
        manager = MemoryManager(temp_db)
        
        # Check database tables exist
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
        assert 'memories' in tables
        assert 'sessions' in tables
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_store_and_retrieve_memory(self, temp_db):
        """Test storing and retrieving memories"""
        manager = MemoryManager(temp_db)
        
        # Store a memory
        manager.store_memory(
            "test_directive_1",
            "This is a test memory",
            "text",
            {"step": 1},
            0.8
        )
        
        # Retrieve memories
        memories = manager.retrieve_memories("test_directive_1")
        
        assert len(memories) == 1
        assert memories[0]["content"] == "This is a test memory"
        assert memories[0]["content_type"] == "text"
        assert memories[0]["importance"] == 0.8
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_session_storage(self, temp_db):
        """Test session storage and retrieval"""
        manager = MemoryManager(temp_db)
        
        # Store a session
        session_data = {"status": "completed", "steps": 5}
        manager.store_session(
            "session_123",
            "test directive",
            "completed",
            session_data,
            {"duration": 30}
        )
        
        # Verify session was stored
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", ("session_123",))
            row = cursor.fetchone()
            
        assert row is not None
        assert row[1] == "test directive"  # directive column
        assert row[2] == "completed"  # status column

class TestEnhancedStargazerAgent:
    """Test enhanced Stargazer agent functionality"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing"""
        with patch('openai.OpenAI'):
            agent = EnhancedStargazerAgent(
                directive="test directive",
                api_key="test_key",
                exec_mode="safe"
            )
            return agent
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_agent_initialization(self, mock_agent):
        """Test enhanced agent initialization"""
        assert mock_agent.directive == "test directive"
        assert mock_agent.exec_mode == "safe"
        assert mock_agent.session_id is not None
        assert mock_agent.memory_manager is not None
        assert mock_agent.instructor_system is not None
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_execution_step_creation(self):
        """Test execution step creation"""
        step = ExecutionStep(
            action_type=ActionType.READ,
            parameters="test.txt",
            context={"test": True},
            timestamp=datetime.now(),
            priority=Priority.HIGH
        )
        
        assert step.action_type == ActionType.READ
        assert step.parameters == "test.txt"
        assert step.priority == Priority.HIGH
        assert step.success is None  # Not executed yet
        assert step.dependencies == []
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    @pytest.mark.asyncio
    async def test_async_execution(self, mock_agent):
        """Test asynchronous execution"""
        # Mock the plan creation
        with patch.object(mock_agent, '_create_enhanced_plan') as mock_plan:
            mock_plan.return_value = [
                ExecutionStep(
                    ActionType.INFO, "test info",
                    {}, datetime.now(), Priority.MEDIUM
                )
            ]
            
            # Mock step execution
            with patch.object(mock_agent, '_execute_step_async') as mock_execute:
                mock_execute.return_value = {
                    "success": True,
                    "output": "Test output",
                    "execution_time": 0.1
                }
                
                result = await mock_agent.execute_async()
                
                assert result["directive"] == "test directive"
                assert result["session_id"] == mock_agent.session_id
                assert "steps" in result

class TestPluginSystem:
    """Test plugin system functionality"""
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_plugin_load_failure(self, tmpdir):
        """Test plugin loading failure handling"""
        plugin_file = tmpdir.join("invalid_plugin.py")
        plugin_file.write("invalid python syntax {")

        manager = PluginManager([str(tmpdir)])

        # Should not raise exception, but should return False
        result = manager.load_plugin("invalid_plugin")
        assert result == False
        assert "invalid_plugin" not in manager.plugins
        
        plugin_content = '''
from cosmic_cli.plugins.base import BasePlugin, PluginMetadata

class TestPlugin(BasePlugin):
    def _initialize_metadata(self):
        self.metadata = PluginMetadata(
            name="Test Plugin",
            version="1.0.0",
            description="Test plugin for unit tests",
            author="Test Author",
            dependencies=[],
            categories=["test"]
        )

    def initialize(self):
        return True

    def can_handle(self, action_type, parameters, context=None):
        return action_type == "TEST"

    def execute(self, action_type, parameters, context=None):
        return {"success": True, "output": "Test executed"}
'''
        with open(plugin_file, 'w', encoding='utf-8') as f:
            f.write(plugin_content)
        
        manager = PluginManager([str(tmpdir)])
        discovered = manager.discover_plugins()
        
        assert "test_plugin" in discovered
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_file_operations_plugin(self):
        """Test file operations plugin"""
        plugin = FileOperationsPlugin()
        
        # Test initialization
        assert plugin.initialize() == True
        assert plugin.metadata.name == "File Operations"
        
        # Test capabilities
        capabilities = plugin.get_capabilities()
        assert "READ_FILE" in capabilities
        assert "WRITE_FILE" in capabilities
        assert "LIST_DIR" in capabilities
        
        # Test can_handle
        assert plugin.can_handle("READ_FILE", "/test/path") == True
        assert plugin.can_handle("INVALID_ACTION", "/test/path") == False

class TestInstructorSystem:
    """Test enhanced instructor system"""
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_instructor_profile_creation(self):
        """Test instructor profile creation and learning"""
        profile = InstructorProfile(
            name="Test Instructor",
            specialization="Testing",
            system_prompt="Test prompt",
            preferred_actions=[ActionType.INFO],
            risk_tolerance=0.5,
            creativity_level=0.7,
            learning_rate=0.1,
            expertise_domains=["testing"]
        )
        
        assert profile.name == "Test Instructor"
        assert profile.get_current_success_rate() == 0.5  # Default when no history
        
        # Test learning
        profile.update_success_rate(True)
        assert len(profile.success_history) == 1
        
        profile.update_success_rate(False)
        assert len(profile.success_history) == 2
        
        # Success rate should be between previous rate and new score
        current_rate = profile.get_current_success_rate()
        assert 0 < current_rate < 1

class TestIntegration:
    """Integration tests for enhanced features"""
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    @pytest.mark.integration
    def test_agent_with_plugins(self):
        """Test agent execution with plugin integration"""
        # Create a simple plugin for testing
        class MockPlugin(BasePlugin):
            def _initialize_metadata(self):
                self.metadata = PluginMetadata(
                    name="Mock Plugin",
                    version="1.0.0",
                    description="Mock plugin for testing",
                    author="Test",
                    dependencies=[],
                    categories=["test"],
                    priority=100
                )
            
            def initialize(self):
                return True
            
            def can_handle(self, action_type, parameters, context=None):
                return action_type == "MOCK_ACTION"
            
            def execute(self, action_type, parameters, context=None):
                return {
                    "success": True,
                    "output": f"Mock executed with parameters: {parameters}"
                }
        
        # Test plugin integration
        mock_plugin = MockPlugin()
        assert mock_plugin.initialize() == True
        
        result = mock_plugin.execute("MOCK_ACTION", "test params")
        assert result["success"] == True
        assert "test params" in result["output"]
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    @pytest.mark.integration
    def test_memory_persistence_workflow(self):
        """Test complete memory persistence workflow"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Create memory manager
            memory_manager = MemoryManager(db_path)
            
            # Simulate agent workflow
            session_id = "test_session_123"
            directive = "test integration directive"
            
            # Store initial session
            memory_manager.store_session(
                session_id, directive, "running"
            )
            
            # Store some memories
            memories = [
                ("Started execution", "log", 0.3),
                ("Read configuration file", "context", 0.7),
                ("Generated response", "output", 0.9),
                ("Completed successfully", "log", 0.5)
            ]
            
            for content, content_type, importance in memories:
                memory_manager.store_memory(
                    session_id, content, content_type,
                    metadata={"timestamp": datetime.now().isoformat()},
                    importance=importance
                )
            
            # Retrieve and verify memories
            retrieved = memory_manager.retrieve_memories(session_id)
            
            assert len(retrieved) == 4
            # Should be sorted by importance (desc) then timestamp (desc)
            assert retrieved[0]["importance"] == 0.9  # Generated response
            assert retrieved[1]["importance"] == 0.7  # Read configuration
            
            # Update session as completed
            memory_manager.store_session(
                session_id, directive, "completed",
                results={"success": True, "steps": 4}
            )
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

class TestUIComponents:
    """Test UI components (basic testing without full TUI execution)"""
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_theme_configuration(self):
        """Test theme configuration"""
        try:
            from cosmic_cli.enhanced_ui import CosmicThemes, Theme
            
            themes = CosmicThemes.get_all_themes()
            
            assert "cosmic_dark" in themes
            assert "cosmic_light" in themes
            assert "neon_cyber" in themes
            
            # Test theme properties
            dark_theme = themes["cosmic_dark"]
            assert isinstance(dark_theme, Theme)
            assert dark_theme.name == "Cosmic Dark"
            assert dark_theme.primary is not None
            
            # Test CSS generation
            css = CosmicThemes.get_theme_css(dark_theme)
            assert "background:" in css
            assert dark_theme.background in css
            
        except ImportError:
            pytest.skip("Enhanced UI not available")

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_agent_with_invalid_api_key(self):
        """Test agent behavior with invalid API key"""
        with patch('openai.OpenAI') as mock_openai:
            mock_openai.side_effect = Exception("Invalid API key")
            
            with pytest.raises(Exception):
                agent = EnhancedStargazerAgent(
                    directive="test",
                    api_key="invalid_key"
                )
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_plugin_load_failure(self, temp_plugin_dir):
        """Test plugin loading failure handling"""
        # Create invalid plugin file
        plugin_file = Path(temp_plugin_dir) / "invalid_plugin.py"
        plugin_file.write_text("invalid python syntax {")
        
        manager = PluginManager([temp_plugin_dir])
        
        # Should not raise exception, but should return False
        result = manager.load_plugin("invalid_plugin")
        assert result == False
        assert "invalid_plugin" not in manager.plugins
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    def test_memory_database_corruption_handling(self):
        """Test handling of corrupted memory database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
            # Write invalid content to create corrupted database
            f.write(b"corrupted database content")
        
        try:
            # Should handle corruption gracefully
            memory_manager = MemoryManager(db_path)
            # This might fail or recreate the database
            # The important thing is it doesn't crash the application
            
        except Exception as e:
            # Some exceptions might be expected for corrupted databases
            assert "database" in str(e).lower() or "sqlite" in str(e).lower()
        
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

# Performance tests
class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.skipif(not ENHANCED_FEATURES_AVAILABLE, reason="Enhanced features not available")
    @pytest.mark.performance
    def test_memory_manager_bulk_operations(self):
        """Test memory manager with bulk operations"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            memory_manager = MemoryManager(db_path)
            
            # Bulk insert test
            import time
            start_time = time.time()
            
            for i in range(100):
                memory_manager.store_memory(
                    f"session_{i % 10}",
                    f"Memory content {i}",
                    "test",
                    {"index": i},
                    0.5
                )
            
            insert_time = time.time() - start_time
            
            # Bulk retrieve test
            start_time = time.time()
            
            for i in range(10):
                memories = memory_manager.retrieve_memories(f"session_{i}")
                assert len(memories) == 10
            
            retrieve_time = time.time() - start_time
            
            # Performance assertions (adjust based on requirements)
            assert insert_time < 5.0  # Should complete in under 5 seconds
            assert retrieve_time < 2.0  # Should complete in under 2 seconds
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

# Fixture for pytest
@pytest.fixture(scope="session")
def enhanced_features_available():
    """Check if enhanced features are available for testing"""
    return ENHANCED_FEATURES_AVAILABLE

@pytest.fixture
def temp_plugin_dir(tmpdir):
    """Create a temporary plugin directory"""
    return tmpdir

if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_enhanced_features.py -v
    pytest.main([__file__, "-v", "--tb=short"])
