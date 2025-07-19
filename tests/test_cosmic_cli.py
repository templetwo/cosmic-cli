import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from main import CosmicCLI, load_manual_env, COSMIC_QUOTES


class TestCosmicCLI:
    """Test suite for CosmicCLI class."""
    
    @pytest.fixture
    def cosmic_cli(self):
        """Create a fresh CosmicCLI instance for each test."""
        return CosmicCLI()
    
    @pytest.fixture
    def temp_memory_file(self):
        """Create a temporary memory file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            temp_path = Path(f.name)
        
        # Temporarily replace the memory file path
        original_memory_file = CosmicCLI.__module__ + '.MEMORY_FILE'
        with patch(original_memory_file, temp_path):
            yield temp_path
        
        # Cleanup
        temp_path.unlink(missing_ok=True)
    
    def test_cosmic_cli_initialization(self, cosmic_cli):
        """Test CosmicCLI initialization."""
        assert cosmic_cli.chat_instance is None
        assert cosmic_cli.client_instance is None
        assert cosmic_cli.console is not None
    
    def test_save_memory_success(self, cosmic_cli, temp_memory_file):
        """Test successful memory saving."""
        test_messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        
        cosmic_cli.save_memory(test_messages)
        
        # Verify the file was written
        assert temp_memory_file.exists()
        with open(temp_memory_file, 'r') as f:
            saved_data = json.load(f)
        assert saved_data == test_messages
    
    def test_save_memory_with_xai_objects(self, cosmic_cli, temp_memory_file):
        """Test saving memory with xai_sdk message objects."""
        # Mock xai_sdk message objects
        mock_user_msg = Mock()
        mock_user_msg.role = 'user'
        mock_user_msg.content = 'Hello'
        
        mock_assistant_msg = Mock()
        mock_assistant_msg.role = 'assistant'
        mock_assistant_msg.content = 'Hi there!'
        
        test_messages = [mock_user_msg, mock_assistant_msg]
        
        cosmic_cli.save_memory(test_messages)
        
        # Verify the file was written correctly
        with open(temp_memory_file, 'r') as f:
            saved_data = json.load(f)
        expected_data = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        assert saved_data == expected_data
    
    def test_load_memory_success(self, cosmic_cli, temp_memory_file):
        """Test successful memory loading."""
        test_data = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        
        with open(temp_memory_file, 'w') as f:
            json.dump(test_data, f)
        
        loaded_data = cosmic_cli.load_memory()
        assert loaded_data == test_data
    
    def test_load_memory_file_not_exists(self, cosmic_cli):
        """Test loading memory when file doesn't exist."""
        # Use a non-existent file path
        with patch('main.MEMORY_FILE', Path('/nonexistent/file.json')):
            loaded_data = cosmic_cli.load_memory()
            assert loaded_data == []
    
    def test_load_memory_corrupted_file(self, cosmic_cli, temp_memory_file):
        """Test loading memory from corrupted JSON file."""
        # Write invalid JSON
        with open(temp_memory_file, 'w') as f:
            f.write('invalid json content')
        
        loaded_data = cosmic_cli.load_memory()
        assert loaded_data == []
    
    def test_append_to_memory(self, cosmic_cli, temp_memory_file):
        """Test appending single message to memory."""
        cosmic_cli.append_to_memory('user', 'Hello')
        cosmic_cli.append_to_memory('assistant', 'Hi there!')
        
        with open(temp_memory_file, 'r') as f:
            saved_data = json.load(f)
        
        expected_data = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        assert saved_data == expected_data
    
    @patch('main.Client')
    def test_initialize_chat_success(self, mock_client, cosmic_cli):
        """Test successful chat initialization."""
        # Mock environment
        with patch.dict(os.environ, {'XAI_API_KEY': 'test_key'}):
            # Mock the client and chat instance
            mock_chat = Mock()
            mock_client.return_value.chat.create.return_value = mock_chat
            
            result = cosmic_cli.initialize_chat(load_history=False)
            
            assert result == mock_chat
            assert cosmic_cli.client_instance is not None
            assert cosmic_cli.chat_instance == mock_chat
    
    @patch('main.Client')
    def test_initialize_chat_no_api_key(self, mock_client, cosmic_cli):
        """Test chat initialization without API key."""
        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            result = cosmic_cli.initialize_chat()
            
            assert result is None
            assert cosmic_cli.client_instance is None
            assert cosmic_cli.chat_instance is None
    
    @patch('main.Client')
    def test_initialize_chat_with_history(self, mock_client, cosmic_cli, temp_memory_file):
        """Test chat initialization with history loading."""
        # Create test history
        test_history = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        with open(temp_memory_file, 'w') as f:
            json.dump(test_history, f)
        
        # Mock environment and client
        with patch.dict(os.environ, {'XAI_API_KEY': 'test_key'}):
            mock_chat = Mock()
            mock_client.return_value.chat.create.return_value = mock_chat
            
            result = cosmic_cli.initialize_chat(load_history=True)
            
            assert result == mock_chat
            # Verify system message was added
            mock_chat.append.assert_called()


class TestLoadManualEnv:
    """Test suite for load_manual_env function."""
    
    def test_load_manual_env_file_exists(self):
        """Test loading .env file when it exists."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('XAI_API_KEY=test_key\nGROK_API_KEY=test_key2\n')
            env_path = Path(f.name)
        
        try:
            with patch('main.Path.cwd') as mock_cwd:
                mock_cwd.return_value = env_path.parent
                
                # Clear environment first
                with patch.dict(os.environ, {}, clear=True):
                    load_manual_env()
                    
                    assert os.environ.get('XAI_API_KEY') == 'test_key'
                    assert os.environ.get('GROK_API_KEY') == 'test_key2'
        finally:
            env_path.unlink(missing_ok=True)
    
    def test_load_manual_env_file_not_exists(self):
        """Test loading .env when file doesn't exist."""
        original_env = dict(os.environ)
        
        load_manual_env()
        
        # Environment should remain unchanged
        assert dict(os.environ) == original_env
    
    def test_load_manual_env_invalid_format(self):
        """Test loading .env file with invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('invalid_line_without_equals\n# comment\nXAI_API_KEY=test_key\n')
            env_path = Path(f.name)
        
        try:
            with patch('main.Path.cwd') as mock_cwd:
                mock_cwd.return_value = env_path.parent
                
                with patch.dict(os.environ, {}, clear=True):
                    load_manual_env()
                    
                    # Should only load valid lines
                    assert os.environ.get('XAI_API_KEY') == 'test_key'
        finally:
            env_path.unlink(missing_ok=True)


class TestCosmicQuotes:
    """Test suite for cosmic quotes functionality."""
    
    def test_cosmic_quotes_not_empty(self):
        """Test that cosmic quotes list is not empty."""
        assert len(COSMIC_QUOTES) > 0
    
    def test_cosmic_quotes_format(self):
        """Test that cosmic quotes have the expected format."""
        for quote in COSMIC_QUOTES:
            assert isinstance(quote, str)
            assert len(quote) > 0
            assert quote.endswith(':')  # All quotes should end with colon


class TestIntegration:
    """Integration tests for the CLI."""
    
    @patch('main.cosmic_cli')
    def test_cli_commands_exist(self, mock_cosmic_cli):
        """Test that all CLI commands are properly defined."""
        from main import cli
        
        # Get all command names
        command_names = [cmd.name for cmd in cli.commands.values()]
        
        expected_commands = ['chat', 'ask', 'analyze', 'hack', 'run_command']
        for cmd in expected_commands:
            assert cmd in command_names
    
    def test_cli_help_formatter(self):
        """Test that the cosmic help formatter works."""
        from main import CosmicHelpFormatter
        
        formatter = CosmicHelpFormatter()
        assert formatter is not None
        
        # Test that methods exist and are callable
        assert hasattr(formatter, 'write_usage')
        assert hasattr(formatter, 'write_heading')
        assert hasattr(formatter, 'write_text')
        assert hasattr(formatter, 'write_dl')


if __name__ == '__main__':
    pytest.main([__file__]) 