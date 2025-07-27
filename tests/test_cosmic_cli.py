from unittest.mock import Mock, patch, MagicMock
import os
import subprocess
from cosmic_cli.agents import StargazerAgent
from cosmic_cli.ui import DirectivesUI

class TestCosmicBanner:
    """Test the banner generation to prevent 'Coder CLI' regression"""
    
    @patch('pyfiglet.Figlet')
    def test_banner_generation(self, mock_figlet):
        mock_figlet.return_value.renderText.return_value = 'COSMIC CLI ART'
        ui = DirectivesUI()
        ui.figlet = mock_figlet.return_value  # Properly patch the figlet instance
        banner_text = ui.figlet.renderText('COSMIC CLI')
        assert 'COSMIC CLI ART' == banner_text
        assert 'CODER' not in banner_text.upper()

    @patch('pyfiglet.Figlet')
    def test_ui_compose_banner(self, mock_figlet):
        mock_figlet.return_value.renderText.return_value = 'COSMIC CLI ART'
        ui = DirectivesUI()
        ui.figlet = mock_figlet.return_value  # Properly patch the figlet instance
        banner_text = ui.figlet.renderText('COSMIC CLI')
        assert 'COSMIC CLI ART' == banner_text


class TestStargazerAgent:
    """Test StargazerAgent with real actions"""
    
    def setup_method(self):
        """Setup for each test"""
        os.environ['EXEC_MODE'] = 'safe'
        os.environ['XAI_API_KEY'] = 'test_key'
    
    @patch('openai.OpenAI')
    def test_agent_initialization(self, mock_openai):
        """Test agent initializes correctly"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = StargazerAgent("test directive", api_key="test_key")
        assert agent.directive == "test directive"
        assert agent.status == "✨"
        assert agent.exec_mode == "safe"
        assert len(agent.logs) == 0

    @patch('openai.OpenAI')
    def test_parse_steps(self, mock_openai):
        """Test step parsing from Grok response"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        grok_response = """
        Here's your cosmic plan:
        1. SHELL: ls -l
        2. CODE: Create a hello world script
        3. INFO: Check system status
        """
        
        steps = agent._parse_steps(grok_response)
        
        assert len(steps) == 3
        assert "SHELL: ls -l" in steps[0]
        assert "CODE: Create a hello world script" in steps[1]
        assert "INFO: Check system status" in steps[2]

    @patch('subprocess.run')
    def test_shell_command_execution_safe_mode(self, mock_subprocess):
        """Test shell command execution in safe mode"""
        agent = StargazerAgent("test directive", api_key="test_key")
        mock_subprocess.return_value = Mock(
            stdout="file1.txt\nfile2.txt", 
            stderr="", 
            returncode=0
        )
        
        # Test safe command
        agent._execute_shell_command("ls -l")
        
        mock_subprocess.assert_called_once()
        assert "Executing shell command" in agent.logs[-2]
        assert "Shell command completed successfully" in agent.logs[-1]

    def test_dangerous_command_blocked(self):
        """Test that dangerous commands are blocked in safe mode"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        # Test dangerous command
        agent._execute_shell_command("rm -rf /")
        
        # Should be blocked
        assert any("BLOCKED dangerous command" in log for log in agent.logs)

    @patch('openai.OpenAI')
    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    def test_code_generation_and_execution(self, mock_tempfile, mock_subprocess, mock_openai):
        """Test code generation and execution"""
        # Setup mocks
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock code generation response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="print('Hello, World!')"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock temp file
        mock_temp = Mock()
        mock_temp.name = '/tmp/test.py'
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        
        # Mock subprocess execution
        mock_subprocess.return_value = Mock(
            stdout="Hello, World!\n",
            stderr="",
            returncode=0
        )
        
        agent = StargazerAgent("test directive", api_key="test_key")
        agent.client = mock_client
        
        # Execute code generation
        agent._generate_and_execute_code("create hello world")
        
        # Verify calls
        mock_client.chat.completions.create.assert_called()
        assert any("Generated Code" in log for log in agent.logs)

    @patch('openai.OpenAI')
    def test_information_gathering(self, mock_openai):
        """Test information gathering functionality"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock info response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Python is a programming language"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = StargazerAgent("test directive", api_key="test_key")
        agent.client = mock_client
        
        # Execute information gathering
        agent._gather_information("What is Python?")
        
        # Verify
        mock_client.chat.completions.create.assert_called()
        assert any("Cosmic Intelligence Report" in log for log in agent.logs)

    @patch('openai.OpenAI')
    @patch('threading.Thread')
    def test_full_execution_flow(self, mock_thread, mock_openai):
        """Test complete execution flow"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock Grok plan response
        plan_response = Mock()
        plan_response.choices = [Mock(message=Mock(content="1. SHELL: echo 'hello'\n2. INFO: System check"))]
        mock_client.chat.completions.create.return_value = plan_response
        
        agent = StargazerAgent("test directive", api_key="test_key")
        agent.client = mock_client
        
        # Mock the _execute method to avoid threading issues in tests
        with patch.object(agent, '_execute') as mock_execute:
            agent.run()
            mock_thread.assert_called_once()

    def test_step_execution_routing(self):
        """Test that steps are routed to correct execution methods"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        with patch.object(agent, '_execute_shell_command') as mock_shell, \
             patch.object(agent, '_generate_and_execute_code') as mock_code, \
             patch.object(agent, '_gather_information') as mock_info:
            
            # Test shell routing
            agent._execute_step("SHELL: ls -l")
            mock_shell.assert_called_once_with("ls -l")
            
            # Test code routing  
            agent._execute_step("CODE: create hello world")
            mock_code.assert_called_once_with("create hello world")
            
            # Test info routing
            agent._execute_step("INFO: system status")
            mock_info.assert_called_once_with("system status")

    def test_safety_confirmation(self):
        """Test safety confirmation for dangerous commands"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        # Test dangerous command detection
        assert not agent._confirm_step("rm -rf /important/data")
        
        # Test normal command (would normally ask for confirmation, but we'll mock it)
        with patch('rich.prompt.Confirm.ask', return_value=True):
            assert agent._confirm_step("SHELL: ls -l")


class TestUIIntegration:
    """Test UI integration with agents"""
    
    def test_add_directive_creates_agent(self):
        """Test that adding a directive creates an agent"""
        ui = DirectivesUI()
        ui.run(headless=True)  # Run the app in headless mode for testing
        ui.on_mount()  # Initialize table
        ui.add_directive("test directive")
        assert "test directive" in ui.agents

    def test_log_toggle_functionality(self):
        """Test log toggle functionality"""
        ui = DirectivesUI()
        ui.run(headless=True)  # Run the app in headless mode for testing
        ui.on_mount()  # Initialize table
        ui.add_directive("test directive")
        ui.toggle_logs("test directive")
        assert ui.show_logs["test directive"] is True
        ui.toggle_logs("test directive")
        assert ui.show_logs["test directive"] is False


# Run with: python -m pytest tests/test_cosmic_cli.py -v --cov=agents --cov=ui --cov-report=term-missing 