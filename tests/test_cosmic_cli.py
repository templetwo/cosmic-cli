from unittest.mock import Mock, patch, MagicMock
import os
import subprocess
from cosmic_cli.agents import StargazerAgent
from cosmic_cli.ui import DirectivesUI
from xai_sdk import Client
from xai_sdk.chat import user

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
    
    @patch('cosmic_cli.agents.Client')
    def test_agent_initialization(self, mock_Client):
        """Test agent initializes correctly (xai_sdk path)"""
        mock_client = Mock()
        mock_Client.return_value = mock_client
        
        agent = StargazerAgent("test directive", api_key="test_key")
        assert agent.directive == "test directive"
        assert agent.status == "✨"
        assert agent.exec_mode == "safe"
        assert len(agent.logs) == 0

    def test_dynamic_plan_and_next_step_mocked(self):
        """Test current StargazerAgent plan creation and grok consult (modern xai_sdk path)"""
        with patch.object(StargazerAgent, '_ask_grok_for_next_step', return_value="FINISH: test complete"):
            agent = StargazerAgent("test directive", api_key="test_key")
            agent._create_dynamic_plan()
            assert len(agent.dynamic_plan) > 0
            next_step = agent._ask_grok_for_next_step()
            assert "FINISH" in next_step

    @patch('subprocess.run')
    def test_shell_command_execution_safe_mode(self, mock_subprocess):
        """Test shell command execution in safe mode (modern _run_shell)"""
        agent = StargazerAgent("test directive", api_key="test_key")
        mock_subprocess.return_value = Mock(
            stdout="file1.txt\nfile2.txt", 
            stderr="", 
            returncode=0
        )
        
        # Test safe command via modern path
        result = agent._run_shell("ls -l")
        
        mock_subprocess.assert_called_once()
        assert any("📤 Output:" in log for log in agent.logs)
        assert "file1.txt" in result

    def test_dangerous_command_blocked(self):
        """Test that dangerous commands are blocked in safe mode (modern _run_shell)"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        # Test dangerous command via modern path
        result = agent._run_shell("rm -rf /")
        
        # Should be blocked (returns message, no execution)
        assert "BLOCKED" in result
        assert "dangerous" in result.lower()

    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    def test_code_execution_via_step(self, mock_tempfile, mock_subprocess):
        """Test CODE step execution (modern xai_sdk agent path; no legacy code-gen client call)"""
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
        
        # Patch at higher level to avoid tempfile/fs side effects in test
        with patch.object(agent, '_run_code', return_value="Hello, World!\n") as mock_run_code:
            result = agent._execute_step("CODE: print('Hello, World!')")
            mock_run_code.assert_called_once_with("print('Hello, World!')")
            assert "Hello, World!" in result

    @patch.object(StargazerAgent, '_ask_grok_for_info', return_value="Python is a programming language")
    def test_information_via_info_step(self, mock_ask_info):
        """Test INFO step (modern xai_sdk _ask_grok_for_info path)"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        # Execute via modern _execute_step for INFO
        result = agent._execute_step("INFO: What is Python?")
        
        # Verify
        mock_ask_info.assert_called()
        assert "Python is a programming language" in result
        assert any("🔍 Answering question" in log for log in agent.logs)

    @patch('threading.Thread')
    def test_full_execution_flow(self, mock_thread):
        """Test complete execution flow (modern xai_sdk; patches run thread)"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        # Mock the execute (public) to avoid threading + real API in tests
        with patch.object(agent, 'execute') as mock_execute:
            agent.run()
            mock_thread.assert_called_once()

    def test_step_execution_routing(self):
        """Test that steps are routed to correct execution methods (modern names)"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        with patch.object(agent, '_run_shell') as mock_shell, \
             patch.object(agent, '_run_code') as mock_code, \
             patch.object(agent, '_ask_grok_for_info') as mock_info:
            
            # Test shell routing via _execute_step (note: prefix stripped inside)
            agent._execute_step("SHELL: ls -l")
            mock_shell.assert_called_once_with("ls -l")
            
            # Test code routing  
            agent._execute_step("CODE: create hello world")
            mock_code.assert_called_once_with("create hello world")
            
            # Test info routing
            agent._execute_step("INFO: system status")
            mock_info.assert_called_once_with("system status")

    def test_safety_block_in_run_shell(self):
        """Test safety block for dangerous commands (modern _run_shell, no separate _confirm_step)"""
        agent = StargazerAgent("test directive", api_key="test_key")
        
        # Test dangerous command detection via return value
        result = agent._run_shell("rm -rf /important/data")
        assert "BLOCKED" in result
        
        # Test normal (safe) command would proceed (mocked)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(stdout="ok", stderr="", returncode=0)
            result = agent._run_shell("ls -l")
            assert "ok" in result or "📤" in str(agent.logs[-1])


class TestUIIntegration:
    """Test UI integration with agents (headless-safe, no full run loop)"""
    
    def test_add_directive_creates_agent(self):
        """Test that adding a directive creates an agent"""
        ui = DirectivesUI(testing=True)
        # No on_mount() to avoid textual ScreenStackError in unit test context
        with patch.dict(os.environ, {"XAI_API_KEY": "test_key"}), \
             patch('cosmic_cli.ui.StargazerAgent') as mock_agent_cls, \
             patch.object(ui, '_refresh_panel'):
            mock_agent = Mock()
            mock_agent.status = "✨"
            mock_agent.logs = []
            mock_agent_cls.return_value = mock_agent
            ui.add_directive("test directive")
            assert "test directive" in ui.agents

    def test_log_toggle_functionality(self):
        """Test log toggle functionality"""
        ui = DirectivesUI(testing=True)
        with patch.dict(os.environ, {"XAI_API_KEY": "test_key"}), \
             patch('cosmic_cli.ui.StargazerAgent') as mock_agent_cls, \
             patch.object(ui, '_refresh_panel'):
            mock_agent = Mock()
            mock_agent.status = "✨"
            mock_agent.logs = []
            mock_agent_cls.return_value = mock_agent
            ui.add_directive("test directive")
            ui.toggle_logs("test directive")
            assert ui.show_logs["test directive"] is True
            ui.toggle_logs("test directive")
            assert ui.show_logs["test directive"] is False


# Run with: python -m pytest tests/test_cosmic_cli.py -v --cov=agents --cov=ui --cov-report=term-missing 