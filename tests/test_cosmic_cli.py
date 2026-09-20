from unittest.mock import Mock, patch, MagicMock
import os
from cosmic_cli.agents import StargazerAgent, DEFAULT_MODEL
from cosmic_cli.ui import DirectivesUI


class TestDefaults:
    def test_default_model_is_latest(self):
        assert DEFAULT_MODEL == "grok-4.5" or os.getenv("COSMIC_GROK_MODEL")


class TestCosmicBanner:
    @patch("pyfiglet.Figlet")
    def test_banner_generation(self, mock_figlet):
        mock_figlet.return_value.renderText.return_value = "COSMIC CLI ART"
        ui = DirectivesUI()
        ui.figlet = mock_figlet.return_value
        banner_text = ui.figlet.renderText("COSMIC CLI")
        assert "COSMIC CLI ART" == banner_text
        assert "CODER" not in banner_text.upper()

    @patch("pyfiglet.Figlet")
    def test_ui_compose_banner(self, mock_figlet):
        mock_figlet.return_value.renderText.return_value = "COSMIC CLI ART"
        ui = DirectivesUI()
        ui.figlet = mock_figlet.return_value
        banner_text = ui.figlet.renderText("COSMIC CLI")
        assert "COSMIC CLI ART" == banner_text


class TestStargazerAgent:
    def setup_method(self):
        os.environ["EXEC_MODE"] = "safe"
        os.environ["XAI_API_KEY"] = "test_key"

    @patch("cosmic_cli.agents.Client")
    def test_agent_initialization(self, mock_Client):
        mock_Client.return_value = Mock()
        agent = StargazerAgent("test directive", api_key="test_key")
        assert agent.directive == "test directive"
        assert agent.status == "ready"
        assert agent.exec_mode == "safe"
        assert agent.model == DEFAULT_MODEL
        assert len(agent.logs) == 0

    def test_parse_step_and_prefer_finish(self):
        assert StargazerAgent.parse_step("```\nREAD: foo.py\n```").startswith("READ:")
        assert StargazerAgent.parse_step("FINISH: done").startswith("FINISH:")
        assert StargazerAgent.parse_step("PASS: no credits").startswith("PASS:")
        # Prefer FINISH when both present
        both = "READ: a.py\nFINISH: 3 lines"
        assert StargazerAgent.parse_step(both).startswith("FINISH:")

    def test_execute_loop_finish_with_echo(self, tmp_path, monkeypatch):
        echo = tmp_path / "echo.jsonl"
        monkeypatch.setattr("cosmic_cli.agents.ECHO_FILE", echo)
        agent = StargazerAgent(
            "count lines",
            api_key="test_key",
            write_echo=True,
            max_steps=3,
            show_progress=False,
            quiet=True,
        )
        with patch.object(
            agent,
            "_ask_grok_for_next_step",
            side_effect=["READ: check_agents_path.py", "FINISH: 3 lines"],
        ), patch.object(
            agent.context_manager, "read_file", return_value="a\nb\nc\n"
        ):
            result = agent.execute()
        # No verifier was supplied: the model's word alone is not a verdict.
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "model_declared"
        assert result["results"][-1]["result"] == "3 lines"
        assert echo.exists()
        assert "count lines" in echo.read_text()

    def test_reread_uses_cache(self):
        agent = StargazerAgent("t", api_key="test_key", quiet=True, show_progress=False)
        with patch.object(
            agent.context_manager, "read_file", return_value="hello"
        ) as mock_read:
            a = agent._execute_step("READ: foo.py")
            b = agent._execute_step("READ: foo.py")
            assert a == b == "hello"
            assert mock_read.call_count == 1

    def test_edit_requires_prior_read(self, tmp_path, monkeypatch):
        agent = StargazerAgent(
            "t",
            api_key="test_key",
            quiet=True,
            show_progress=False,
            work_dir=str(tmp_path),
        )
        (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
        out = agent._execute_step("EDIT: f.py|||x = 1|||x = 2")
        assert "READ-before-EDIT" in out
        agent._execute_step("READ: f.py")
        out = agent._execute_step("EDIT: f.py|||x = 1\n|||x = 2\n")
        assert "EDIT ok" in out
        assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 2\n"

    def test_parse_prefers_finish_and_knows_grep(self):
        assert StargazerAgent.parse_step("GREP: foo").startswith("GREP:")
        assert StargazerAgent.parse_step("GLOB: **/*.py").startswith("GLOB:")

    def test_parse_step_preserves_trailing_quotes(self):
        raw = 'EDIT: note.py|||greeting = "hello"|||greeting = "hello cosmos"'
        parsed = StargazerAgent.parse_step(raw)
        assert parsed.endswith('"')
        assert "hello cosmos\"" in parsed

    def test_loop_breaker_on_repeat_read(self, tmp_path, monkeypatch):
        echo = tmp_path / "echo.jsonl"
        monkeypatch.setattr("cosmic_cli.agents.ECHO_FILE", echo)
        # directive has no mutation words → pure discovery loop trips FINISH
        agent = StargazerAgent(
            "count lines only",
            api_key="test_key",
            write_echo=True,
            max_steps=6,
            show_progress=False,
            quiet=True,
        )
        with patch.object(
            agent,
            "_ask_grok_for_next_step",
            side_effect=["READ: f.py", "READ: f.py", "READ: f.py", "READ: f.py"],
        ), patch.object(
            agent.context_manager, "read_file", return_value="line1\nline2\n"
        ):
            result = agent.execute()
        # The model never said FINISH; the harness wrote one to stop the loop.
        assert result["status"] == "needs_review"
        assert result["finish_basis"] == "synthesized"
        assert result["results"][-1]["step"] == "FINISH"

    def test_productive_shell_does_not_trigger_discovery_steer(self, tmp_path, monkeypatch):
        """date/version after READ must run; not be steered as discovery thrash."""
        from cosmic_cli.agents import _is_discovery_step

        assert _is_discovery_step("SHELL: LS -LA", "SHELL: ls -la")
        assert _is_discovery_step("SHELL: FIND .", "SHELL: find . -name x")
        assert not _is_discovery_step(
            "SHELL: DATE -U", "SHELL: date -u +%Y-%m-%dT%H:%M:%SZ"
        )
        assert not _is_discovery_step(
            "SHELL: COSMIC-CLI --VERSION", "SHELL: cosmic-cli --version"
        )

        echo = tmp_path / "echo.jsonl"
        monkeypatch.setattr("cosmic_cli.agents.ECHO_FILE", echo)
        agent = StargazerAgent(
            "write a file after checking version",
            api_key="test_key",
            write_echo=True,
            max_steps=6,
            show_progress=False,
            quiet=True,
            work_dir=str(tmp_path),
            auto_verify=False,
            use_helix=False,
        )
        shell_calls: list[str] = []

        def fake_shell(cmd: str) -> str:
            shell_calls.append(cmd)
            return f"[exit 0]\n{cmd}"

        with patch.object(
            agent,
            "_ask_grok_for_next_step",
            side_effect=[
                "READ: f.py",
                "SHELL: cosmic-cli --version",
                "SHELL: date -u +%Y-%m-%dT%H:%M:%SZ",
                "CREATE: out.txt|||ok\n",
                "FINISH: done",
            ],
        ), patch.object(
            agent.context_manager, "read_file", return_value="x\n"
        ), patch.object(agent, "_run_shell", side_effect=fake_shell):
            result = agent.execute()
        assert "cosmic-cli --version" in shell_calls
        assert any("date" in c for c in shell_calls), f"date never ran: {shell_calls}"
        assert result["status"] == "needs_review"

    def test_mutation_directive_steers_instead_of_false_finish(self, tmp_path, monkeypatch):
        echo = tmp_path / "echo.jsonl"
        monkeypatch.setattr("cosmic_cli.agents.ECHO_FILE", echo)
        (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
        agent = StargazerAgent(
            "change x to 2 in f.py",
            api_key="test_key",
            write_echo=True,
            max_steps=5,
            show_progress=False,
            quiet=True,
            work_dir=str(tmp_path),
            auto_verify=False,
        )
        with patch.object(
            agent,
            "_ask_grok_for_next_step",
            side_effect=[
                "READ: f.py",
                "READ: f.py",  # first repeat → steer, skip
                "EDIT: f.py|||x = 1\n|||x = 2\n",
                "FINISH: done",
            ],
        ):
            result = agent.execute()
        assert result["status"] == "needs_review"
        assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 2\n"
        assert "f.py" in result.get("edited", [])

    @patch("subprocess.run")
    def test_shell_command_execution_safe_mode(self, mock_subprocess, monkeypatch):
        # Floor usable so wrap returns argv; avoid live probe polluting the mock.
        monkeypatch.setattr(
            "cosmic_cli.sandbox.seatbelt_applies", lambda force_probe=False: True
        )
        monkeypatch.setattr(
            "cosmic_cli.sandbox.sandbox_available", lambda: True
        )
        monkeypatch.setattr(
            "cosmic_cli.sandbox.wrap_argv_for_l0_shell",
            lambda cmd, force_bare=False, **kw: ["/bin/zsh", "-c", cmd],
        )
        agent = StargazerAgent(
            "test directive", api_key="test_key", quiet=True, use_helix=False
        )
        mock_subprocess.return_value = Mock(
            stdout="file1.txt\nfile2.txt", stderr="", returncode=0
        )
        result = agent._run_shell("ls -l")
        mock_subprocess.assert_called_once()
        assert "file1.txt" in result

    def test_dangerous_command_blocked(self):
        agent = StargazerAgent(
            "test directive", api_key="test_key", quiet=True, use_helix=False
        )
        result = agent._run_shell("rm -rf /")
        assert "BLOCKED" in result
        assert "BLOCKED" in agent._run_shell("RM -RF /tmp/x")

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_code_execution_via_step(self, mock_tempfile, mock_subprocess):
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.py"
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        mock_subprocess.return_value = Mock(
            stdout="Hello, World!\n", stderr="", returncode=0
        )
        agent = StargazerAgent("test directive", api_key="test_key", quiet=True)
        with patch.object(agent, "_run_code", return_value="Hello, World!\n") as mock_run:
            result = agent._execute_step("CODE: print('Hello, World!')")
            mock_run.assert_called_once_with("print('Hello, World!')")
            assert "Hello, World!" in result

    @patch.object(StargazerAgent, "_ask_grok_for_info", return_value="Python is a language")
    def test_information_via_info_step(self, mock_ask_info):
        agent = StargazerAgent("test directive", api_key="test_key", quiet=True)
        result = agent._execute_step("INFO: What is Python?")
        mock_ask_info.assert_called()
        assert "Python is a language" in result
        assert any("INFO" in log for log in agent.logs)

    @patch("threading.Thread")
    def test_full_execution_flow(self, mock_thread):
        agent = StargazerAgent("test directive", api_key="test_key", quiet=True)
        with patch.object(agent, "execute"):
            agent.run()
            mock_thread.assert_called_once()

    def test_step_execution_routing(self):
        agent = StargazerAgent("test directive", api_key="test_key", quiet=True)
        with patch.object(agent, "_run_shell") as mock_shell, patch.object(
            agent, "_run_code"
        ) as mock_code, patch.object(agent, "_ask_grok_for_info") as mock_info:
            agent._execute_step("SHELL: ls -l")
            mock_shell.assert_called_once_with("ls -l")
            agent._execute_step("CODE: create hello world")
            mock_code.assert_called_once_with("create hello world")
            agent._execute_step("INFO: system status")
            mock_info.assert_called_once_with("system status")

    def test_safety_block_in_run_shell(self, monkeypatch):
        agent = StargazerAgent(
            "test directive", api_key="test_key", quiet=True, use_helix=False
        )
        result = agent._run_shell("rm -rf /important/data")
        assert "BLOCKED" in result
        # Floor usable for the allow path (nested hosts fail-closed without this).
        monkeypatch.setattr(
            "cosmic_cli.sandbox.wrap_argv_for_l0_shell",
            lambda cmd, force_bare=False, **kw: ["/bin/zsh", "-c", cmd],
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="ok", stderr="", returncode=0)
            result = agent._run_shell("ls -l")
            assert "ok" in result

class TestUIIntegration:
    def test_add_directive_creates_agent(self):
        ui = DirectivesUI(testing=True)
        with patch.dict(os.environ, {"XAI_API_KEY": "test_key"}), patch(
            "cosmic_cli.ui.StargazerAgent"
        ) as mock_agent_cls, patch.object(ui, "_refresh_panel"):
            mock_agent = Mock()
            mock_agent.status = "✨"
            mock_agent.logs = []
            mock_agent_cls.return_value = mock_agent
            ui.add_directive("test directive")
            assert "test directive" in ui.agents

    def test_log_toggle_functionality(self):
        ui = DirectivesUI(testing=True)
        with patch.dict(os.environ, {"XAI_API_KEY": "test_key"}), patch(
            "cosmic_cli.ui.StargazerAgent"
        ) as mock_agent_cls, patch.object(ui, "_refresh_panel"):
            mock_agent = Mock()
            mock_agent.status = "✨"
            mock_agent.logs = []
            mock_agent_cls.return_value = mock_agent
            ui.add_directive("test directive")
            ui.toggle_logs("test directive")
            assert ui.show_logs["test directive"] is True
            ui.toggle_logs("test directive")
            assert ui.show_logs["test directive"] is False

    def test_two_directives_keep_two_agents(self):
        ui = DirectivesUI(testing=True)
        with patch.dict(os.environ, {"XAI_API_KEY": "test_key"}), patch(
            "cosmic_cli.ui.StargazerAgent"
        ) as mock_agent_cls, patch.object(ui, "_refresh_panel"):
            first, second = Mock(), Mock()
            first.status = "ready"
            first.logs = []
            first.mission_id = "M1"
            second.status = "ready"
            second.logs = []
            second.mission_id = "M2"
            mock_agent_cls.side_effect = [first, second]
            ui.add_directive("alpha")
            ui.add_directive("beta")
            assert set(ui.agents) == {"alpha", "beta"}
            assert ui.agents["alpha"] is first
            assert ui.agents["beta"] is second
