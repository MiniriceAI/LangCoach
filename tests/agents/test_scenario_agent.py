"""
Unit tests for scenario_agent module
"""
import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.agents.scenario_agent import ScenarioAgent
from langchain_core.messages import AIMessage


class TestScenarioAgent:
    """Test cases for ScenarioAgent class"""

    @patch('src.agents.scenario_agent.AgentBase.__init__')
    @patch('src.agents.scenario_agent.get_session_history')
    def test_init(self, mock_get_history, mock_base_init, tmp_path):
        """Test ScenarioAgent initialization"""
        # Create test files
        prompt_file = tmp_path / "prompts" / "test_scenario_prompt.txt"
        prompt_file.parent.mkdir(parents=True)
        prompt_file.write_text("Test prompt")
        
        intro_file = tmp_path / "content" / "intro" / "test_scenario.json"
        intro_file.parent.mkdir(parents=True)
        intro_file.write_text(json.dumps(["Intro 1", "Intro 2"]), encoding="utf-8")
        
        mock_base_init.return_value = None
        mock_history = MagicMock()
        mock_history.messages = []
        mock_get_history.return_value = mock_history
        
        agent = ScenarioAgent("test_scenario")
        
        # Verify parent init was called with correct paths
        mock_base_init.assert_called_once()
        call_kwargs = mock_base_init.call_args[1]
        assert "test_scenario_prompt.txt" in call_kwargs["prompt_file"]
        assert "test_scenario.json" in call_kwargs["intro_file"]

    @patch('src.agents.scenario_agent.AgentBase.__init__')
    @patch('src.agents.scenario_agent.get_session_history')
    @patch('src.agents.scenario_agent.random.choice')
    def test_start_new_session_empty_history(self, mock_choice, mock_get_history, 
                                            mock_base_init, tmp_path):
        """Test start_new_session with empty history"""
        mock_base_init.return_value = None
        mock_history = MagicMock()
        mock_history.messages = []
        mock_get_history.return_value = mock_history
        
        # Create agent with mocked intro messages
        agent = ScenarioAgent("test_scenario")
        agent.intro_messages = ["Message 1", "Message 2", "Message 3"]
        agent.session_id = "test_session"
        
        mock_choice.return_value = "Message 2"
        
        result = agent.start_new_session()
        
        assert result == "Message 2"
        mock_history.add_message.assert_called_once()
        assert isinstance(mock_history.add_message.call_args[0][0], AIMessage)

    @patch('src.agents.scenario_agent.AgentBase.__init__')
    @patch('src.agents.scenario_agent.get_session_history')
    @patch('src.agents.scenario_agent.random.choice')
    def test_start_new_session_existing_history(self, mock_choice, mock_get_history, mock_base_init):
        """Test start_new_session with existing history - should clear history and start new session"""
        mock_base_init.return_value = None
        mock_history = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Last message"
        mock_history.messages = [mock_message]  # 模拟已有历史记录
        mock_get_history.return_value = mock_history
        
        agent = ScenarioAgent("test_scenario")
        agent.intro_messages = ["Message 1", "Message 2", "Message 3"]  # 设置 intro_messages
        agent.session_id = "test_session"
        
        mock_choice.return_value = "Message 2"
        
        result = agent.start_new_session()
        
        # 现在总是清除历史并创建新会话
        assert result == "Message 2"
        mock_history.clear.assert_called_once()  # 应该清除历史
        mock_history.add_message.assert_called_once()  # 应该添加新的初始消息

    @patch('src.agents.scenario_agent.AgentBase.__init__')
    @patch('src.agents.scenario_agent.get_session_history')
    def test_start_new_session_custom_session_id(self, mock_get_history, mock_base_init):
        """Test start_new_session with custom session_id"""
        mock_base_init.return_value = None
        mock_history = MagicMock()
        mock_history.messages = []
        mock_get_history.return_value = mock_history
        
        agent = ScenarioAgent("test_scenario")
        agent.intro_messages = ["Message 1"]
        
        with patch('src.agents.scenario_agent.random.choice', return_value="Message 1"):
            agent.start_new_session(session_id="custom_session")
        
        mock_get_history.assert_called_with("custom_session")

