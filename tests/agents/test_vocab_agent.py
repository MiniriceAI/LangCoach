"""
Unit tests for vocab_agent module
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.agents.vocab_agent import VocabAgent
from src.agents.conversation_config import create_config, DifficultyLevel


class TestVocabAgent:
    """Test cases for VocabAgent class"""

    @patch('src.agents.vocab_agent.AgentBase.__init__')
    def test_init(self, mock_base_init):
        """Test VocabAgent initialization"""
        mock_base_init.return_value = None

        agent = VocabAgent()

        mock_base_init.assert_called_once_with(
            name="vocab_study",
            prompt_file="prompts/vocab_study_prompt.txt",
            session_id=None,
            config=None
        )

    @patch('src.agents.vocab_agent.AgentBase.__init__')
    @patch('src.agents.vocab_agent.get_session_history')
    def test_restart_session(self, mock_get_history, mock_base_init):
        """Test restart_session method"""
        mock_base_init.return_value = None

        agent = VocabAgent()
        agent.session_id = "test_session"

        mock_history = MagicMock()
        mock_get_history.return_value = mock_history

        result = agent.restart_session()

        assert result is mock_history
        mock_history.clear.assert_called_once()
        mock_get_history.assert_called_with("test_session")

    @patch('src.agents.vocab_agent.AgentBase.__init__')
    @patch('src.agents.vocab_agent.get_session_history')
    def test_restart_session_custom_id(self, mock_get_history, mock_base_init):
        """Test restart_session with custom session_id"""
        mock_base_init.return_value = None

        agent = VocabAgent()
        agent.session_id = "default_session"

        mock_history = MagicMock()
        mock_get_history.return_value = mock_history

        agent.restart_session(session_id="custom_session")

        mock_get_history.assert_called_with("custom_session")
        mock_history.clear.assert_called_once()


class TestVocabAgentWithConfig:
    """Test cases for VocabAgent with ConversationConfig (Phase 1)"""

    @patch('src.agents.vocab_agent.AgentBase.__init__')
    def test_init_with_config(self, mock_base_init):
        """Test VocabAgent initialization with custom config"""
        mock_base_init.return_value = None

        custom_config = create_config(difficulty="advanced")

        agent = VocabAgent(config=custom_config)

        mock_base_init.assert_called_once_with(
            name="vocab_study",
            prompt_file="prompts/vocab_study_prompt.txt",
            session_id=None,
            config=custom_config
        )

    @patch('src.agents.vocab_agent.AgentBase.__init__')
    @patch('src.agents.vocab_agent.AgentBase.update_config')
    @patch('src.agents.vocab_agent.get_session_history')
    def test_restart_session_with_config(self, mock_get_history, mock_update_config, mock_base_init):
        """Test restart_session with new config"""
        mock_base_init.return_value = None

        agent = VocabAgent()
        agent.session_id = "test_session"

        mock_history = MagicMock()
        mock_get_history.return_value = mock_history

        new_config = create_config(difficulty="primary")
        agent.restart_session(config=new_config)

        # Verify update_config was called with new config
        mock_update_config.assert_called_once_with(new_config)
        mock_history.clear.assert_called_once()

