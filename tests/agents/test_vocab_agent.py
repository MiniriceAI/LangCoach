"""
Unit tests for vocab_agent module
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.agents.vocab_agent import VocabAgent


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
            session_id=None
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

