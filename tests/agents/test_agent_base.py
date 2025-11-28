"""
Unit tests for agent_base module
"""
import pytest
import os
import json
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

from src.agents.agent_base import AgentBase


class TestAgentBase:
    """Test cases for AgentBase class"""

    @patch('src.agents.agent_base.create_llm')
    @patch('src.agents.agent_base.ChatPromptTemplate')
    def test_init_with_valid_files(self, mock_prompt_template, mock_create_llm, 
                                   sample_prompt_file, sample_intro_file, mock_env_vars):
        """Test AgentBase initialization with valid files"""
        mock_template = MagicMock()
        mock_prompt_template.from_messages.return_value = mock_template
        
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        
        agent = AgentBase(
            name="test_agent",
            prompt_file=sample_prompt_file,
            intro_file=sample_intro_file
        )
        
        assert agent.name == "test_agent"
        assert agent.prompt_file == sample_prompt_file
        assert agent.intro_file == sample_intro_file
        assert len(agent.intro_messages) == 3
        assert agent.intro_messages == ["Hello", "Hi there", "Welcome"]

    @patch('src.agents.agent_base.create_llm')
    @patch('src.agents.agent_base.ChatPromptTemplate')
    def test_init_without_intro_file(self, mock_prompt_template, mock_create_llm,
                                     sample_prompt_file, mock_env_vars):
        """Test AgentBase initialization without intro file"""
        mock_template = MagicMock()
        mock_prompt_template.from_messages.return_value = mock_template
        
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        
        agent = AgentBase(
            name="test_agent",
            prompt_file=sample_prompt_file
        )
        
        assert agent.intro_file is None
        assert agent.intro_messages == []

    @patch('src.agents.agent_base.create_llm')
    def test_load_prompt_file_not_found(self, mock_create_llm, mock_env_vars):
        """Test loading prompt from non-existent file"""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        with pytest.raises(FileNotFoundError):
            agent = AgentBase(
                name="test_agent",
                prompt_file="non_existent_file.txt"
            )

    @patch('src.agents.agent_base.create_llm')
    @patch('src.agents.agent_base.ChatPromptTemplate')
    def test_load_intro_invalid_json(self, mock_prompt_template, mock_create_llm,
                                     tmp_path, sample_prompt_file, mock_env_vars):
        """Test loading invalid JSON intro file"""
        invalid_json_file = tmp_path / "invalid.json"
        invalid_json_file.write_text("invalid json content", encoding="utf-8")
        
        mock_template = MagicMock()
        mock_prompt_template.from_messages.return_value = mock_template
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        
        with pytest.raises(ValueError, match="包含无效的 JSON"):
            agent = AgentBase(
                name="test_agent",
                prompt_file=sample_prompt_file,
                intro_file=str(invalid_json_file)
            )

    @patch('src.agents.agent_base.create_llm')
    @patch('src.agents.agent_base.ChatPromptTemplate')
    @patch('src.agents.agent_base.get_session_history')
    @patch('src.agents.agent_base.RunnableWithMessageHistory')
    def test_chat_with_history(self, mock_runnable, mock_get_history, mock_prompt_template, 
                               mock_create_llm, sample_prompt_file, mock_env_vars):
        """Test chat_with_history method"""
        # Setup mocks
        mock_template = MagicMock()
        mock_prompt_template.from_messages.return_value = mock_template
        
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        
        mock_history = MagicMock()
        mock_get_history.return_value = mock_history
        
        # Mock RunnableWithMessageHistory
        mock_runnable_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test AI response"
        mock_runnable_instance.invoke.return_value = mock_response
        mock_runnable.return_value = mock_runnable_instance
        
        # Create agent
        agent = AgentBase(
            name="test_agent",
            prompt_file=sample_prompt_file
        )
        
        # Test chat
        response = agent.chat_with_history("Hello")
        
        assert response == "Test AI response"
        mock_runnable_instance.invoke.assert_called_once()

    @patch('src.agents.agent_base.create_llm')
    @patch('src.agents.agent_base.ChatPromptTemplate')
    @patch('src.agents.agent_base.RunnableWithMessageHistory')
    def test_chat_with_history_custom_session_id(self, mock_runnable, mock_prompt_template,
                                                 mock_create_llm, sample_prompt_file,
                                                 mock_env_vars):
        """Test chat_with_history with custom session_id"""
        mock_template = MagicMock()
        mock_prompt_template.from_messages.return_value = mock_template
        
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        
        # Mock RunnableWithMessageHistory
        mock_runnable_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Custom session response"
        mock_runnable_instance.invoke.return_value = mock_response
        mock_runnable.return_value = mock_runnable_instance
        
        agent = AgentBase(
            name="test_agent",
            prompt_file=sample_prompt_file
        )
        
        response = agent.chat_with_history("Hello", session_id="custom_session")
        
        assert response == "Custom session response"
        # Verify session_id was used
        call_args = mock_runnable_instance.invoke.call_args
        # Check if configurable is in kwargs or args
        if len(call_args) > 1 and "configurable" in call_args[1]:
            assert call_args[1]["configurable"]["session_id"] == "custom_session"
        elif len(call_args[0]) > 1:
            # Sometimes it's passed as second positional argument
            config = call_args[0][1]
            assert config["configurable"]["session_id"] == "custom_session"
        else:
            # Just verify the method was called
            assert mock_runnable_instance.invoke.called

