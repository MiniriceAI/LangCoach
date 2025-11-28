"""
Unit tests for session_history module
"""
import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.session_history import get_session_history, store


class TestSessionHistory:
    """Test cases for session history management"""

    def test_get_session_history_new_session(self, clean_session_store):
        """Test creating a new session history"""
        history = get_session_history("test_session_1")
        assert isinstance(history, InMemoryChatMessageHistory)
        assert len(history.messages) == 0

    def test_get_session_history_existing_session(self, clean_session_store):
        """Test retrieving an existing session history"""
        session_id = "test_session_2"
        history1 = get_session_history(session_id)
        history1.add_message(HumanMessage(content="Hello"))
        
        history2 = get_session_history(session_id)
        assert history1 is history2
        assert len(history2.messages) == 1

    def test_get_session_history_multiple_sessions(self, clean_session_store):
        """Test multiple independent sessions"""
        history1 = get_session_history("session_1")
        history2 = get_session_history("session_2")
        
        assert history1 is not history2
        
        history1.add_message(HumanMessage(content="Message 1"))
        history2.add_message(AIMessage(content="Message 2"))
        
        assert len(history1.messages) == 1
        assert len(history2.messages) == 1
        assert history1.messages[0].content == "Message 1"
        assert history2.messages[0].content == "Message 2"

    def test_session_history_clear(self, clean_session_store):
        """Test clearing session history"""
        session_id = "test_clear_session"
        history = get_session_history(session_id)
        history.add_message(HumanMessage(content="Test message"))
        assert len(history.messages) == 1
        
        history.clear()
        assert len(history.messages) == 0

    def test_session_history_add_messages(self, clean_session_store):
        """Test adding messages to session history"""
        history = get_session_history("test_add_messages")
        
        history.add_message(HumanMessage(content="User message"))
        history.add_message(AIMessage(content="AI response"))
        
        assert len(history.messages) == 2
        assert history.messages[0].content == "User message"
        assert history.messages[1].content == "AI response"

