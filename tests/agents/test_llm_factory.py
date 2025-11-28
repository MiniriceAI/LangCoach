"""
Unit tests for llm_factory module
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agents.llm_factory import create_llm


class TestLLMFactory:
    """Test cases for create_llm function"""

    @patch('src.agents.llm_factory.os.getenv')
    @patch('src.agents.llm_factory.ChatOpenAI')
    @patch('src.agents.llm_factory.LOG')
    def test_create_llm_deepseek(self, mock_log, mock_chat_openai, mock_getenv):
        """Test create_llm with DeepSeek API key"""
        # Mock environment variables
        def getenv_side_effect(key, default=None):
            if key == "DEEPSEEK_API_KEY":
                return "test_deepseek_key"
            return None
        mock_getenv.side_effect = getenv_side_effect
        
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        result = create_llm()
        
        assert result == mock_llm
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"
        assert call_kwargs["openai_api_base"] == "https://api.deepseek.com"
        assert call_kwargs["openai_api_key"] == "test_deepseek_key"
        mock_log.info.assert_called_with("[LLM Provider] Using DeepSeek API")

    @patch('src.agents.llm_factory.os.getenv')
    @patch('src.agents.llm_factory.ChatOpenAI')
    @patch('src.agents.llm_factory.LOG')
    def test_create_llm_openai(self, mock_log, mock_chat_openai, mock_getenv):
        """Test create_llm with OpenAI API key"""
        # Mock environment variables - no DeepSeek, but has OpenAI
        def getenv_side_effect(key, default=None):
            if key == "DEEPSEEK_API_KEY":
                return None
            if key == "OPENAI_API_KEY":
                return "test_openai_key"
            if key == "OPENAI_MODEL":
                return default  # Use default
            return default
        mock_getenv.side_effect = getenv_side_effect
        
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        result = create_llm()
        
        assert result == mock_llm
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["openai_api_key"] == "test_openai_key"
        mock_log.info.assert_called()

    @patch('src.agents.llm_factory.os.getenv')
    @patch('src.agents.llm_factory.ChatOpenAI')
    def test_create_llm_openai_custom_model(self, mock_chat_openai, mock_getenv):
        """Test create_llm with OpenAI custom model"""
        def getenv_side_effect(key, default=None):
            if key == "DEEPSEEK_API_KEY":
                return None
            if key == "OPENAI_API_KEY":
                return "test_openai_key"
            if key == "OPENAI_MODEL":
                return "gpt-4"
            return default
        mock_getenv.side_effect = getenv_side_effect
        
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        result = create_llm()
        
        assert result == mock_llm
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == "gpt-4"

    @patch('src.agents.llm_factory.os.getenv')
    @patch('src.agents.llm_factory.ChatOllama')
    @patch('src.agents.llm_factory.LOG')
    def test_create_llm_ollama(self, mock_log, mock_chat_ollama, mock_getenv):
        """Test create_llm with Ollama fallback"""
        # Mock environment variables - no API keys
        def getenv_side_effect(key, default=None):
            if key == "DEEPSEEK_API_KEY":
                return None
            if key == "OPENAI_API_KEY":
                return None
            if key == "OLLAMA_BASE_URL":
                return default
            if key == "OLLAMA_MODEL":
                return default
            return default
        mock_getenv.side_effect = getenv_side_effect
        
        mock_llm = MagicMock()
        mock_chat_ollama.return_value = mock_llm
        
        result = create_llm()
        
        assert result == mock_llm
        mock_chat_ollama.assert_called_once()
        call_kwargs = mock_chat_ollama.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:11434"
        assert call_kwargs["model"] == "llama3.1:8b"
        mock_log.warning.assert_called()

    @patch('src.agents.llm_factory.os.getenv')
    @patch('src.agents.llm_factory.ChatOllama')
    def test_create_llm_ollama_custom_config(self, mock_chat_ollama, mock_getenv):
        """Test create_llm with Ollama custom configuration"""
        def getenv_side_effect(key, default=None):
            if key == "DEEPSEEK_API_KEY":
                return None
            if key == "OPENAI_API_KEY":
                return None
            if key == "OLLAMA_BASE_URL":
                return "http://custom:11434"
            if key == "OLLAMA_MODEL":
                return "custom-model"
            return default
        mock_getenv.side_effect = getenv_side_effect
        
        mock_llm = MagicMock()
        mock_chat_ollama.return_value = mock_llm
        
        result = create_llm()
        
        assert result == mock_llm
        call_kwargs = mock_chat_ollama.call_args[1]
        assert call_kwargs["base_url"] == "http://custom:11434"
        assert call_kwargs["model"] == "custom-model"

    @patch('src.agents.llm_factory.os.getenv')
    @patch('src.agents.llm_factory.ChatOllama')
    @patch('src.agents.llm_factory.LOG')
    def test_create_llm_ollama_error_handling(self, mock_log, mock_chat_ollama, mock_getenv):
        """Test create_llm with Ollama initialization error"""
        def getenv_side_effect(key, default=None):
            if key == "DEEPSEEK_API_KEY":
                return None
            if key == "OPENAI_API_KEY":
                return None
            if key == "OLLAMA_BASE_URL":
                return default
            if key == "OLLAMA_MODEL":
                return default
            return default
        mock_getenv.side_effect = getenv_side_effect
        
        mock_chat_ollama.side_effect = Exception("Connection error")
        
        with pytest.raises(ValueError) as exc_info:
            create_llm()
        
        assert "Failed to initialize Ollama model" in str(exc_info.value)
        mock_log.error.assert_called()

