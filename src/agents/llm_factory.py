"""
LLM 工厂模块，用于创建和管理不同的 LLM 提供者实例。
支持 DeepSeek、OpenAI 和 Ollama 三种 LLM 提供者。
"""
import os

from langchain_openai import ChatOpenAI  # 导入 ChatOpenAI 模型
from langchain_ollama.chat_models import ChatOllama  # 导入 ChatOllama 模型
from langchain_core.language_models.chat_models import BaseChatModel  # 导入基础聊天模型类

from utils.logger import LOG  # 导入日志工具


def create_llm() -> BaseChatModel:
    """
    创建 LLM 实例，支持多种 LLM 提供者。
    优先级：DeepSeek > OpenAI > Ollama
    
    环境变量说明:
    - DEEPSEEK_API_KEY: DeepSeek API 密钥（优先使用）
    - OPENAI_API_KEY: OpenAI API 密钥
    - OPENAI_MODEL: OpenAI 模型名称（默认: gpt-4o-mini）
    - OLLAMA_BASE_URL: Ollama 服务地址（默认: http://localhost:11434）
    - OLLAMA_MODEL: Ollama 模型名称（默认: llama3.1:8b-instruct-q8_0）
    
    返回:
        BaseChatModel: LLM 实例
        
    异常:
        ValueError: 如果未配置任何 LLM 提供者
    """
    # 优先使用 DeepSeek
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        LOG.info("[LLM Provider] Using DeepSeek API")
        return ChatOpenAI(
            model="deepseek-chat",
            openai_api_base="https://api.deepseek.com",
            openai_api_key=deepseek_api_key,
            max_tokens=8192,
            temperature=0.8,
        )
    
    # 其次使用 OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        LOG.info(f"[LLM Provider] Using OpenAI API with model {openai_model}")
        return ChatOpenAI(
            model=openai_model,
            openai_api_key=openai_api_key,
            max_tokens=8192,
            temperature=0.8,
        )
    
    # 最后使用 Ollama (本地部署)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    LOG.warning(
        f"[LLM Provider] No API keys found (DEEPSEEK_API_KEY, OPENAI_API_KEY). "
        f"Falling back to Ollama at {ollama_base_url} with model {ollama_model}. "
        f"Make sure Ollama is running locally and the model is available."
    )
    
    # 尝试创建 Ollama 模型，如果失败会抛出异常
    try:
        return ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.8,
            num_predict=8192,  # Ollama 的参数名是 num_predict
        )
    except Exception as e:
        error_msg = (
            f"Failed to initialize Ollama model '{ollama_model}' at {ollama_base_url}. "
            f"Please ensure:\n"
            f"1. Ollama is running (check: ollama serve)\n"
            f"2. The model is available (check: ollama list)\n"
            f"3. Install the model if needed (run: ollama pull {ollama_model})\n"
            f"4. Set OLLAMA_MODEL environment variable to use a different model\n"
            f"Error details: {str(e)}"
        )
        LOG.error(f"[LLM Provider] {error_msg}")
        raise ValueError(error_msg) from e

