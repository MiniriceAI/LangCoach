"""
LLM 工厂模块，用于创建和管理不同的 LLM 提供者实例。
支持 Ollama、DeepSeek 和 OpenAI 三种 LLM 提供者。

Phase 2 更新：支持从配置文件读取，支持可配置的提供者优先级。
默认提供者：Ollama (unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL)
"""

from langchain_openai import ChatOpenAI  # 导入 ChatOpenAI 模型
from langchain_ollama.chat_models import ChatOllama  # 导入 ChatOllama 模型
from langchain_core.language_models.chat_models import BaseChatModel  # 导入基础聊天模型类

from .llm_config import get_llm_config, LLMProviderConfig
from utils.logger import LOG  # 导入日志工具


def _create_ollama_llm(config: LLMProviderConfig) -> BaseChatModel:
    """
    创建 Ollama LLM 实例

    Args:
        config: Ollama 提供者配置

    Returns:
        ChatOllama 实例

    Raises:
        ValueError: 如果 Ollama 服务不可用或模型未安装
    """
    try:
        LOG.info(
            f"[LLM Factory] 创建 Ollama 实例: "
            f"model={config.model}, base_url={config.base_url}"
        )

        return ChatOllama(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            num_predict=config.extra_params.get("num_predict", config.max_tokens),
        )
    except Exception as e:
        error_msg = (
            f"无法初始化 Ollama 模型 '{config.model}' at {config.base_url}。\n"
            f"请确保：\n"
            f"1. Ollama 服务正在运行（检查: ollama serve）\n"
            f"2. 模型已安装（检查: ollama list）\n"
            f"3. 如需安装模型，运行: ollama pull {config.model}\n"
            f"4. 设置 OLLAMA_MODEL 环境变量以使用其他模型\n"
            f"错误详情: {str(e)}"
        )
        LOG.error(f"[LLM Factory] {error_msg}")
        raise ValueError(error_msg) from e


def _create_deepseek_llm(config: LLMProviderConfig) -> BaseChatModel:
    """
    创建 DeepSeek LLM 实例

    Args:
        config: DeepSeek 提供者配置

    Returns:
        ChatOpenAI 实例（使用 DeepSeek API）
    """
    LOG.info(
        f"[LLM Factory] 创建 DeepSeek 实例: "
        f"model={config.model}, base_url={config.base_url}"
    )

    return ChatOpenAI(
        model=config.model,
        openai_api_base=config.base_url,
        openai_api_key=config.api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


def _create_openai_llm(config: LLMProviderConfig) -> BaseChatModel:
    """
    创建 OpenAI LLM 实例

    Args:
        config: OpenAI 提供者配置

    Returns:
        ChatOpenAI 实例
    """
    LOG.info(
        f"[LLM Factory] 创建 OpenAI 实例: model={config.model}"
    )

    params = {
        "model": config.model,
        "openai_api_key": config.api_key,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }

    # 如果指定了自定义 base_url
    if config.base_url:
        params["openai_api_base"] = config.base_url
        LOG.info(f"[LLM Factory] 使用自定义 OpenAI base_url: {config.base_url}")

    return ChatOpenAI(**params)


def create_llm(provider_name: str = None) -> BaseChatModel:
    """
    创建 LLM 实例，支持多种 LLM 提供者。

    优先级（可通过 LLM_PROVIDER_PRIORITY 环境变量配置）:
    - 默认: Ollama > DeepSeek > OpenAI
    - 可配置为: deepseek,openai,ollama 或其他顺序

    环境变量说明:
        # 提供者优先级
        LLM_PROVIDER_PRIORITY: 逗号分隔的提供者列表（如：ollama,deepseek,openai）

        # Ollama 配置（默认提供者）
        OLLAMA_MODEL: Ollama 模型名称（默认: unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL）
        OLLAMA_BASE_URL: Ollama 服务地址（默认: http://localhost:11434）
        OLLAMA_TEMPERATURE: 温度参数（默认: 0.8）
        OLLAMA_MAX_TOKENS: 最大 token 数（默认: 8192）
        OLLAMA_ENABLED: 是否启用（默认: true）

        # DeepSeek 配置
        DEEPSEEK_API_KEY: DeepSeek API 密钥（必需）
        DEEPSEEK_MODEL: 模型名称（默认: deepseek-chat）
        DEEPSEEK_BASE_URL: API 地址（默认: https://api.deepseek.com）
        DEEPSEEK_TEMPERATURE: 温度参数（默认: 0.8）
        DEEPSEEK_MAX_TOKENS: 最大 token 数（默认: 8192）
        DEEPSEEK_ENABLED: 是否启用（默认: true）

        # OpenAI 配置
        OPENAI_API_KEY: OpenAI API 密钥（必需）
        OPENAI_MODEL: 模型名称（默认: gpt-4o-mini）
        OPENAI_BASE_URL: API 地址（可选，用于兼容的 API）
        OPENAI_TEMPERATURE: 温度参数（默认: 0.8）
        OPENAI_MAX_TOKENS: 最大 token 数（默认: 8192）
        OPENAI_ENABLED: 是否启用（默认: true）

    Args:
        provider_name: 指定要使用的提供者名称（可选）
                      如果指定，则直接使用该提供者
                      如果未指定，按优先级自动选择

    Returns:
        BaseChatModel: LLM 实例

    Raises:
        ValueError: 如果未配置任何 LLM 提供者或指定的提供者不可用

    Examples:
        # 自动选择（按优先级）
        llm = create_llm()

        # 指定使用 Ollama
        llm = create_llm("ollama")

        # 指定使用 DeepSeek
        llm = create_llm("deepseek")
    """
    # 获取配置
    llm_config = get_llm_config()

    # 如果指定了提供者名称
    if provider_name:
        config = llm_config.get_provider_config(provider_name)
        if not config:
            available = llm_config.list_available_providers()
            raise ValueError(
                f"指定的提供者 '{provider_name}' 不可用。\n"
                f"可用的提供者: {', '.join(available) if available else '无'}\n"
                f"请检查环境变量配置。"
            )
        LOG.info(f"[LLM Factory] 使用指定的提供者: {provider_name}")
    else:
        # 按优先级自动选择
        config = llm_config.get_first_available_provider()
        if not config:
            raise ValueError(
                "未找到可用的 LLM 提供者。\n"
                "请至少配置以下之一：\n"
                "1. Ollama: 确保 Ollama 服务运行在 http://localhost:11434\n"
                "2. DeepSeek: 设置 DEEPSEEK_API_KEY 环境变量\n"
                "3. OpenAI: 设置 OPENAI_API_KEY 环境变量\n"
                "\n"
                "提示: Ollama 是默认提供者，无需 API key。\n"
                "运行 'ollama serve' 启动服务，然后 'ollama pull unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL' 下载模型。"
            )
        LOG.info(
            f"[LLM Factory] 自动选择提供者: {config.name} "
            f"(优先级: {' > '.join(llm_config.priority)})"
        )

    # 根据提供者类型创建 LLM 实例
    if config.name == "ollama":
        return _create_ollama_llm(config)
    elif config.name == "deepseek":
        return _create_deepseek_llm(config)
    elif config.name == "openai":
        return _create_openai_llm(config)
    else:
        raise ValueError(f"不支持的提供者类型: {config.name}")


def list_available_providers() -> list[str]:
    """
    列出所有可用的 LLM 提供者

    Returns:
        可用提供者名称列表
    """
    llm_config = get_llm_config()
    return llm_config.list_available_providers()


def get_current_provider_info() -> dict:
    """
    获取当前使用的提供者信息

    Returns:
        包含提供者信息的字典
    """
    llm_config = get_llm_config()
    config = llm_config.get_first_available_provider()

    if not config:
        return {
            "provider": None,
            "model": None,
            "available": False,
        }

    return {
        "provider": config.name,
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "available": True,
    }
