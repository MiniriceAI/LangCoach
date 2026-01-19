"""
LLM 配置模块
管理所有 LLM 提供者的配置

所有配置都从 .env 文件读取，不使用硬编码默认值。
"""
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from utils.logger import LOG


@dataclass
class LLMProviderConfig:
    """单个 LLM 提供者的配置"""
    name: str
    enabled: bool = True
    model: str = ""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.8
    max_tokens: int = 8192
    extra_params: Dict[str, Any] = field(default_factory=dict)


class LLMConfig:
    """
    LLM 配置管理器

    所有配置从 .env 文件读取，支持的环境变量：

    # 优先级配置
    LLM_PROVIDER_PRIORITY=ollama,deepseek,openai

    # Ollama 配置
    OLLAMA_MODEL=<model_name>
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_TEMPERATURE=0.8
    OLLAMA_MAX_TOKENS=8192
    OLLAMA_ENABLED=true

    # DeepSeek 配置
    DEEPSEEK_API_KEY=<api_key>
    DEEPSEEK_MODEL=deepseek-chat
    DEEPSEEK_BASE_URL=https://api.deepseek.com
    DEEPSEEK_TEMPERATURE=0.8
    DEEPSEEK_MAX_TOKENS=8192
    DEEPSEEK_ENABLED=true

    # OpenAI 配置
    OPENAI_API_KEY=<api_key>
    OPENAI_MODEL=gpt-4o-mini
    OPENAI_BASE_URL=<optional>
    OPENAI_TEMPERATURE=0.8
    OPENAI_MAX_TOKENS=8192
    OPENAI_ENABLED=true
    """

    # 默认优先级
    DEFAULT_PRIORITY = ["ollama", "deepseek", "openai"]

    def __init__(self):
        """初始化 LLM 配置"""
        self.providers: Dict[str, LLMProviderConfig] = {}
        self.priority: List[str] = []
        self._load_config()

    def _load_config(self):
        """从环境变量加载配置"""
        # 1. 加载优先级配置
        priority_str = os.getenv("LLM_PROVIDER_PRIORITY", "")
        if priority_str:
            self.priority = [p.strip() for p in priority_str.split(",") if p.strip()]
            LOG.info(f"[LLM Config] 优先级: {self.priority}")
        else:
            self.priority = self.DEFAULT_PRIORITY.copy()
            LOG.info(f"[LLM Config] 使用默认优先级: {self.priority}")

        # 2. 加载每个提供者的配置
        for provider_name in self.priority:
            config = self._load_provider_config(provider_name)
            if config:
                self.providers[provider_name] = config
                LOG.info(f"[LLM Config] 提供者 {provider_name} 已加载")
            else:
                LOG.info(f"[LLM Config] 提供者 {provider_name} 不可用")

        # 3. 显示最终选择
        available = self.list_available_providers()
        LOG.info(f"[LLM Config] 可用提供者: {available}")
        if available:
            LOG.info(f"[LLM Config] 默认提供者: {available[0]}")

    def _load_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """加载单个提供者的配置"""
        provider_upper = provider_name.upper()

        # 检查是否启用该提供者
        enabled_env = os.getenv(f"{provider_upper}_ENABLED", "true").lower()
        enabled = enabled_env in ("true", "1", "yes", "on")

        if not enabled:
            LOG.debug(f"[LLM Config] 提供者 {provider_name} 已禁用")
            return None

        # 构建配置
        if provider_name == "ollama":
            return self._load_ollama_config()
        elif provider_name == "deepseek":
            return self._load_deepseek_config()
        elif provider_name == "openai":
            return self._load_openai_config()
        else:
            LOG.warning(f"[LLM Config] 未知的提供者: {provider_name}")
            return None

    def _load_ollama_config(self) -> Optional[LLMProviderConfig]:
        """加载 Ollama 配置（从 .env）"""
        model = os.getenv("OLLAMA_MODEL")
        base_url = os.getenv("OLLAMA_BASE_URL")

        LOG.debug(f"[LLM Config] Ollama 环境变量: OLLAMA_MODEL={model}, OLLAMA_BASE_URL={base_url}")

        # Ollama 必须配置 model 和 base_url
        if not model or not base_url:
            LOG.info("[LLM Config] Ollama 未配置 (需要 OLLAMA_MODEL 和 OLLAMA_BASE_URL)")
            return None

        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.8"))
        max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "8192"))

        LOG.info(f"[LLM Config] Ollama: model={model}, base_url={base_url}")

        return LLMProviderConfig(
            name="ollama",
            enabled=True,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params={
                "num_predict": max_tokens,
            }
        )

    def _load_deepseek_config(self) -> Optional[LLMProviderConfig]:
        """加载 DeepSeek 配置（从 .env）"""
        api_key = os.getenv("DEEPSEEK_API_KEY")

        LOG.debug(f"[LLM Config] DeepSeek 环境变量: DEEPSEEK_API_KEY={'***' if api_key else 'None'}")

        # 检查 API key 是否有效（非空且非占位符）
        if not api_key or api_key.startswith("your_") or api_key == "sk-xxx":
            LOG.info("[LLM Config] DeepSeek API key 未配置或为占位符")
            return None

        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.8"))
        max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))

        LOG.info(f"[LLM Config] DeepSeek: model={model}, base_url={base_url}")

        return LLMProviderConfig(
            name="deepseek",
            enabled=True,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _load_openai_config(self) -> Optional[LLMProviderConfig]:
        """加载 OpenAI 配置（从 .env）"""
        api_key = os.getenv("OPENAI_API_KEY")

        # 检查 API key 是否有效（非空且非占位符）
        if not api_key or api_key.startswith("your_") or api_key.startswith("sk-xxx"):
            LOG.debug("[LLM Config] OpenAI API key 未配置或为占位符")
            return None

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL")  # 可选
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.8"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "8192"))

        LOG.info(f"[LLM Config] OpenAI: model={model}")

        return LLMProviderConfig(
            name="openai",
            enabled=True,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """获取指定提供者的配置"""
        return self.providers.get(provider_name)

    def get_first_available_provider(self) -> Optional[LLMProviderConfig]:
        """按优先级获取第一个可用的提供者配置"""
        for provider_name in self.priority:
            config = self.providers.get(provider_name)
            if config and config.enabled:
                return config
        return None

    def list_available_providers(self) -> List[str]:
        """列出所有可用的提供者"""
        return [name for name, config in self.providers.items() if config.enabled]


# 全局配置实例
_llm_config_instance: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """获取全局 LLM 配置实例（单例模式）"""
    global _llm_config_instance
    if _llm_config_instance is None:
        _llm_config_instance = LLMConfig()
    return _llm_config_instance


def reload_llm_config():
    """重新加载 LLM 配置（用于配置变更后）"""
    global _llm_config_instance
    _llm_config_instance = LLMConfig()
    LOG.info("[LLM Config] 配置已重新加载")
