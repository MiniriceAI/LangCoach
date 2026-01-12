"""
LLM 配置模块
管理所有 LLM 提供者的配置，包括默认值、优先级等
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
    """LLM 配置管理器"""

    # 默认配置
    DEFAULT_PROVIDERS = {
        "ollama": {
            "model": "hf.co/unsloth/GLM-4-9B-0414-GGUF:Q4_K_M",
            "base_url": "http://localhost:11434",
            "temperature": 0.8,
            "max_tokens": 8192,
        },
        "deepseek": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "temperature": 0.8,
            "max_tokens": 8192,
        },
        "openai": {
            "model": "gpt-4o-mini",
            "base_url": None,  # 使用默认
            "temperature": 0.8,
            "max_tokens": 8192,
        }
    }

    # 默认优先级（数字越小优先级越高）
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
            # 从环境变量读取优先级，例如：LLM_PROVIDER_PRIORITY=ollama,deepseek,openai
            self.priority = [p.strip() for p in priority_str.split(",") if p.strip()]
            LOG.info(f"[LLM Config] 使用自定义优先级: {self.priority}")
        else:
            self.priority = self.DEFAULT_PRIORITY.copy()
            LOG.debug(f"[LLM Config] 使用默认优先级: {self.priority}")

        # 2. 加载每个提供者的配置
        for provider_name in self.priority:
            config = self._load_provider_config(provider_name)
            if config:
                self.providers[provider_name] = config

    def _load_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """加载单个提供者的配置"""
        provider_upper = provider_name.upper()

        # 获取默认配置
        default_config = self.DEFAULT_PROVIDERS.get(provider_name, {})

        # 检查是否启用该提供者
        enabled_env = os.getenv(f"{provider_upper}_ENABLED", "true").lower()
        enabled = enabled_env in ("true", "1", "yes", "on")

        if not enabled:
            LOG.debug(f"[LLM Config] 提供者 {provider_name} 已禁用")
            return None

        # 构建配置
        if provider_name == "ollama":
            return self._load_ollama_config(default_config)
        elif provider_name == "deepseek":
            return self._load_deepseek_config(default_config)
        elif provider_name == "openai":
            return self._load_openai_config(default_config)
        else:
            LOG.warning(f"[LLM Config] 未知的提供者: {provider_name}")
            return None

    def _load_ollama_config(self, default: Dict[str, Any]) -> Optional[LLMProviderConfig]:
        """加载 Ollama 配置"""
        # Ollama 不需要 API key，始终尝试使用
        base_url = os.getenv("OLLAMA_BASE_URL", default.get("base_url"))
        model = os.getenv("OLLAMA_MODEL", default.get("model"))
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", default.get("temperature", 0.8)))
        max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", default.get("max_tokens", 8192)))

        LOG.info(f"[LLM Config] Ollama 配置: model={model}, base_url={base_url}")

        return LLMProviderConfig(
            name="ollama",
            enabled=True,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params={
                "num_predict": max_tokens,  # Ollama 使用 num_predict 而不是 max_tokens
            }
        )

    def _load_deepseek_config(self, default: Dict[str, Any]) -> Optional[LLMProviderConfig]:
        """加载 DeepSeek 配置"""
        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            LOG.debug("[LLM Config] DeepSeek API key 未配置")
            return None

        base_url = os.getenv("DEEPSEEK_BASE_URL", default.get("base_url"))
        model = os.getenv("DEEPSEEK_MODEL", default.get("model"))
        temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", default.get("temperature", 0.8)))
        max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", default.get("max_tokens", 8192)))

        LOG.info(f"[LLM Config] DeepSeek 配置: model={model}, base_url={base_url}")

        return LLMProviderConfig(
            name="deepseek",
            enabled=True,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _load_openai_config(self, default: Dict[str, Any]) -> Optional[LLMProviderConfig]:
        """加载 OpenAI 配置"""
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            LOG.debug("[LLM Config] OpenAI API key 未配置")
            return None

        base_url = os.getenv("OPENAI_BASE_URL", default.get("base_url"))
        model = os.getenv("OPENAI_MODEL", default.get("model"))
        temperature = float(os.getenv("OPENAI_TEMPERATURE", default.get("temperature", 0.8)))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", default.get("max_tokens", 8192)))

        LOG.info(f"[LLM Config] OpenAI 配置: model={model}")

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
