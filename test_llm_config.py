#!/usr/bin/env python3
"""
测试 LLM 配置和工厂功能
验证所有提供者的配置和创建流程
"""

import os
import sys

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.llm_config import get_llm_config, reload_llm_config
from src.agents.llm_factory import (
    create_llm,
    list_available_providers,
    get_current_provider_info
)
from src.utils.logger import LOG


def print_section(title: str):
    """打印分段标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_llm_config():
    """测试 LLM 配置加载"""
    print_section("测试 1: LLM 配置加载")

    config = get_llm_config()

    print(f"\n📋 配置优先级: {' > '.join(config.priority)}")
    print(f"📋 可用提供者: {', '.join(config.list_available_providers())}")

    print("\n📝 详细配置:")
    for provider_name in config.priority:
        provider_config = config.get_provider_config(provider_name)
        if provider_config:
            print(f"\n  ✅ {provider_name.upper()}:")
            print(f"     模型: {provider_config.model}")
            print(f"     地址: {provider_config.base_url}")
            print(f"     温度: {provider_config.temperature}")
            print(f"     Max Tokens: {provider_config.max_tokens}")
            if provider_config.api_key:
                masked_key = provider_config.api_key[:8] + "..." + provider_config.api_key[-4:]
                print(f"     API Key: {masked_key}")
        else:
            print(f"\n  ⚠️  {provider_name.upper()}: 未配置或已禁用")


def test_list_available_providers():
    """测试列出可用提供者"""
    print_section("测试 2: 列出可用提供者")

    providers = list_available_providers()
    print(f"\n可用提供者数量: {len(providers)}")

    if providers:
        print("\n提供者列表:")
        for i, provider in enumerate(providers, 1):
            print(f"  {i}. {provider}")
    else:
        print("\n⚠️  没有可用的提供者")
        print("请至少配置以下之一:")
        print("  - Ollama (默认，无需 API key)")
        print("  - DeepSeek (设置 DEEPSEEK_API_KEY)")
        print("  - OpenAI (设置 OPENAI_API_KEY)")


def test_get_current_provider_info():
    """测试获取当前提供者信息"""
    print_section("测试 3: 获取当前提供者信息")

    info = get_current_provider_info()

    if info["available"]:
        print(f"\n✅ 当前提供者: {info['provider'].upper()}")
        print(f"   模型: {info['model']}")
        print(f"   地址: {info['base_url']}")
        print(f"   温度: {info['temperature']}")
        print(f"   Max Tokens: {info['max_tokens']}")
    else:
        print("\n❌ 没有可用的提供者")


def test_create_llm_auto():
    """测试自动创建 LLM（按优先级）"""
    print_section("测试 4: 自动创建 LLM（按优先级）")

    try:
        llm = create_llm()
        print(f"\n✅ 成功创建 LLM 实例")
        print(f"   类型: {type(llm).__name__}")
        print(f"   模型: {getattr(llm, 'model', 'N/A')}")

        # 测试简单调用
        print("\n🧪 测试简单对话...")
        response = llm.invoke("Say 'Hello' in one word")
        print(f"   回复: {response.content[:100]}...")

    except Exception as e:
        print(f"\n❌ 创建 LLM 失败: {str(e)}")


def test_create_llm_specific():
    """测试创建指定提供者的 LLM"""
    print_section("测试 5: 创建指定提供者的 LLM")

    providers = list_available_providers()

    for provider_name in providers:
        print(f"\n  测试创建 {provider_name.upper()} LLM...")
        try:
            llm = create_llm(provider_name)
            print(f"  ✅ 成功创建 {provider_name} LLM")
            print(f"     类型: {type(llm).__name__}")
        except Exception as e:
            print(f"  ❌ 创建失败: {str(e)[:100]}")


def test_priority_override():
    """测试优先级覆盖"""
    print_section("测试 6: 优先级覆盖")

    print("\n📝 原始优先级:")
    original_priority = os.getenv("LLM_PROVIDER_PRIORITY", "未设置")
    print(f"   LLM_PROVIDER_PRIORITY={original_priority}")

    config = get_llm_config()
    print(f"   实际优先级: {' > '.join(config.priority)}")

    print("\n💡 提示:")
    print("   可以通过设置 LLM_PROVIDER_PRIORITY 环境变量来覆盖优先级")
    print("   例如: export LLM_PROVIDER_PRIORITY=deepseek,openai,ollama")


def test_provider_enable_disable():
    """测试提供者启用/禁用"""
    print_section("测试 7: 提供者启用/禁用")

    providers = ["ollama", "deepseek", "openai"]

    print("\n📝 各提供者启用状态:")
    for provider in providers:
        env_var = f"{provider.upper()}_ENABLED"
        enabled = os.getenv(env_var, "true").lower() in ("true", "1", "yes", "on")
        status = "✅ 启用" if enabled else "❌ 禁用"
        print(f"   {provider.upper()}: {status} ({env_var}={os.getenv(env_var, 'true')})")

    print("\n💡 提示:")
    print("   可以通过设置 {PROVIDER}_ENABLED=false 来禁用某个提供者")
    print("   例如: export OLLAMA_ENABLED=false")


def print_summary():
    """打印测试总结"""
    print_section("🎯 测试总结")

    config = get_llm_config()
    info = get_current_provider_info()

    print(f"\n✅ 配置加载: 成功")
    print(f"✅ 可用提供者: {len(config.list_available_providers())} 个")

    if info["available"]:
        print(f"✅ 当前提供者: {info['provider'].upper()} ({info['model']})")
        print(f"\n🎉 LLM 配置系统正常工作！")
    else:
        print(f"⚠️  当前提供者: 无")
        print(f"\n⚠️  请配置至少一个 LLM 提供者")


def print_env_config_help():
    """打印环境变量配置帮助"""
    print_section("💡 环境变量配置帮助")

    print("\n📋 核心配置:")
    print("  LLM_PROVIDER_PRIORITY    # 提供者优先级（逗号分隔）")
    print("")
    print("📋 Ollama 配置 (默认提供者):")
    print("  OLLAMA_MODEL             # 模型名称")
    print("  OLLAMA_BASE_URL          # 服务地址")
    print("  OLLAMA_TEMPERATURE       # 温度参数")
    print("  OLLAMA_MAX_TOKENS        # 最大 token 数")
    print("  OLLAMA_ENABLED           # 是否启用")
    print("")
    print("📋 DeepSeek 配置:")
    print("  DEEPSEEK_API_KEY         # API 密钥 (必需)")
    print("  DEEPSEEK_MODEL           # 模型名称")
    print("  DEEPSEEK_BASE_URL        # API 地址")
    print("  DEEPSEEK_TEMPERATURE     # 温度参数")
    print("  DEEPSEEK_MAX_TOKENS      # 最大 token 数")
    print("  DEEPSEEK_ENABLED         # 是否启用")
    print("")
    print("📋 OpenAI 配置:")
    print("  OPENAI_API_KEY           # API 密钥 (必需)")
    print("  OPENAI_MODEL             # 模型名称")
    print("  OPENAI_BASE_URL          # API 地址 (可选)")
    print("  OPENAI_TEMPERATURE       # 温度参数")
    print("  OPENAI_MAX_TOKENS        # 最大 token 数")
    print("  OPENAI_ENABLED           # 是否启用")

    print("\n📖 详细文档: 请参考 .env.example 文件")


def main():
    """主测试函数"""
    print("\n" + "🚀" * 35)
    print("  LangCoach - LLM 配置与工厂测试")
    print("  Phase 2.5: 增强的 LLM 配置系统")
    print("🚀" * 35)

    print("\n📝 环境信息:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  工作目录: {os.getcwd()}")

    # 运行所有测试
    try:
        test_llm_config()
        test_list_available_providers()
        test_get_current_provider_info()
        test_priority_override()
        test_provider_enable_disable()
        test_create_llm_auto()
        test_create_llm_specific()

        print_summary()
        print_env_config_help()

        print("\n" + "=" * 70)
        print("  ✅ 所有测试完成")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
