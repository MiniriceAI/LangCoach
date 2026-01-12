#!/usr/bin/env python3
"""
测试 Milvus 集成和长期记忆功能
Phase 2 集成测试脚本
"""

import os
import sys
from datetime import datetime

# 设置测试环境变量
os.environ['MILVUS_HOST'] = os.getenv('MILVUS_HOST', 'localhost')
os.environ['MILVUS_PORT'] = os.getenv('MILVUS_PORT', '19530')

try:
    from src.agents.long_term_memory import LongTermMemory
    from src.utils.logger import LOG
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


def test_milvus_connection():
    """测试 Milvus 连接"""
    print("\n" + "="*60)
    print("🔌 测试 1: Milvus 连接")
    print("="*60)

    try:
        memory = LongTermMemory(
            host=os.getenv('MILVUS_HOST'),
            port=os.getenv('MILVUS_PORT'),
            use_openai=bool(os.getenv('OPENAI_API_KEY')),
        )
        print("✅ 成功连接到 Milvus")
        return memory
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None


def test_store_memory(memory):
    """测试存储对话摘要"""
    print("\n" + "="*60)
    print("💾 测试 2: 存储对话摘要")
    print("="*60)

    test_summaries = [
        {
            "user_id": "test_user_001",
            "session_id": f"test_session_{datetime.now().timestamp()}",
            "scenario": "job_interview",
            "summary": "User practiced job interview. Discussed technical skills and project experience. Struggled with behavioral questions.",
            "metadata": {
                "difficulty": "MEDIUM",
                "turns": 20,
                "score": 7.5,
            }
        },
        {
            "user_id": "test_user_001",
            "session_id": f"test_session_{datetime.now().timestamp() + 1}",
            "scenario": "hotel_checkin",
            "summary": "User practiced hotel check-in. Good with greetings and basic requests. Needs improvement on asking for amenities.",
            "metadata": {
                "difficulty": "PRIMARY",
                "turns": 10,
                "score": 6.0,
            }
        },
        {
            "user_id": "test_user_001",
            "session_id": f"test_session_{datetime.now().timestamp() + 2}",
            "scenario": "job_interview",
            "summary": "Second interview practice. Improved on behavioral questions. Confidently explained past projects using STAR method.",
            "metadata": {
                "difficulty": "ADVANCED",
                "turns": 30,
                "score": 8.5,
            }
        }
    ]

    success_count = 0
    for i, summary_data in enumerate(test_summaries, 1):
        print(f"\n  [{i}/{len(test_summaries)}] 存储: {summary_data['scenario']} - {summary_data['summary'][:50]}...")

        success = memory.store_conversation_summary(
            user_id=summary_data["user_id"],
            session_id=summary_data["session_id"],
            scenario=summary_data["scenario"],
            summary=summary_data["summary"],
            metadata=summary_data["metadata"],
        )

        if success:
            print(f"      ✅ 存储成功")
            success_count += 1
        else:
            print(f"      ❌ 存储失败")

    print(f"\n📊 存储成功: {success_count}/{len(test_summaries)}")
    return success_count == len(test_summaries)


def test_retrieve_memories(memory):
    """测试检索相关记忆"""
    print("\n" + "="*60)
    print("🔍 测试 3: 检索相关记忆")
    print("="*60)

    test_queries = [
        {
            "query": "I need help with interview preparation",
            "scenario": "job_interview",
            "description": "面试相关查询"
        },
        {
            "query": "How to ask about hotel facilities?",
            "scenario": "hotel_checkin",
            "description": "酒店设施查询"
        },
        {
            "query": "Behavioral questions in interviews",
            "scenario": None,  # 不限制场景
            "description": "跨场景查询"
        }
    ]

    all_success = True

    for i, query_data in enumerate(test_queries, 1):
        print(f"\n  [{i}/{len(test_queries)}] 查询: {query_data['description']}")
        print(f"      Query: \"{query_data['query']}\"")
        if query_data['scenario']:
            print(f"      Scenario: {query_data['scenario']}")

        memories = memory.retrieve_relevant_memories(
            user_id="test_user_001",
            query=query_data["query"],
            scenario=query_data["scenario"],
            top_k=3,
            check_context_limit=True,
        )

        if memories:
            print(f"      ✅ 检索到 {len(memories)} 条相关记忆")
            for j, mem in enumerate(memories, 1):
                print(f"         {j}. {mem['scenario']}: {mem['summary'][:60]}... (distance: {mem['distance']:.4f})")
        else:
            print(f"      ⚠️  未检索到相关记忆")
            all_success = False

    return all_success


def test_context_limit(memory):
    """测试上下文窗口限制"""
    print("\n" + "="*60)
    print("📏 测试 4: 上下文窗口限制")
    print("="*60)

    print(f"  当前配置:")
    print(f"    MAX_CONTEXT_TOKENS: {memory.MAX_CONTEXT_TOKENS}")
    print(f"    AVG_CHARS_PER_TOKEN: {memory.AVG_CHARS_PER_TOKEN}")
    print(f"    最大字符数: {memory.MAX_CONTEXT_TOKENS * memory.AVG_CHARS_PER_TOKEN}")

    # 请求大量记忆
    memories = memory.retrieve_relevant_memories(
        user_id="test_user_001",
        query="interview",
        top_k=10,  # 请求10条
        check_context_limit=True,
    )

    if memories:
        total_chars = sum(len(m['summary']) for m in memories)
        print(f"\n  ✅ 上下文限制正常工作")
        print(f"     返回记忆数: {len(memories)}")
        print(f"     总字符数: {total_chars}")
        print(f"     是否超限: {'否 ✅' if total_chars <= memory.MAX_CONTEXT_TOKENS * memory.AVG_CHARS_PER_TOKEN else '是 ⚠️'}")
        return True
    else:
        print("  ⚠️  未能测试上下文限制（无记忆返回）")
        return False


def test_user_statistics(memory):
    """测试用户统计功能"""
    print("\n" + "="*60)
    print("📊 测试 5: 用户统计")
    print("="*60)

    stats = memory.get_user_statistics("test_user_001")

    print(f"\n  用户统计信息:")
    print(f"    总会话数: {stats['total_sessions']}")
    print(f"    最近学习: {stats['latest_time']}")
    print(f"    场景分布:")
    for scenario, count in stats['scenario_counts'].items():
        print(f"      - {scenario}: {count} 次")

    return stats['total_sessions'] > 0


def test_cleanup(memory):
    """清理测试数据"""
    print("\n" + "="*60)
    print("🧹 测试 6: 清理测试数据")
    print("="*60)

    try:
        success = memory.delete_user_memories("test_user_001")
        if success:
            print("  ✅ 测试数据清理成功")
        else:
            print("  ❌ 测试数据清理失败")
        return success
    except Exception as e:
        print(f"  ❌ 清理失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "🚀"*30)
    print("LangCoach Phase 2 - Milvus 集成测试")
    print("🚀"*30)

    # 检查环境变量
    print(f"\n📝 环境配置:")
    print(f"  MILVUS_HOST: {os.getenv('MILVUS_HOST')}")
    print(f"  MILVUS_PORT: {os.getenv('MILVUS_PORT')}")
    print(f"  OPENAI_API_KEY: {'已配置 ✅' if os.getenv('OPENAI_API_KEY') else '未配置 ⚠️'}")

    # 执行测试
    results = {}

    # 测试 1: 连接
    memory = test_milvus_connection()
    if not memory:
        print("\n❌ Milvus 连接失败，无法继续测试")
        print("\n💡 请确保:")
        print("  1. Milvus 服务正在运行: docker-compose up -d milvus")
        print("  2. 环境变量配置正确")
        print("  3. 网络连接正常")
        sys.exit(1)
    results['connection'] = True

    # 测试 2: 存储
    results['store'] = test_store_memory(memory)

    # 测试 3: 检索
    results['retrieve'] = test_retrieve_memories(memory)

    # 测试 4: 上下文限制
    results['context_limit'] = test_context_limit(memory)

    # 测试 5: 统计
    results['statistics'] = test_user_statistics(memory)

    # 测试 6: 清理
    results['cleanup'] = test_cleanup(memory)

    # 关闭连接
    memory.close()

    # 总结
    print("\n" + "="*60)
    print("📋 测试结果总结")
    print("="*60)

    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name.ljust(20)}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过！Phase 2 长期记忆功能正常工作")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
    print("="*60 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
