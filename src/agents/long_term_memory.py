"""
长期记忆模块 - 使用 Milvus 向量数据库存储和检索对话记忆
实现 Phase 2 要求：
1. 创建对话摘要的嵌入
2. 存储到 Milvus 向量数据库
3. 检索相关历史记忆
4. 上下文窗口限制检查
"""

import os
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from utils.logger import LOG


class LongTermMemory:
    """长期记忆管理器 - 负责对话记忆的存储和检索"""

    # Milvus collection schema
    COLLECTION_NAME = "langcoach_memory"
    DIMENSION = 1536  # OpenAI text-embedding-ada-002 dimension

    # 上下文窗口限制（以 token 数计算）
    MAX_CONTEXT_TOKENS = 3000  # 为 LLM 主对话留出空间
    AVG_CHARS_PER_TOKEN = 4  # 平均每个 token 的字符数（估算）

    def __init__(
        self,
        host: str = None,
        port: str = None,
        use_openai: bool = True,
        openai_api_key: str = None,
        ollama_base_url: str = None,
        ollama_model: str = "llama3.1:8b",
    ):
        """
        初始化长期记忆管理器

        Args:
            host: Milvus 主机地址
            port: Milvus 端口
            use_openai: 是否使用 OpenAI embeddings（否则使用 Ollama）
            openai_api_key: OpenAI API key
            ollama_base_url: Ollama 服务地址
            ollama_model: Ollama 模型名称
        """
        # Milvus 连接配置
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")

        # 初始化 embeddings
        if use_openai and (openai_api_key or os.getenv("OPENAI_API_KEY")):
            LOG.info("[LongTermMemory] 使用 OpenAI Embeddings")
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002"
            )
            self.dimension = 1536
        else:
            LOG.info("[LongTermMemory] 使用 Ollama Embeddings")
            self.embeddings = OllamaEmbeddings(
                base_url=ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            )
            # Ollama embeddings 维度取决于模型，这里使用常见的 4096
            self.dimension = 4096

        self.collection: Optional[Collection] = None
        self._connected = False
        self._connect()

    def _connect(self):
        """连接到 Milvus 并初始化 collection"""
        try:
            # 连接到 Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=5  # 5秒超时，避免长时间等待
            )
            LOG.info(f"[LongTermMemory] 成功连接到 Milvus: {self.host}:{self.port}")

            # 初始化 collection
            self._init_collection()
            self._connected = True

        except Exception as e:
            LOG.warning(f"[LongTermMemory] Milvus 不可用，长期记忆功能已禁用: {e}")
            self._connected = False
            # 不再抛出异常，允许应用继续运行

    @property
    def is_connected(self) -> bool:
        """检查是否已连接到 Milvus"""
        return self._connected

    def _init_collection(self):
        """初始化或加载 Milvus collection"""
        try:
            # 检查 collection 是否存在
            if utility.has_collection(self.COLLECTION_NAME):
                LOG.info(f"[LongTermMemory] 加载现有 collection: {self.COLLECTION_NAME}")
                self.collection = Collection(self.COLLECTION_NAME)
            else:
                LOG.info(f"[LongTermMemory] 创建新 collection: {self.COLLECTION_NAME}")
                # 定义 schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="scenario", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="timestamp", dtype=DataType.INT64),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                ]
                schema = CollectionSchema(fields=fields, description="LangCoach conversation memory")

                # 创建 collection
                self.collection = Collection(
                    name=self.COLLECTION_NAME,
                    schema=schema,
                    using='default',
                )

                # 创建索引以提高搜索效率
                index_params = {
                    "metric_type": "L2",  # 欧氏距离
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                LOG.info("[LongTermMemory] Collection 和索引创建成功")

            # 加载 collection 到内存
            self.collection.load()
            LOG.info("[LongTermMemory] Collection 已加载到内存")

        except Exception as e:
            LOG.error(f"[LongTermMemory] 初始化 collection 失败: {e}")
            raise

    def store_conversation_summary(
        self,
        user_id: str,
        session_id: str,
        scenario: str,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        存储对话摘要到 Milvus

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            scenario: 场景名称
            summary: 对话摘要
            metadata: 额外的元数据（如难度级别、轮次等）

        Returns:
            bool: 是否存储成功
        """
        if not self._connected:
            return False

        try:
            LOG.debug(f"[LongTermMemory] 存储对话摘要: user={user_id}, session={session_id}, scenario={scenario}")

            # 生成嵌入向量
            embedding = self.embeddings.embed_query(summary)

            # 准备数据
            timestamp = int(datetime.now().timestamp())
            metadata_str = json.dumps(metadata or {}, ensure_ascii=False)

            # 插入数据
            entities = [
                [user_id],  # user_id
                [session_id],  # session_id
                [scenario],  # scenario
                [summary],  # summary
                [metadata_str],  # metadata
                [timestamp],  # timestamp
                [embedding],  # embedding
            ]

            insert_result = self.collection.insert(entities)
            self.collection.flush()

            LOG.info(f"[LongTermMemory] 成功存储对话摘要: {len(insert_result.primary_keys)} 条记录")
            return True

        except Exception as e:
            LOG.error(f"[LongTermMemory] 存储对话摘要失败: {e}")
            return False

    def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        scenario: Optional[str] = None,
        top_k: int = 3,
        check_context_limit: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        检索相关的历史记忆

        Args:
            user_id: 用户 ID
            query: 查询文本（通常是当前对话的摘要或关键词）
            scenario: 可选的场景过滤
            top_k: 返回最相关的 k 条记忆
            check_context_limit: 是否检查上下文窗口限制

        Returns:
            List[Dict]: 相关记忆列表，每条记忆包含 summary、metadata、timestamp 等
        """
        if not self._connected:
            return []

        try:
            LOG.debug(f"[LongTermMemory] 检索记忆: user={user_id}, query={query[:50]}...")

            # 生成查询嵌入
            query_embedding = self.embeddings.embed_query(query)

            # 构建搜索表达式
            expr = f'user_id == "{user_id}"'
            if scenario:
                expr += f' && scenario == "{scenario}"'

            # 执行相似度搜索
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["session_id", "scenario", "summary", "metadata", "timestamp"],
            )

            # 处理搜索结果
            memories = []
            total_chars = 0

            for hits in results:
                for hit in hits:
                    memory = {
                        "session_id": hit.entity.get("session_id"),
                        "scenario": hit.entity.get("scenario"),
                        "summary": hit.entity.get("summary"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}")),
                        "timestamp": hit.entity.get("timestamp"),
                        "distance": hit.distance,  # 距离越小越相似
                    }

                    # 检查上下文窗口限制
                    if check_context_limit:
                        summary_chars = len(memory["summary"])
                        estimated_tokens = summary_chars // self.AVG_CHARS_PER_TOKEN

                        if total_chars + summary_chars > self.MAX_CONTEXT_TOKENS * self.AVG_CHARS_PER_TOKEN:
                            LOG.warning(
                                f"[LongTermMemory] 达到上下文窗口限制，返回 {len(memories)} 条记忆"
                            )
                            break

                        total_chars += summary_chars

                    memories.append(memory)

            LOG.info(f"[LongTermMemory] 检索到 {len(memories)} 条相关记忆")
            return memories

        except Exception as e:
            LOG.error(f"[LongTermMemory] 检索记忆失败: {e}")
            return []

    def format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """
        将记忆格式化为适合注入到 prompt 的文本

        Args:
            memories: 记忆列表

        Returns:
            str: 格式化的记忆文本
        """
        if not memories:
            return ""

        formatted = "### Relevant Past Conversations:\n\n"

        for i, memory in enumerate(memories, 1):
            timestamp = datetime.fromtimestamp(memory["timestamp"]).strftime("%Y-%m-%d %H:%M")
            formatted += f"{i}. **{memory['scenario']}** ({timestamp})\n"
            formatted += f"   {memory['summary']}\n\n"

        return formatted

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户的学习统计信息

        Args:
            user_id: 用户 ID

        Returns:
            Dict: 统计信息
        """
        if not self._connected:
            return {
                "total_sessions": 0,
                "scenario_counts": {},
                "latest_time": None,
            }

        try:
            expr = f'user_id == "{user_id}"'

            # 获取总记录数
            results = self.collection.query(
                expr=expr,
                output_fields=["scenario", "timestamp"],
            )

            total_sessions = len(results)

            # 按场景统计
            scenario_counts = {}
            for result in results:
                scenario = result.get("scenario", "unknown")
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

            # 最近学习时间
            if results:
                latest_timestamp = max(r.get("timestamp", 0) for r in results)
                latest_time = datetime.fromtimestamp(latest_timestamp).strftime("%Y-%m-%d %H:%M")
            else:
                latest_time = None

            stats = {
                "total_sessions": total_sessions,
                "scenario_counts": scenario_counts,
                "latest_time": latest_time,
            }

            LOG.info(f"[LongTermMemory] 用户统计: {stats}")
            return stats

        except Exception as e:
            LOG.error(f"[LongTermMemory] 获取用户统计失败: {e}")
            return {
                "total_sessions": 0,
                "scenario_counts": {},
                "latest_time": None,
            }

    def delete_user_memories(self, user_id: str) -> bool:
        """
        删除用户的所有记忆（用于隐私保护）

        Args:
            user_id: 用户 ID

        Returns:
            bool: 是否删除成功
        """
        if not self._connected:
            return False

        try:
            expr = f'user_id == "{user_id}"'
            self.collection.delete(expr)
            self.collection.flush()

            LOG.info(f"[LongTermMemory] 成功删除用户记忆: {user_id}")
            return True

        except Exception as e:
            LOG.error(f"[LongTermMemory] 删除用户记忆失败: {e}")
            return False

    def close(self):
        """关闭 Milvus 连接"""
        if not self._connected:
            return

        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
            self._connected = False
            LOG.info("[LongTermMemory] Milvus 连接已关闭")
        except Exception as e:
            LOG.error(f"[LongTermMemory] 关闭连接失败: {e}")


# 全局单例
_memory_instance: Optional[LongTermMemory] = None


def get_memory_instance() -> LongTermMemory:
    """获取长期记忆管理器的全局单例"""
    global _memory_instance

    if _memory_instance is None:
        _memory_instance = LongTermMemory()

    return _memory_instance
