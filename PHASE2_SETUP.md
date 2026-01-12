# Phase 2 Setup Guide: Long-Term Memory with Milvus

## Overview

Phase 2 实现了长期记忆系统，使用 Milvus 向量数据库存储和检索对话历史，让 LangCoach 能够：

1. **记住过往对话**：存储对话摘要到向量数据库
2. **智能检索**：基于语义相似度检索相关历史对话
3. **个性化学习**：根据用户过往表现调整教学内容
4. **上下文窗口管理**：自动限制记忆注入量，避免超出 LLM 上下文限制

## Architecture

```
┌─────────────┐
│   Gradio    │  用户界面
│     UI      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  AgentBase  │  对话代理
│  + Memory   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ LongTerm    │────▶│   OpenAI     │  嵌入生成
│   Memory    │     │  Embeddings  │  (或 Ollama)
└──────┬──────┘     └──────────────┘
       │
       ▼
┌─────────────┐
│   Milvus    │  向量数据库
│  (etcd +    │
│   minio)    │
└─────────────┘
```

## Environment Variables

### Required for Memory System

```bash
# Milvus 配置
MILVUS_HOST=milvus              # Milvus 主机（Docker 内使用服务名）
MILVUS_PORT=19530               # Milvus 端口

# 嵌入模型配置（二选一）
# 选项 1: OpenAI Embeddings (推荐，更准确)
OPENAI_API_KEY=your_openai_key

# 选项 2: Ollama Embeddings (本地，隐私优先)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Optional Configuration

```bash
# Gradio 端口 (Phase 2 默认改为 8300)
GRADIO_PORT=8300

# LLM 配置 (已有)
DEEPSEEK_API_KEY=your_deepseek_key
```

## Quick Start

### 1. 使用 Docker Compose (推荐)

**生产环境：**

```bash
# 1. 设置环境变量
export DEEPSEEK_API_KEY=your_key
export OPENAI_API_KEY=your_key  # 用于 embeddings

# 2. 启动所有服务（包括 Milvus）
docker-compose up -d

# 3. 查看日志
docker-compose logs -f langcoach

# 4. 访问应用
# http://localhost:8300
```

**开发环境（代码热重载）：**

```bash
# 使用开发配置
docker-compose -f docker-compose.dev.yml up

# 修改代码会自动重载
```

### 2. 本地开发（不使用 Docker）

```bash
# 1. 启动 Milvus (使用 Docker)
docker-compose up -d etcd minio milvus

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置环境变量
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
export OPENAI_API_KEY=your_key

# 4. 运行应用
python src/main.py
```

## How It Works

### 1. Memory Storage

当对话结束时，调用 `save_conversation_summary()` 保存摘要：

```python
# 在 UI 或 agent 中调用
agent.save_conversation_summary(
    summary="User practiced hotel check-in. Struggled with past tense.",
    session_id="session_123",
    metadata={
        "vocabulary_mistakes": ["reservation", "amenities"],
        "grammar_issues": ["past_tense"]
    }
)
```

### 2. Memory Retrieval

在每次对话时，系统自动检索相关历史：

```python
# AgentBase 自动执行
# 1. 生成用户输入的嵌入向量
# 2. 在 Milvus 中搜索相似的历史对话
# 3. 将相关记忆注入到 prompt
# 4. LLM 基于历史记忆生成更个性化的回复
```

### 3. Context Window Management

系统自动管理上下文窗口：

- **默认限制**：3000 tokens (约 12,000 字符)
- **动态裁剪**：按相似度排序，优先注入最相关的记忆
- **字符估算**：平均 4 字符/token

配置项在 `long_term_memory.py`:

```python
MAX_CONTEXT_TOKENS = 3000  # 可根据模型调整
AVG_CHARS_PER_TOKEN = 4
```

## Milvus Collection Schema

```python
Collection: langcoach_memory

Fields:
- id (INT64, Primary Key, Auto ID)
- user_id (VARCHAR[100])        # 用户标识
- session_id (VARCHAR[100])     # 会话 ID
- scenario (VARCHAR[50])        # 场景类型
- summary (VARCHAR[2000])       # 对话摘要
- metadata (VARCHAR[1000])      # JSON 格式的元数据
- timestamp (INT64)             # Unix 时间戳
- embedding (FLOAT_VECTOR[1536]) # 嵌入向量

Index: IVF_FLAT (L2 distance)
```

## API Reference

### AgentBase Methods

#### `save_conversation_summary(summary, session_id, metadata)`

保存对话摘要到长期记忆。

**参数：**
- `summary` (str): 对话摘要文本
- `session_id` (str): 会话 ID
- `metadata` (dict): 额外的元数据

**返回：**
- `bool`: 是否保存成功

**示例：**

```python
agent.save_conversation_summary(
    summary="Practiced job interview. Confident with technical questions.",
    metadata={"score": 8.5, "focus_area": "behavioral_questions"}
)
```

### LongTermMemory Methods

#### `retrieve_relevant_memories(user_id, query, scenario, top_k, check_context_limit)`

检索相关的历史记忆。

**参数：**
- `user_id` (str): 用户 ID
- `query` (str): 查询文本
- `scenario` (str, optional): 场景过滤
- `top_k` (int): 返回数量（默认 3）
- `check_context_limit` (bool): 是否检查上下文限制（默认 True）

**返回：**
- `List[Dict]`: 记忆列表

#### `get_user_statistics(user_id)`

获取用户学习统计。

**返回：**

```python
{
    "total_sessions": 15,
    "scenario_counts": {
        "job_interview": 6,
        "hotel_checkin": 5,
        "renting": 4
    },
    "latest_time": "2024-03-15 14:30"
}
```

#### `delete_user_memories(user_id)`

删除用户所有记忆（隐私保护）。

## Monitoring & Debugging

### 查看 Milvus 状态

```bash
# 检查服务健康状态
docker-compose ps

# 查看 Milvus 日志
docker-compose logs milvus

# 访问 MinIO 控制台
# http://localhost:9001
# Username: minioadmin
# Password: minioadmin
```

### 检查 Collection 信息

```python
from pymilvus import connections, utility, Collection

# 连接到 Milvus
connections.connect(host="localhost", port="19530")

# 检查 collection
if utility.has_collection("langcoach_memory"):
    collection = Collection("langcoach_memory")
    print(f"Entity count: {collection.num_entities}")
    print(f"Schema: {collection.schema}")
```

### 日志级别

在 `src/utils/logger.py` 中调整：

```python
# 查看详细的记忆检索日志
LOG.debug("[LongTermMemory] ...")  # 开启 DEBUG 级别
```

## Troubleshooting

### 问题 1: Milvus 连接失败

**错误：** `[LongTermMemory] 连接 Milvus 失败`

**解决方案：**
```bash
# 检查 Milvus 是否运行
docker-compose ps milvus

# 检查端口是否正确
echo $MILVUS_HOST  # 应为 "milvus" (Docker) 或 "localhost" (本地)
echo $MILVUS_PORT  # 应为 "19530"

# 重启 Milvus
docker-compose restart milvus
```

### 问题 2: 嵌入生成失败

**错误：** `[LongTermMemory] 存储对话摘要失败`

**解决方案：**
```bash
# 检查 API key
echo $OPENAI_API_KEY

# 或切换到 Ollama
export OLLAMA_BASE_URL=http://localhost:11434
# 确保 Ollama 服务运行并且模型已下载
```

### 问题 3: 记忆未注入到对话

**检查步骤：**

1. 确认 `MILVUS_HOST` 环境变量已设置
2. 查看日志：`[LongTermMemory] 检索到 X 条相关记忆`
3. 检查是否有历史对话存储
4. 确认 `enable_long_term_memory=True`

### 问题 4: 磁盘空间不足

**现象：** Milvus/etcd/minio 容器异常退出

**解决方案：**
```bash
# 查看数据卷大小
du -sh volumes/

# 清理旧数据
docker-compose down -v  # 警告：会删除所有数据
docker-compose up -d
```

## Performance Tuning

### 优化检索性能

在 `long_term_memory.py` 中调整索引参数：

```python
# 对于大规模数据（>100k 记录）
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_SQ8",  # 量化版本，更快
    "params": {"nlist": 1024}  # 增加分区数
}
```

### 减少内存占用

```python
# 使用更小的嵌入维度（切换到 Ollama 小模型）
OLLAMA_MODEL=nomic-embed-text  # 768 维，更轻量
```

### 加速 Docker 启动

```bash
# 预拉取镜像
docker-compose pull

# 使用更快的镜像源（中国大陆）
# 在 docker-compose.yml 中添加镜像加速配置
```

## Upgrade from Phase 1

如果从 Phase 1 升级，需要：

1. **更新依赖：**
   ```bash
   pip install -r requirements.txt
   ```

2. **启动 Milvus：**
   ```bash
   docker-compose up -d etcd minio milvus
   ```

3. **设置环境变量：**
   ```bash
   export MILVUS_HOST=milvus
   export MILVUS_PORT=19530
   ```

4. **代码兼容性：**
   - 所有现有代码向后兼容
   - 长期记忆功能默认禁用（除非设置 `MILVUS_HOST`）
   - 可以逐步迁移，无需一次性修改所有代码

## Next Steps (Phase 3)

Phase 2 为未来功能奠定基础：

- **Speech Integration**: 存储语音识别错误，改进 STT
- **Vocabulary Tracking**: 自动记录用户掌握的词汇
- **Adaptive Difficulty**: 根据历史表现动态调整难度
- **Multi-user Support**: 完整的用户系统和身份验证

## Resources

- [Milvus Documentation](https://milvus.io/docs)
- [LangChain Memory Guide](https://python.langchain.com/docs/modules/memory/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)

## Support

如有问题，请查看：

1. 项目日志：`logs/app.log`
2. Docker 日志：`docker-compose logs`
3. GitHub Issues: 提交问题报告

---

**Version:** 1.2.0 (Phase 2 - Smart Memory)
**Last Updated:** 2024-03-15
