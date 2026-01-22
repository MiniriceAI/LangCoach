# LangCoach 聊天延时优化分析报告

## 📊 当前延时流程分析

### 用户发送消息到收到回复的完整流程

```
用户发送消息
    ↓
[小程序] 上传音频 (STT) - 120s超时
    ↓
[服务端] 语音识别 (Whisper) - ~2-5s
    ↓
[小程序] 发送文本到 /api/chat/message - 60s超时
    ↓
[服务端] LLM推理 (GLM-4-9B) - ~3-8s ⚠️ 最大瓶颈
    ↓
[服务端] 解析LLM响应 - ~0.1s
    ↓
[服务端] TTS生成音频 (Edge-TTS) - ~1-3s
    ↓
[小程序] 接收响应 + 自动播放
    ↓
用户听到回复
```

**总延时估计: 6-19秒** (不含网络延迟)

---

## 🔴 关键性能瓶颈

### 1. **LLM推理延时** (最大瓶颈 - 占比 40-60%)
**位置**: `/api/chat/message` 第687行
```python
response = llm.invoke(full_prompt)  # 3-8秒
```

**问题分析**:
- GLM-4-9B 模型推理速度受限于硬件
- 完整prompt包含场景上下文 + 对话历史 + 系统提示
- 没有流式输出，必须等待完整响应

**优化空间**: ⭐⭐⭐⭐⭐ (最高优先级)

---

### 2. **TTS音频生成延时** (占比 15-30%)
**位置**: `/api/chat/message` 第729行
```python
audio_url = await generate_audio_url(reply, ...)  # 1-3秒
```

**问题分析**:
- Edge-TTS 需要网络请求到微软服务器
- 音频生成后还要保存到临时文件
- 是串行操作，必须等待完成才能返回

**优化空间**: ⭐⭐⭐⭐ (高优先级)

---

### 3. **STT语音识别延时** (占比 20-40%)
**位置**: `/api/transcribe` 端点
```python
result = service.transcribe(audio=audio_data, ...)  # 2-5秒
```

**问题分析**:
- Whisper-large-v3 模型较大，推理时间长
- 小程序上传音频前需要录音完成
- 没有流式识别

**优化空间**: ⭐⭐⭐ (中等优先级)

---

### 4. **网络往返延迟** (占比 5-15%)
**位置**: 小程序 ↔ 服务端通信

**问题分析**:
- 至少需要 2 次往返 (STT + LLM)
- 如果启用TTS，需要额外的音频下载

**优化空间**: ⭐⭐ (低优先级)

---

## 💡 优化方案

### 方案 A: 流式输出 (推荐 - 可减少 30-40% 延时)

**实现思路**:
```python
# 改造 /api/chat/message 为流式响应
@app.post("/api/chat/message")
async def chat_message(request: ChatMessageRequest):
    # 1. 立即返回 LLM 流式输出
    # 2. 小程序边接收边显示
    # 3. 同时启动 TTS 生成
    # 4. 音频准备好后推送给小程序
```

**优点**:
- 用户能更快看到文本回复
- 可以并行处理 TTS
- 改善用户体验

**缺点**:
- 需要改造小程序接收逻辑
- 需要 WebSocket 或 Server-Sent Events

**预期效果**: 延时从 6-19s → 4-12s

---

### 方案 B: 并行处理 (推荐 - 可减少 20-30% 延时)

**实现思路**:
```python
# 当前: 串行处理
reply = llm.invoke(prompt)           # 3-8s
audio_url = await generate_audio()   # 1-3s (等待LLM完成)
# 总计: 4-11s

# 优化: 并行处理
async def chat_message_optimized():
    # 1. 启动 LLM 推理
    llm_task = asyncio.create_task(llm.invoke_async(prompt))

    # 2. 等待 LLM 完成
    reply = await llm_task

    # 3. 立即启动 TTS (不等待)
    tts_task = asyncio.create_task(generate_audio_url(reply))

    # 4. 返回响应 (TTS 在后台继续)
    return ChatMessageResponse(
        reply=reply,
        audio_url=None,  # 先返回 None
        # ...
    )

    # 5. TTS 完成后通过 WebSocket 推送给小程序
    audio_url = await tts_task
    await notify_client(session_id, audio_url)
```

**优点**:
- 实现相对简单
- 无需改造小程序接收逻辑
- 立即返回文本回复

**缺点**:
- 需要后续推送音频 URL
- 需要 WebSocket 连接

**预期效果**: 延时从 6-19s → 5-14s

---

### 方案 C: 模型优化 (推荐 - 可减少 40-60% 延时)

**实现思路**:
```python
# 当前: GLM-4-9B (Q8_K_XL 量化)
# 优化选项:

# 1. 使用更小的模型
# GLM-4-9B → GLM-3-6B (更快 30-40%)
# 或 Qwen-7B (更快 20-30%)

# 2. 更激进的量化
# Q8_K_XL → Q4_K_M (更快 50-60%, 质量略降)

# 3. 使用 vLLM 替代 Ollama
# vLLM 支持连续批处理，吞吐量提升 2-3 倍
```

**优点**:
- 最直接的性能提升
- 无需改造架构

**缺点**:
- 可能影响回复质量
- 需要重新部署模型

**预期效果**: 延时从 6-19s → 3-8s

---

### 方案 D: 缓存优化 (推荐 - 可减少 10-20% 延时)

**实现思路**:
```python
# 1. 缓存场景 prompt
scenario_cache = {}
def get_scenario_context(scenario, difficulty):
    key = f"{scenario}_{difficulty}"
    if key not in scenario_cache:
        scenario_cache[key] = load_prompt(scenario, difficulty)
    return scenario_cache[key]

# 2. 缓存常见回复
# 使用 Redis 缓存相似问题的回复

# 3. 预加载 TTS 语音
# 对常见回复预生成音频
```

**优点**:
- 实现简单
- 无需改造架构

**缺点**:
- 效果有限
- 只对重复问题有效

**预期效果**: 延时减少 10-20%

---

## 🎯 推荐优化方案 (分阶段)

### 第一阶段 (立即实施 - 预期减少 20-30%)
1. **启用缓存** (方案 D)
   - 缓存 scenario context
   - 缓存 TTS 结果
   - 预加载常见回复

2. **优化 prompt 长度**
   - 减少 `max_recent_messages` 从 6 → 4
   - 简化 system prompt
   - 移除不必要的格式说明

3. **调整超时配置**
   ```python
   # config.py
   ollama_num_predict: int = 256  # 从 512 → 256
   max_reply_length: int = 200    # 从 300 → 200
   ```

### 第二阶段 (1-2周 - 预期减少 30-40%)
1. **实现并行处理** (方案 B)
   - TTS 后台生成
   - WebSocket 推送音频 URL
   - 小程序接收并播放

2. **模型优化** (方案 C)
   - 测试 GLM-3-6B 或 Qwen-7B
   - 尝试 Q4_K_M 量化
   - 对比质量和速度

### 第三阶段 (2-4周 - 预期减少 40-50%)
1. **流式输出** (方案 A)
   - 改造 LLM 调用为流式
   - 小程序实时显示文本
   - 改善用户体验

2. **迁移到 vLLM**
   - 替代 Ollama
   - 启用连续批处理
   - 支持多并发请求

---

## 📋 具体代码优化建议

### 优化 1: 减少 Prompt 长度
```python
# 当前 (config.py 第 53 行)
max_recent_messages: int = 6

# 优化为
max_recent_messages: int = 4  # 减少历史消息数
```

### 优化 2: 限制生成长度
```python
# 当前 (config.py 第 27 行)
ollama_num_predict: int = 512

# 优化为
ollama_num_predict: int = 256  # 减少生成 token 数
```

### 优化 3: 缓存 Scenario Context
```python
# miniprogram_api.py 第 94 行
_scenario_cache = {}

def get_scenario_context(scenario: str, difficulty: str) -> str:
    """获取场景上下文提示 (带缓存)"""
    cache_key = f"{scenario}_{difficulty}"

    if cache_key in _scenario_cache:
        return _scenario_cache[cache_key]

    # ... 原有逻辑 ...

    _scenario_cache[cache_key] = context
    return context
```

### 优化 4: 并行 TTS 生成
```python
# miniprogram_api.py 第 639 行
@app.post("/api/chat/message", response_model=ChatMessageResponse)
async def chat_message(request: ChatMessageRequest):
    # ... 前面的代码 ...

    # 立即返回响应，TTS 在后台生成
    response = ChatMessageResponse(
        reply=reply,
        audio_url=None,  # 先返回 None
        chat_tips=chat_tips,
        # ...
    )

    # 后台任务: 生成 TTS
    asyncio.create_task(
        generate_and_cache_audio(session_id, reply, speaker)
    )

    return response
```

---

## 📊 性能对比表

| 方案 | 实施难度 | 效果 | 优先级 | 预期延时 |
|------|--------|------|-------|---------|
| 当前 | - | - | - | 6-19s |
| 缓存优化 | ⭐ | 10-20% | 🔴 高 | 5-17s |
| Prompt优化 | ⭐ | 15-25% | 🔴 高 | 5-16s |
| 并行处理 | ⭐⭐⭐ | 20-30% | 🟠 中 | 4-13s |
| 模型优化 | ⭐⭐ | 40-60% | 🟠 中 | 3-8s |
| 流式输出 | ⭐⭐⭐⭐ | 30-40% | 🟡 低 | 4-12s |
| **全部优化** | ⭐⭐⭐⭐⭐ | **60-75%** | - | **2-5s** |

---

## ✅ 立即可实施的优化 (无需改架构)

### 1. 修改 config.py
```python
# 减少历史消息数
max_recent_messages: int = 4  # 从 6 改为 4

# 减少生成长度
ollama_num_predict: int = 256  # 从 512 改为 256

# 减少回复长度
max_reply_length: int = 200  # 从 300 改为 200
```

### 2. 添加缓存
在 `miniprogram_api.py` 顶部添加:
```python
# 缓存 scenario context
_scenario_context_cache = {}

# 缓存 TTS 结果
_tts_cache = {}
```

### 3. 修改 get_scenario_context 函数
```python
def get_scenario_context(scenario: str, difficulty: str) -> str:
    cache_key = f"{scenario}_{difficulty}"
    if cache_key in _scenario_context_cache:
        return _scenario_context_cache[cache_key]

    # ... 原有逻辑 ...
    _scenario_context_cache[cache_key] = context
    return context
```

---

## 🎓 总结

**最大的延时瓶颈是 LLM 推理 (40-60% 的总延时)**

**快速见效的优化 (可立即实施)**:
1. ✅ 减少 prompt 长度 (15-25% 改进)
2. ✅ 添加缓存 (10-20% 改进)
3. ✅ 限制生成长度 (10-15% 改进)

**中期优化 (1-2周)**:
1. 🔄 并行处理 TTS (20-30% 改进)
2. 🔄 模型优化 (40-60% 改进)

**长期优化 (2-4周)**:
1. 🚀 流式输出 (30-40% 改进)
2. 🚀 迁移 vLLM (50-70% 改进)

**预期最终效果**: 从 6-19s 优化到 2-5s (减少 70-75%)
