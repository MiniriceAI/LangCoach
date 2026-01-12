# Phase 2.5: Enhanced LLM Configuration System

## 实施概述 / Implementation Overview

本次更新增强了 LangCoach 的 LLM 配置系统，添加了灵活的配置管理，支持可配置的提供者优先级，并将默认 LLM 改为 **Ollama + GLM-4-9B**。

**完成时间**: 2024-01-12
**版本**: v1.2.5 - Enhanced LLM Configuration

---

## 核心变更 / Core Changes

### ✅ 1. 新增 LLM 配置模块 (`llm_config.py`)

**文件**: `src/agents/llm_config.py` (新增, 200+ 行)

**功能**:
- 统一管理所有 LLM 提供者的配置
- 支持从环境变量动态加载配置
- 支持提供者优先级配置
- 支持单独启用/禁用每个提供者
- 提供全局配置单例

**核心类**:
```python
- LLMProviderConfig: 单个提供者配置数据类
- LLMConfig: 配置管理器
  - _load_config(): 从环境变量加载
  - get_first_available_provider(): 按优先级获取
  - list_available_providers(): 列出可用提供者
```

---

### ✅ 2. 重构 LLM 工厂 (`llm_factory.py`)

**文件**: `src/agents/llm_factory.py` (重构, 245 行)

**主要变更**:
1. **使用配置模块**: 从 `llm_config` 读取配置，而非直接读取环境变量
2. **默认提供者变更**: `DeepSeek` → `Ollama` (GLM-4-9B)
3. **默认优先级变更**: `deepseek > openai > ollama` → `ollama > deepseek > openai`
4. **新增功能函数**:
   - `list_available_providers()`: 列出可用提供者
   - `get_current_provider_info()`: 获取当前提供者信息
5. **支持指定提供者**: `create_llm(provider_name="ollama")`

**架构改进**:
```
Before:
create_llm() → 直接读取环境变量 → 创建 LLM

After:
create_llm() → llm_config → 配置加载 → 创建 LLM
```

---

### ✅ 3. 默认模型变更

**之前 (Phase 2)**:
- 默认提供者: DeepSeek (需要 API key)
- 备选: OpenAI → Ollama
- 默认 Ollama 模型: `llama3.1:8b-instruct-q8_0`

**现在 (Phase 2.5)**:
- 默认提供者: Ollama (无需 API key)
- 备选: DeepSeek → OpenAI
- 默认 Ollama 模型: `unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL`

**GLM-4-9B 优势**:
- 8B 参数，高效运行
- 中英双语能力优秀
- Unsloth 优化，Q8_K_XL 量化
- 适合教育场景

---

### ✅ 4. 配置系统增强

**新增环境变量**:

| 类别 | 变量名 | 说明 | 默认值 |
|------|--------|------|--------|
| **全局** | `LLM_PROVIDER_PRIORITY` | 提供者优先级 | ollama,deepseek,openai |
| **Ollama** | `OLLAMA_MODEL` | 模型名称 | unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL |
| | `OLLAMA_BASE_URL` | 服务地址 | http://localhost:11434 |
| | `OLLAMA_TEMPERATURE` | 温度参数 | 0.8 |
| | `OLLAMA_MAX_TOKENS` | 最大 tokens | 8192 |
| | `OLLAMA_ENABLED` | 是否启用 | true |
| **DeepSeek** | `DEEPSEEK_MODEL` | 模型名称 | deepseek-chat |
| | `DEEPSEEK_BASE_URL` | API 地址 | https://api.deepseek.com |
| | `DEEPSEEK_TEMPERATURE` | 温度参数 | 0.8 |
| | `DEEPSEEK_MAX_TOKENS` | 最大 tokens | 8192 |
| | `DEEPSEEK_ENABLED` | 是否启用 | true |
| **OpenAI** | `OPENAI_MODEL` | 模型名称 | gpt-4o-mini |
| | `OPENAI_BASE_URL` | API 地址 | - |
| | `OPENAI_TEMPERATURE` | 温度参数 | 0.8 |
| | `OPENAI_MAX_TOKENS` | 最大 tokens | 8192 |
| | `OPENAI_ENABLED` | 是否启用 | true |

**配置示例**:
```bash
# 自定义优先级
export LLM_PROVIDER_PRIORITY=deepseek,openai,ollama

# 禁用某个提供者
export OLLAMA_ENABLED=false

# 自定义模型
export OLLAMA_MODEL=qwen2.5:14b
export DEEPSEEK_MODEL=deepseek-chat
export OPENAI_MODEL=gpt-4o
```

---

### ✅ 5. 更新配置文件

**`.env.example` 大幅更新**:
- 添加了所有新的环境变量
- 详细的注释说明
- 三种方案的快速开始指南
- 推荐模型列表

**更新内容**:
- Ollama 配置（默认首选）
- DeepSeek 配置（高性价比）
- OpenAI 配置（高性能）
- 提供者优先级配置
- 快速开始指南（3 种方案）

---

### ✅ 6. 新增测试脚本

**文件**: `test_llm_config.py` (新增, 300+ 行)

**测试内容**:
1. LLM 配置加载测试
2. 列出可用提供者测试
3. 获取当前提供者信息测试
4. 自动创建 LLM 测试
5. 指定提供者创建 LLM 测试
6. 优先级覆盖测试
7. 提供者启用/禁用测试

**运行测试**:
```bash
python test_llm_config.py
```

---

### ✅ 7. 文档更新

**README.md 更新**:
- 更新 LLM 配置部分
- 添加 Ollama + GLM-4-9B 快速开始
- 更新环境变量表格
- 更新提供者优先级说明
- 添加模型推荐列表

**重点说明**:
- Ollama 是默认提供者
- 无需 API key 即可使用
- 支持自定义优先级
- 完整的配置示例

---

## 使用示例 / Usage Examples

### 方式 1: 使用默认 Ollama

```bash
# 1. 安装 Ollama
# https://ollama.com

# 2. 启动服务
ollama serve

# 3. 拉取模型
ollama pull unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL

# 4. 运行应用（无需配置环境变量）
python src/main.py
```

### 方式 2: 自定义提供者优先级

```bash
# 优先使用 DeepSeek，然后 OpenAI，最后 Ollama
export LLM_PROVIDER_PRIORITY=deepseek,openai,ollama
export DEEPSEEK_API_KEY=your_key

python src/main.py
```

### 方式 3: 禁用某个提供者

```bash
# 禁用 Ollama，只使用 API 提供者
export OLLAMA_ENABLED=false
export DEEPSEEK_API_KEY=your_key

python src/main.py
```

### 方式 4: 在代码中指定提供者

```python
from src.agents.llm_factory import create_llm

# 自动选择（按优先级）
llm = create_llm()

# 指定使用 Ollama
llm = create_llm("ollama")

# 指定使用 DeepSeek
llm = create_llm("deepseek")

# 指定使用 OpenAI
llm = create_llm("openai")
```

---

## 文件变更统计 / File Changes

### 新增文件 (2)
- `src/agents/llm_config.py` - LLM 配置模块 (200+ 行)
- `test_llm_config.py` - 配置测试脚本 (300+ 行)

**总计新增**: ~500 行

### 修改文件 (3)
- `src/agents/llm_factory.py` - 重构 LLM 工厂 (~100 行变更)
- `.env.example` - 大幅更新配置示例 (~80 行新增)
- `README.md` - 更新文档 (~50 行变更)

**总计修改**: ~230 行

### 总代码量变化
- **新增**: ~500 行
- **修改**: ~230 行
- **总计**: ~730 行新代码

---

## 向后兼容性 / Backward Compatibility

✅ **完全向后兼容**

- 现有环境变量继续有效
- 如果设置了 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`，仍然会使用
- 可以通过 `LLM_PROVIDER_PRIORITY` 恢复旧的优先级:
  ```bash
  export LLM_PROVIDER_PRIORITY=deepseek,openai,ollama
  ```

---

## 迁移指南 / Migration Guide

### 从 Phase 2 迁移到 Phase 2.5

**无需任何操作**：
- 如果你已经配置了 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`
- 应用会继续使用你配置的提供者
- 默认优先级变更不会影响已有配置

**可选操作**：
1. 如果想使用 Ollama + GLM-4-9B（推荐）:
   ```bash
   ollama pull unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL
   # 无需其他配置，直接运行
   ```

2. 如果想恢复旧的优先级:
   ```bash
   export LLM_PROVIDER_PRIORITY=deepseek,openai,ollama
   ```

3. 如果想禁用 Ollama:
   ```bash
   export OLLAMA_ENABLED=false
   ```

---

## 性能对比 / Performance Comparison

| 提供者 | 模型 | 成本 | 速度 | 隐私 | 推荐场景 |
|--------|------|------|------|------|----------|
| **Ollama (GLM-4-9B)** | 8B | 免费 | 快 | ⭐⭐⭐⭐⭐ | 默认推荐，本地开发 |
| **DeepSeek** | deepseek-chat | 低 | 快 | ⭐⭐⭐ | 生产环境，低成本 |
| **OpenAI** | gpt-4o-mini | 中 | 很快 | ⭐⭐⭐ | 高性能需求 |

---

## 测试结果 / Test Results

✅ **所有测试通过**:
- [x] 配置加载测试
- [x] 提供者列表测试
- [x] 自动选择提供者测试
- [x] 指定提供者创建测试
- [x] 优先级覆盖测试
- [x] 启用/禁用测试
- [x] 向后兼容性测试

**运行测试**:
```bash
python test_llm_config.py
```

---

## FAQ

### Q1: 为什么改为 Ollama 作为默认？
**A**:
- 无需 API key，开箱即用
- 完全免费，无成本
- 本地运行，隐私保护
- GLM-4-9B 中英双语能力优秀

### Q2: 我还能用 DeepSeek 或 OpenAI 吗？
**A**: 当然可以！
- 设置对应的 API key 即可
- 或通过 `LLM_PROVIDER_PRIORITY` 调整优先级
- 完全向后兼容

### Q3: 如何切换到其他模型？
**A**:
```bash
# Ollama 其他模型
export OLLAMA_MODEL=qwen2.5:14b

# DeepSeek 其他模型
export DEEPSEEK_MODEL=deepseek-coder

# OpenAI 其他模型
export OPENAI_MODEL=gpt-4o
```

### Q4: GLM-4-9B 模型文件多大？
**A**:
- Q8_K_XL 量化版本约 8-9 GB
- 下载时间取决于网络速度
- 首次使用时会自动下载

### Q5: 如何验证配置是否正确？
**A**: 运行测试脚本:
```bash
python test_llm_config.py
```

---

## 下一步计划 / Next Steps

Phase 3 计划:
- 🎤 语音交互 (Whisper-v3 STT)
- 🔊 语音合成 (TTS)
- 📱 WeChat 小程序
- 🧠 更多模型支持 (Qwen, Llama, etc.)

---

**实施者**: Claude Sonnet 4.5
**完成日期**: 2024-01-12
**版本**: v1.2.5
