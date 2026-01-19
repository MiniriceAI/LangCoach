# LangCoach

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-blue.svg)](https://www.langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-green.svg)](https://gradio.app/)

**🤖 AI-Powered English Learning Platform | AI 驱动的英语学习平台**

*Practice English conversation through scenario-based dialogues with intelligent memory*

*通过场景化对话和智能记忆提升英语能力*

**🎉 Phase 2 已完成：长期记忆系统 | Phase 2 Complete: Long-Term Memory System**

</div>

---

## 📖 简介 / Introduction

LangCoach 是一款基于大语言模型（LLM）的智能英语私教系统，通过场景化对话练习和词汇学习，帮助学习者提升英语口语表达能力。系统支持多种 LLM 提供者（DeepSeek、OpenAI、Ollama），并提供友好的 Web 界面，让英语学习变得轻松有趣。

**Phase 2 新增**：集成 Milvus 向量数据库，实现长期记忆功能，让 AI 能够记住你的学习历程并提供个性化建议。

LangCoach is an AI-powered English learning platform that helps learners improve their English speaking skills through scenario-based dialogue practice and vocabulary learning. Supporting multiple LLM providers (DeepSeek, OpenAI, Ollama), it provides a user-friendly web interface for an engaging learning experience.

**Phase 2 Update**: Integrated Milvus vector database for long-term memory, enabling AI to remember your learning journey and provide personalized recommendations.

## ✨ 核心特性 / Features

### 🎯 场景化对话练习 / Scenario-Based Practice
- **求职面试** (Job Interview) - 模拟真实面试场景，练习自我介绍和回答面试问题
- **酒店入住** (Hotel Check-in) - 练习酒店预订、入住登记等实用英语
- **租房** (Renting) - 学习租房咨询、价格协商等生活场景对话
- **薪资谈判** (Salary Negotiation) - 掌握职场薪资谈判技巧和表达方式

### 📚 词汇学习 / Vocabulary Learning
- 基于场景的词汇教学
- 互动式词汇练习
- 实时反馈和学习建议

### 🧠 长期记忆系统 / Long-Term Memory (Phase 2 New!)
- **智能记忆** - 使用 Milvus 向量数据库存储对话历史
- **个性化学习** - AI 基于你的学习历程提供定制化建议
- **语义检索** - 自动检索相关的历史对话，强化学习效果
- **上下文管理** - 智能控制记忆注入量，避免超出 LLM 限制

### 🤖 多 LLM 支持 / Multi-LLM Support
- **DeepSeek** (优先推荐) - 高性价比的 AI 模型
- **OpenAI** - GPT-4o-mini 等模型
- **Ollama** - 本地部署，完全免费

### 💡 智能对话特性 / Smart Features
- 实时对话提示（中英双语）
- 自动纠错和反馈
- 学习进度跟踪
- 会话历史管理
- **Phase 2**: 可配置的难度级别和对话轮次

## 🖼️ 功能演示 / Screenshots

### 📹 视频介绍 / Video Demo

<div align="center">

**点击观看完整演示视频**

[![LangCoach Demo Video](https://img.youtube.com/vi/ntxbJ3Cagb8/maxresdefault.jpg)](https://youtu.be/ntxbJ3Cagb8)

</div>


### 在线体验 / Live Demo  
点击体验: <a href="http://13.217.141.2:7860/" target="_blank">http://13.217.141.2:7860/</a>

## 🏗️ 技术架构 / Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio Web UI                          │
│  ┌────────────────────┐    ┌────────────────────┐          │
│  │  Scenario Tab      │    │   Vocab Tab        │          │
│  └────────────────────┘    └────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer (Phase 2 Enhanced)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ScenarioAgent │  │  VocabAgent  │  │  AgentBase   │     │
│  │  + Memory    │  │   + Memory   │  │  + Memory    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
┌──────────────────┐  ┌────────────┐  ┌─────────────────┐
│  LangChain Layer │  │  Memory    │  │ OpenAI/Ollama   │
│   + History      │  │  Manager   │  │   Embeddings    │
└──────────────────┘  └─────┬──────┘  └─────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    Milvus     │ (Phase 2 New!)
                    │ Vector Store  │
                    │ (etcd + minio)│
                    └───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Factory                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   DeepSeek   │  │   OpenAI     │  │   Ollama     │     │
│  │   (优先)     │  │   (备选)     │  │   (本地)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈 / Tech Stack

- **前端框架**: [Gradio](https://gradio.app/) - 快速构建机器学习 Web 应用
- **LLM 框架**: [LangChain](https://www.langchain.com/) - 构建 LLM 应用的框架
- **向量数据库**: [Milvus](https://milvus.io/) - 高性能向量相似度搜索 (Phase 2 新增)
- **Python 版本**: 3.10+
- **部署方式**: Docker / Docker Compose

### 核心模块 / Core Modules

- **Agent Layer** (`src/agents/`)
  - `AgentBase`: 代理基类，提供通用功能和长期记忆支持 (Phase 2 增强)
  - `ScenarioAgent`: 场景对话代理
  - `VocabAgent`: 词汇学习代理
  - `llm_factory.py`: LLM 工厂，统一管理多 LLM 提供者
  - `long_term_memory.py`: 长期记忆管理器 (Phase 2 新增)
  - `conversation_config.py`: 对话配置管理 (Phase 1 新增)

- **UI Layer** (`src/tabs/`)
  - `scenario_tab.py`: 场景练习界面
  - `vocab_tab.py`: 词汇学习界面

- **Utilities** (`src/utils/`)
  - `logger.py`: 日志管理工具

## 🚀 快速开始 / Quick Start

### 前置要求 / Prerequisites

- Python 3.10 或更高版本
- pip 包管理器
- （可选）Docker 和 Docker Compose（用于容器化部署）

### 方式一：本地安装 / Local Installation

#### 1. 克隆仓库 / Clone Repository

```bash
git clone https://github.com/LangCoach/LangCoach.git
cd LangCoach
```

#### 2. 创建虚拟环境 / Create Virtual Environment

使用 conda（推荐）:
```bash
conda create -n lm python=3.10
conda activate lm
```

或使用 venv:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

#### 3. 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. 配置 LLM 提供者 / Configure LLM Provider

**Phase 2.5 更新**：支持 Ollama、DeepSeek、OpenAI 三种提供者，**默认使用 Ollama + GLM-4-9B**。

**推荐配置方式：使用 .env 文件**

应用会自动加载项目根目录下的 `.env` 文件，您可以通过以下步骤配置:

```bash
# 1. 复制示例配置文件
cp .env.example .env

# 2. 编辑 .env 文件，根据需要修改配置
# 例如：设置 DeepSeek API Key
nano .env  # 或使用您喜欢的编辑器

# 3. 测试配置是否正确加载
python test_env_loading.py
```

**方式 1: 使用 Ollama（推荐 - 免费本地运行）**:

> **⚠️ 重要提示**：Ollama 服务需要**手动启动**，LangCoach 不会自动启动 Ollama 服务。

```bash
# 1. 安装 Ollama
# 访问 https://ollama.com 下载安装

# 2. 启动 Ollama 服务（必需步骤）
# 方式 A: 前台运行（会占用终端）
ollama serve

# 方式 B: 后台运行（推荐）
ollama serve &
# 或者使用 nohup
nohup ollama serve > /dev/null 2>&1 &

# 方式 C: 使用 systemd（Linux，开机自启）
# 创建服务文件: /etc/systemd/system/ollama.service
# 然后: systemctl enable ollama && systemctl start ollama

# 3. 验证服务是否运行
curl http://localhost:11434/api/tags
# 或者
ollama list

# 4. 拉取 GLM-4-9B 模型（默认模型）
ollama pull unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL

# 5. 验证模型已安装
ollama list

# 6. 可选：配置环境变量（使用默认值可跳过）
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL
```

**常见问题排查**：
- 如果启动 LangCoach 时提示无法连接 Ollama，请检查：
  1. Ollama 服务是否正在运行：`curl http://localhost:11434/api/tags`
  2. 模型是否已安装：`ollama list`
  3. 如果使用 Docker，确保 Ollama 运行在宿主机或配置正确的 `OLLAMA_BASE_URL`

**方式 2: 使用 DeepSeek（高性价比 API）**:
```bash
export DEEPSEEK_API_KEY=your_deepseek_api_key
# 可选配置
export DEEPSEEK_MODEL=deepseek-chat  # 默认值
```

**方式 3: 使用 OpenAI（高性能）**:
```bash
export OPENAI_API_KEY=your_openai_api_key
# 可选配置
export OPENAI_MODEL=gpt-4o-mini  # 默认值
```

**高级配置 - 自定义优先级**:
```bash
# 默认优先级: ollama > deepseek > openai
# 可以自定义优先级顺序
export LLM_PROVIDER_PRIORITY=deepseek,openai,ollama

# 禁用某个提供者
export OLLAMA_ENABLED=false
```

> **💡 提示**:
> - **推荐使用 `.env` 文件管理配置** - 应用会自动加载,无需手动 export
> - 环境变量优先级: 系统环境变量 > `.env` 文件
> - Ollama 是默认提供者，无需 API key，完全免费
> - 如果未配置任何提供者，系统会尝试连接本地 Ollama
> - 详细配置请参考 `.env.example` 文件
> - 运行 `python test_env_loading.py` 验证配置是否正确加载

#### 5. 运行应用 / Run Application

```bash
python src/main.py
```

应用启动后，在浏览器中访问 `http://localhost:8300` 开始使用。

> **注意**: Phase 2 默认端口已从 7860 改为 8300。

**高级启动选项 / Advanced Options**:

```bash
# 指定端口
python src/main.py 7861

# 强制重启（自动停止占用端口的旧进程）
python src/main.py --force

# 指定端口并强制重启
python src/main.py 7861 --force

# 使用环境变量配置
GRADIO_PORT=7861 python src/main.py
GRADIO_FORCE_RESTART=true python src/main.py
```

### 方式二：Docker 部署 / Docker Deployment

#### 使用 Docker Compose（推荐）

1. **设置环境变量**:
```bash
export DEEPSEEK_API_KEY=your_api_key
# 或创建 .env 文件
echo "DEEPSEEK_API_KEY=your_api_key" > .env
```

2. **启动服务**:
```bash
docker-compose up -d
```

3. **查看日志**:
```bash
docker-compose logs -f
```

4. **停止服务**:
```bash
docker-compose down
```

详细的 Docker 部署指南请参考 [DOCKER.md](DOCKER.md)。

## ⚙️ 配置说明 / Configuration

### 环境变量 / Environment Variables

| 变量名 | 说明 | 必需 | 默认值 |
|--------|------|------|--------|
| **LLM 配置 (Phase 2.5 增强)** |
| `LLM_PROVIDER_PRIORITY` | 提供者优先级（逗号分隔） | 否 | ollama,deepseek,openai |
| **Ollama 配置（默认提供者）** |
| `OLLAMA_MODEL` | Ollama 模型名称 | 否 | unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | 否 | http://localhost:11434 |
| `OLLAMA_TEMPERATURE` | 温度参数 | 否 | 0.8 |
| `OLLAMA_MAX_TOKENS` | 最大 token 数 | 否 | 8192 |
| `OLLAMA_ENABLED` | 是否启用 | 否 | true |
| **DeepSeek 配置** |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 否* | - |
| `DEEPSEEK_MODEL` | 模型名称 | 否 | deepseek-chat |
| `DEEPSEEK_BASE_URL` | API 地址 | 否 | https://api.deepseek.com |
| `DEEPSEEK_TEMPERATURE` | 温度参数 | 否 | 0.8 |
| `DEEPSEEK_MAX_TOKENS` | 最大 token 数 | 否 | 8192 |
| `DEEPSEEK_ENABLED` | 是否启用 | 否 | true |
| **OpenAI 配置** |
| `OPENAI_API_KEY` | OpenAI API 密钥 | 否* | - |
| `OPENAI_MODEL` | 模型名称 | 否 | gpt-4o-mini |
| `OPENAI_BASE_URL` | API 地址（可选） | 否 | - |
| `OPENAI_TEMPERATURE` | 温度参数 | 否 | 0.8 |
| `OPENAI_MAX_TOKENS` | 最大 token 数 | 否 | 8192 |
| `OPENAI_ENABLED` | 是否启用 | 否 | true |
| **长期记忆配置 (Phase 2)** |
| `MILVUS_HOST` | Milvus 主机地址 | 否** | - |
| `MILVUS_PORT` | Milvus 端口 | 否 | 19530 |
| **应用配置** |
| `GRADIO_PORT` | 应用监听端口 | 否 | 8300 (Phase 2 更新) |
| `GRADIO_FORCE_RESTART` | 自动停止占用端口的进程 | 否 | false |

\* Ollama 是默认提供者，无需 API key。如果想使用 DeepSeek 或 OpenAI，需要配置对应的 API key。

\** 如果不设置 `MILVUS_HOST`，长期记忆功能将被禁用，应用将以 Phase 1 模式运行。

**完整配置示例**: 参见 `.env.example` 文件。

### LLM 提供者优先级 / LLM Provider Priority

**Phase 2.5 更新**：系统现在支持可配置的提供者优先级。

**默认优先级**（从高到低）:
1. **Ollama** - 本地运行，免费，默认使用 GLM-4-9B 模型
2. **DeepSeek** - 如果配置了 `DEEPSEEK_API_KEY`，使用 DeepSeek
3. **OpenAI** - 如果配置了 `OPENAI_API_KEY`，使用 OpenAI

**自定义优先级**:
```bash
# 示例：优先使用 DeepSeek，然后 OpenAI，最后 Ollama
export LLM_PROVIDER_PRIORITY=deepseek,openai,ollama
```

**禁用某个提供者**:
```bash
# 禁用 Ollama，只使用 API 提供者
export OLLAMA_ENABLED=false
```

**模型推荐**:
- **Ollama 推荐模型**:
  - `unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL` (默认，8B 参数，中英双语优秀)
  - `llama3.1:8b-instruct-q8_0` (Llama 3.1 8B)
  - `qwen2.5:14b` (Qwen 2.5 14B，更强大)

- **DeepSeek 模型**: `deepseek-chat` (高性价比)

- **OpenAI 模型**:
  - `gpt-4o-mini` (便宜快速)
  - `gpt-4o` (性能最佳)

## 📁 项目结构 / Project Structure

```
LangCoach/
├── src/                        # 源代码目录
│   ├── agents/                 # 代理模块
│   │   ├── agent_base.py       # 代理基类 (Phase 2 增强)
│   │   ├── scenario_agent.py   # 场景代理
│   │   ├── vocab_agent.py      # 词汇代理
│   │   ├── llm_factory.py      # LLM 工厂
│   │   ├── long_term_memory.py # 长期记忆管理器 (Phase 2 新增)
│   │   ├── conversation_config.py # 对话配置 (Phase 1 新增)
│   │   └── session_history.py  # 会话历史管理
│   ├── tabs/                   # UI 标签页
│   │   ├── scenario_tab.py     # 场景练习界面
│   │   └── vocab_tab.py        # 词汇学习界面
│   ├── utils/                  # 工具模块
│   │   └── logger.py           # 日志工具
│   └── main.py                 # 应用入口
├── content/                    # 内容资源
│   ├── intro/                  # 初始消息（JSON）
│   └── page/                   # 页面介绍（Markdown）
├── prompts/                    # 提示词文件
│   ├── templates/              # Jinja2 模板 (Phase 1 新增)
│   ├── hotel_checkin_prompt.txt
│   ├── job_interview_prompt.txt
│   ├── renting_prompt.txt
│   ├── salary_negotiation_prompt.txt
│   └── vocab_study_prompt.txt
├── tests/                      # 测试目录
│   ├── agents/                 # 代理测试
│   ├── utils/                  # 工具测试
│   └── run_tests.sh            # 测试运行脚本
├── scripts/                    # 脚本文件
├── images/                     # 项目图片
├── volumes/                    # Docker 数据卷 (Phase 2 新增, gitignored)
├── docker-compose.yml          # Docker Compose 配置 (Phase 2 更新)
├── docker-compose.dev.yml      # 开发环境配置 (Phase 2 更新)
├── Dockerfile                  # Docker 镜像构建文件
├── requirements.txt            # Python 依赖 (Phase 2 新增 pymilvus)
├── .env.example                # 环境变量示例 (Phase 2 新增)
├── PHASE2_SETUP.md            # Phase 2 设置指南 (Phase 2 新增)
├── PRODUCT_PLAN.md            # 产品开发计划
├── LICENSE                     # 许可证
└── README.md                   # 项目说明文档
```

## 🧪 开发指南 / Development

### 运行测试 / Running Tests

```bash
# 运行所有测试
pytest

# 运行特定模块的测试
pytest tests/agents/

# 运行测试并查看覆盖率
pytest --cov=src --cov-report=term-missing

# 生成 HTML 覆盖率报告
pytest --cov=src --cov-report=html

# 使用测试脚本（自动检查覆盖率）
./tests/run_tests.sh
```

更多测试信息请参考 [tests/README.md](tests/README.md)。

### 开发环境 / Development Environment

**本地热重载开发 / Local Hot Reload**:

应用支持 Gradio 热重载模式，修改代码后自动生效：

```bash
# 使用 gradio 命令启动（推荐）
gradio src/main.py

# 或使用 --force 参数避免端口冲突
python src/main.py --force
```

**使用 Docker Compose 开发**:

使用开发模式的 Docker Compose 配置，支持代码热重载：

```bash
docker-compose -f docker-compose.dev.yml up
```

开发环境会挂载源代码目录，修改代码后自动生效。

### 代码规范 / Code Style

- 遵循 PEP 8 Python 代码规范
- 使用类型提示（Type Hints）
- 编写清晰的文档字符串（Docstrings）

## 📝 许可证 / License

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 📮 联系我们 / Contact

- 项目主页: [https://github.com/miniriceai/LangCoach](https://github.com/miniriceai/LangCoach)
- 问题反馈: [GitHub Issues](https://github.com/MiniriceAI/LangCoach/issues)

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐ Star！**

**If this project helps you, please give us a ⭐ Star!**

Made with ❤️ by the LangCoach Team

</div>
