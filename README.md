# LangCoach

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-blue.svg)](https://www.langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-green.svg)](https://gradio.app/)

**🤖 AI-Powered English Learning Platform | AI 驱动的英语学习平台**

*Practice English conversation through scenario-based dialogues and vocabulary learning*

*通过场景化对话和词汇学习提升英语能力*

</div>

---

## 📖 简介 / Introduction

LangCoach 是一款基于大语言模型（LLM）的智能英语私教系统，通过场景化对话练习和词汇学习，帮助学习者提升英语口语表达能力。系统支持多种 LLM 提供者（DeepSeek、OpenAI、Ollama），并提供友好的 Web 界面，让英语学习变得轻松有趣。

LangCoach is an AI-powered English learning platform that helps learners improve their English speaking skills through scenario-based dialogue practice and vocabulary learning. Supporting multiple LLM providers (DeepSeek, OpenAI, Ollama), it provides a user-friendly web interface for an engaging learning experience.

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

### 🤖 多 LLM 支持 / Multi-LLM Support
- **DeepSeek** (优先推荐) - 高性价比的 AI 模型
- **OpenAI** - GPT-4o-mini 等模型
- **Ollama** - 本地部署，完全免费

### 💡 智能对话特性 / Smart Features
- 实时对话提示（中英双语）
- 自动纠错和反馈
- 学习进度跟踪
- 会话历史管理

## 🖼️ 功能演示 / Screenshots

### 📹 视频介绍 / Video Demo
<div align="center">
  <video width="800" controls>
    <source src="images/langcoach.mp4" type="video/mp4">
    您的浏览器不支持视频标签，请使用现代浏览器查看。
  </video>
</div>


### 在线体验 / Live Demo  
点击体验: <a href="http://34.207.175.3:7860/" target="_blank">http://34.207.175.3:7860/</a>

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
│                    Agent Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ScenarioAgent │  │  VocabAgent  │  │  AgentBase   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangChain Layer                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         RunnableWithMessageHistory                  │   │
│  │  ┌──────────────┐  ┌──────────────────────────┐    │   │
│  │  │ChatPrompt    │  │  Session History Manager │    │   │
│  │  │Template      │  │  (InMemoryChatMessage)   │    │   │
│  │  └──────────────┘  └──────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
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
- **Python 版本**: 3.10+
- **部署方式**: Docker / Docker Compose

### 核心模块 / Core Modules

- **Agent Layer** (`src/agents/`)
  - `AgentBase`: 代理基类，提供通用功能
  - `ScenarioAgent`: 场景对话代理
  - `VocabAgent`: 词汇学习代理
  - `llm_factory.py`: LLM 工厂，统一管理多 LLM 提供者

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

#### 4. 配置环境变量 / Configure Environment Variables

至少需要配置一个 LLM 提供者的 API 密钥：

**使用 DeepSeek（推荐）**:
```bash
export DEEPSEEK_API_KEY=your_deepseek_api_key
```

**使用 OpenAI**:
```bash
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_MODEL=gpt-4o-mini  # 可选，默认为 gpt-4o-mini
```

**使用 Ollama（本地部署）**:
```bash
# 首先确保 Ollama 正在运行
ollama serve

# 拉取模型（首次使用）
ollama pull llama3.1:8b

# 配置环境变量（可选）
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b
```

#### 5. 运行应用 / Run Application

```bash
python src/main.py
```

应用启动后，在浏览器中访问 `http://localhost:7860` 开始使用。

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

| 变量名 | 说明 | 必需 | 优先级 |
|--------|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 否* | 1（最高） |
| `OPENAI_API_KEY` | OpenAI API 密钥 | 否* | 2 |
| `OPENAI_MODEL` | OpenAI 模型名称 | 否 | 2（默认: gpt-4o-mini） |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | 否 | 3（默认: http://localhost:11434） |
| `OLLAMA_MODEL` | Ollama 模型名称 | 否 | 3（默认: llama3.1:8b） |

\* 至少需要配置一个 LLM 提供者的 API 密钥，或者确保 Ollama 在本地运行。

### LLM 提供者优先级 / LLM Provider Priority

系统按以下优先级选择 LLM 提供者：
1. **DeepSeek** - 如果配置了 `DEEPSEEK_API_KEY`，优先使用
2. **OpenAI** - 如果配置了 `OPENAI_API_KEY`，使用 OpenAI
3. **Ollama** - 如果以上都未配置，回退到本地 Ollama（需要 Ollama 服务运行）

## 📁 项目结构 / Project Structure

```
LangCoach/
├── src/                        # 源代码目录
│   ├── agents/                 # 代理模块
│   │   ├── agent_base.py       # 代理基类
│   │   ├── scenario_agent.py   # 场景代理
│   │   ├── vocab_agent.py      # 词汇代理
│   │   ├── llm_factory.py      # LLM 工厂
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
│   ├── docker-build.sh
│   └── docker-run.sh
├── images/                     # 项目图片
├── docker-compose.yml          # Docker Compose 配置
├── docker-compose.dev.yml      # 开发环境配置
├── Dockerfile                  # Docker 镜像构建文件
├── requirements.txt            # Python 依赖
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

使用开发模式的 Docker Compose 配置，支持代码热重载：

```bash
docker-compose -f docker-compose.dev.yml up
```

开发环境会挂载源代码目录，修改代码后自动生效。

### 代码规范 / Code Style

- 遵循 PEP 8 Python 代码规范
- 使用类型提示（Type Hints）
- 编写清晰的文档字符串（Docstrings）

## 🤝 贡献指南 / Contributing

我们欢迎所有形式的贡献！请遵循以下步骤：

1. **Fork 本仓库**
2. **创建特性分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送到分支** (`git push origin feature/AmazingFeature`)
5. **开启 Pull Request**

### 贡献类型 / Types of Contributions

- 🐛 修复 Bug
- ✨ 添加新功能
- 📝 改进文档
- 🎨 UI/UX 优化
- ⚡ 性能优化
- 🧪 增加测试覆盖率

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
