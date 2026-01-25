# LangCoach Mini Program API 文档

## 概述

本文档描述 LangCoach 小程序后端 API 的架构设计和使用方法。API 服务整合了对话管理、语音识别、语音合成、词典查询等功能，通过单一端口（8600）对外提供服务。

## 服务架构

### 技术栈

| 服务 | 技术方案 | 说明 |
|------|----------|------|
| **LLM** | Ollama + GLM-4-9B | `hf.co/unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL` |
| **TTS** | Edge-TTS (Microsoft Azure) | 快速模式，无需本地模型 |
| **STT** | Whisper-large-v3 + 4bit | `unsloth/whisper-large-v3` |

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    WeChat Mini Program                          │
│                   (LangCoach-MiniProgram)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Home Tab  │  │  Chat Tab   │  │ Profile Tab │             │
│  │ (自定义场景)│  │ (对话练习)  │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Unified API Server (:8600)                      │
│                  (miniprogram_api.py)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      FastAPI App                          │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │  │
│  │  │ Chat API   │ │ Custom     │ │ Speech API │            │  │
│  │  │ /api/chat/*│ │ Scenario   │ │/api/trans* │            │  │
│  │  │            │ │ /api/      │ │/api/synth* │            │  │
│  │  │            │ │ custom-*   │ │            │            │  │
│  │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘            │  │
│  │        │              │              │                    │  │
│  │        ▼              ▼              ▼                    │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐    │  │
│  │  │  Session   │ │  Custom    │ │   Speech Services  │    │  │
│  │  │  Manager   │ │  Prompt    │ │  STT: Whisper      │    │  │
│  │  │            │ │  Cache     │ │  TTS: Edge-TTS     │    │  │
│  │  └─────┬──────┘ └────────────┘ └────────────────────┘    │  │
│  └────────┼──────────────────────────────────────────────────┘  │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              LLM Agent Layer                              │  │
│  │  ┌────────────────┐  ┌────────────────┐                  │  │
│  │  │ ScenarioAgent  │  │ CustomScenario │                  │  │
│  │  │ (job_interview │  │ (动态生成)     │                  │  │
│  │  │  hotel_checkin │  │                │                  │  │
│  │  │  renting, etc) │  │                │                  │  │
│  │  └───────┬────────┘  └────────────────┘                  │  │
│  └──────────┼────────────────────────────────────────────────┘  │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Ollama + GLM-4-9B-0414-GGUF:Q8_K_XL            │  │
│  │              (Local LLM Inference)                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 统一 API | `src/api/miniprogram_api.py` | 整合所有小程序接口 |
| 场景代理 | `src/agents/scenario_agent.py` | 处理场景对话逻辑 |
| TTS 服务 | Edge-TTS | 文字转语音 (Microsoft Azure) |
| STT 服务 | `src/stt/service.py` | 语音转文字 (Whisper) |
| LLM 工厂 | `src/agents/llm_factory.py` | 管理 LLM 提供者 |

---

## 快速开始

### 前置条件

1. **Ollama 服务运行中**
   ```bash
   # 启动 Ollama
   ollama serve

   # 拉取 GLM-4-9B 模型
   ollama pull hf.co/unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL

   # 验证
   ollama list
   ```

2. **Python 环境**
   ```bash
   source /workspace/miniconda3/bin/activate base
   ```

### 启动服务

```bash
cd /workspace/LangCoach

# 方式 1: 使用启动脚本（推荐，默认预加载所有服务）
./run_miniprogram_api.sh

# 方式 2: 直接使用 uvicorn
source /workspace/miniconda3/bin/activate base
uvicorn src.api.miniprogram_api:app --host 0.0.0.0 --port 8600
```

### 验证服务

```bash
# 健康检查
curl http://localhost:8600/health

# 获取场景列表
curl http://localhost:8600/api/scenarios
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `API_HOST` | 监听地址 | `0.0.0.0` |
| `API_PORT` | 监听端口 | `8600` |
| `PRELOAD_MODELS` | 启动时预加载模型 | `false` |

---

## API 接口列表

### 基础接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/scenarios` | GET | 获取可用场景列表 |
| `/api/speakers` | GET | 获取 TTS 语音角色列表 |

### 对话接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat/start` | POST | 开始新的对话会话 |
| `/api/chat/message` | POST | 发送消息并获取 AI 回复 |
| `/api/chat/rate` | POST | 评价会话 |
| `/api/chat/feedback` | POST | 消息反馈（点赞/踩） |

### 自定义场景接口 (新增)

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/custom-scenario/extract` | POST | 从用户输入提取场景信息 |
| `/api/custom-scenario/generate` | POST | 生成自定义场景的 prompt |

### 语音接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/transcribe` | POST | 语音转文字 (STT) |
| `/api/synthesize` | POST | 文字转语音 (TTS) |

### 其他接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/dictionary` | GET | 词典查询 |
| `/api/auth/wechat` | POST | 微信登录 |

---

## 接口详细说明

### 1. 健康检查

**GET** `/health`

**响应示例:**
```json
{
  "status": "healthy",
  "service": "langcoach-miniprogram-api",
  "timestamp": "2024-01-21T12:00:00.000000",
  "sessions_count": 5,
  "tts_initialized": true,
  "stt_initialized": true
}
```

---

### 2. 获取场景列表

**GET** `/api/scenarios`

**响应示例:**
```json
{
  "scenarios": [
    {
      "id": "job_interview",
      "title": "Job Interview",
      "greeting": "Hello! I'm your interviewer today..."
    },
    {
      "id": "hotel_checkin",
      "title": "Hotel Checkin",
      "greeting": "Good evening! Welcome to our hotel..."
    }
  ]
}
```

---

### 3. 开始对话会话

**POST** `/api/chat/start`

**请求体:**
```json
{
  "scenario": {
    "scenario": "job_interview"
  },
  "level": "B1",
  "turns": 20
}
```

**参数说明:**
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `scenario` | object | 否 | 场景配置，包含 `scenario` 或 `id` 字段 |
| `level` | string | 否 | 难度等级: A1, A2, B1, B2, C1, C2 |
| `turns` | int | 否 | 对话轮数，默认 20 |

**响应示例:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "greeting": "Hello! I'm your interviewer today. Please have a seat and let's begin. Could you start by telling me a little about yourself?",
  "scenario": "job_interview",
  "level": "B1",
  "max_turns": 20
}
```

---

### 4. 发送消息

**POST** `/api/chat/message`

**请求体:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "I have 5 years of experience in software development."
}
```

**响应示例:**
```json
{
  "reply": "That's great! Could you tell me more about the projects you've worked on?",
  "feedback": null,
  "session_ended": false,
  "current_turn": 1,
  "report": null
}
```

**会话结束时的响应:**
```json
{
  "reply": "Thank you for your time today. We'll be in touch soon.",
  "feedback": null,
  "session_ended": true,
  "current_turn": 20,
  "report": {
    "grammarScore": 85,
    "vocabularyScore": 78,
    "fluencyScore": 82,
    "totalTurns": 20,
    "tips": [
      "Try using more complex sentence structures",
      "Good use of vocabulary!"
    ]
  }
}
```

---

### 5. 语音转文字

**POST** `/api/transcribe`

**请求:** `multipart/form-data`

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `audio` | file | 是 | 音频文件 (MP3, WAV 等) |
| `session_id` | string | 否 | 会话 ID |
| `language` | string | 否 | 语言代码 |

**响应示例:**
```json
{
  "text": "I have five years of experience in software development.",
  "language": "en"
}
```

---

### 6. 文字转语音

**POST** `/api/synthesize`

**请求体:**
```json
{
  "text": "Hello, how are you today?",
  "speaker": "Ceylia",
  "fast_mode": true
}
```

**参数说明:**
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 要合成的文本 |
| `speaker` | string | 否 | 语音角色: Ceylia, Tifa |
| `fast_mode` | bool | 否 | 使用 Edge-TTS 快速模式，默认 true |

**响应示例:**
```json
{
  "audio_base64": "//uQxAAAAAANIAAAAAExBTUUzLjEwMFVV...",
  "sample_rate": 24000,
  "speaker": "Ceylia",
  "text": "Hello, how are you today?",
  "format": "mp3"
}
```

---

### 7. 词典查询

**GET** `/api/dictionary?word=interview`

**响应示例:**
```json
{
  "word": "interview",
  "phonetic": "/ˈɪntərˌvjuː/",
  "definition": "a formal meeting for assessment"
}
```

---

### 8. 会话评分

**POST** `/api/chat/rate`

**请求体:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "rating": 5,
  "feedback": "Very helpful practice session!"
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "Rating submitted"
}
```

---

### 9. 消息反馈

**POST** `/api/chat/feedback`

**请求体:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": "msg_1705833600000",
  "feedback": "up"
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "Feedback submitted"
}
```

---

### 10. 自定义场景 - 提取场景信息

**POST** `/api/custom-scenario/extract`

从用户输入的场景描述中提取关键信息，包括角色、目标、挑战、难度等。

**请求体:**
```json
{
  "user_input": "小学三年级学生，去超市买文具"
}
```

**响应示例:**
```json
{
  "ai_role": "supermarket cashier",
  "ai_role_cn": "超市收银员",
  "user_role": "third-grade primary school student",
  "user_role_cn": "小学三年级学生",
  "goal": "Successfully purchase stationery items at the supermarket",
  "goal_cn": "成功在超市购买文具",
  "challenge": "Communicate clearly about what items you need and handle the payment process",
  "challenge_cn": "清楚地表达需要什么物品并完成付款流程",
  "greeting": "Hello there! Welcome to our store. What can I help you find today?",
  "difficulty_level": "easy",
  "speaking_speed": "slow",
  "vocabulary": "simple",
  "scenario_summary": "A third-grade student buying stationery at a supermarket",
  "scenario_summary_cn": "小学三年级学生在超市买文具"
}
```

**响应字段说明:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `ai_role` | string | AI 扮演的角色（英文） |
| `ai_role_cn` | string | AI 扮演的角色（中文） |
| `user_role` | string | 用户扮演的角色（英文） |
| `user_role_cn` | string | 用户扮演的角色（中文） |
| `goal` | string | 对话目标（英文） |
| `goal_cn` | string | 对话目标（中文） |
| `challenge` | string | 挑战描述（英文） |
| `challenge_cn` | string | 挑战描述（中文） |
| `greeting` | string | AI 的开场白 |
| `difficulty_level` | string | 难度级别: easy, medium, hard |
| `speaking_speed` | string | 语速: slow, medium, fast |
| `vocabulary` | string | 词汇难度: simple, medium, advanced |
| `scenario_summary` | string | 场景摘要（英文） |
| `scenario_summary_cn` | string | 场景摘要（中文） |

---

### 11. 自定义场景 - 生成场景 Prompt

**POST** `/api/custom-scenario/generate`

根据提取的场景信息生成对话 prompt，并返回场景 ID 和开场白音频。

**请求体:**
```json
{
  "scenario_info": {
    "ai_role": "supermarket cashier",
    "ai_role_cn": "超市收银员",
    "user_role": "third-grade primary school student",
    "user_role_cn": "小学三年级学生",
    "goal": "Successfully purchase stationery items",
    "goal_cn": "成功购买文具",
    "challenge": "Communicate clearly about items needed",
    "challenge_cn": "清楚表达需要的物品",
    "greeting": "Hello! What can I help you find today?",
    "difficulty_level": "easy",
    "speaking_speed": "slow",
    "vocabulary": "simple",
    "scenario_summary": "Student buying stationery",
    "scenario_summary_cn": "学生买文具"
  },
  "user_input": "小学三年级学生，去超市买文具"
}
```

**响应示例:**
```json
{
  "scenario_id": "custom_a1b2c3d4",
  "prompt_content": "**System Prompt: Custom Scenario...**",
  "greeting": "Hello! What can I help you find today?",
  "audio_url": "/api/audio/550e8400-e29b-41d4-a716-446655440000"
}
```

**响应字段说明:**

| 字段 | 类型 | 说明 |
|------|------|------|
| `scenario_id` | string | 生成的场景 ID（以 `custom_` 开头） |
| `prompt_content` | string | 生成的完整 prompt 内容 |
| `greeting` | string | 开场白文本 |
| `audio_url` | string | 开场白音频 URL（可选） |

---

### 使用自定义场景开始对话

生成场景后，使用 `/api/chat/start` 开始对话：

**请求体:**
```json
{
  "scenario": {
    "scenario": "custom_a1b2c3d4",
    "greeting": "Hello! What can I help you find today?"
  },
  "level": "A1",
  "turns": 9999
}
```

**注意:** 自定义场景的 `turns` 设置为 9999 表示无限对话，用户可以随时退出。

---

## 小程序端配置

### API 配置文件

**文件:** `utils/api.js`

```javascript
// API 基础地址
const BASE_URL = 'https://www.minirice.xyz';  // 生产环境
// const BASE_URL = 'http://localhost:8600';  // 本地开发
```

### 全局配置

**文件:** `app.js`

```javascript
App({
  globalData: {
    baseUrl: 'https://www.minirice.xyz',
    settings: {
      turns: 20,        // 对话轮数
      level: 'B1',      // 难度等级
      scenario: null    // 当前场景
    }
  }
});
```

### API 调用示例

```javascript
const { api } = require('../../utils/api');

// 开始对话
const session = await api.chat.start({
  scenario: { scenario: 'job_interview' },
  level: 'B1',
  turns: 20
});

// 发送消息
const response = await api.chat.message({
  session_id: session.session_id,
  message: 'Hello, I am here for the interview.'
});

// 语音转文字
const result = await api.speech.transcribe(audioFilePath, sessionId);

// 文字转语音
const audio = await api.speech.synthesize('Hello!', 'Ceylia', true);
```

---

## 可用场景

| 场景 ID | 名称 | 说明 |
|---------|------|------|
| `job_interview` | 求职面试 | 模拟技术面试场景 |
| `hotel_checkin` | 酒店入住 | 酒店前台对话练习 |
| `renting` | 租房咨询 | 租房看房对话练习 |
| `salary_negotiation` | 薪资谈判 | 薪资协商对话练习 |
| `custom_*` | 自定义场景 | 用户自定义的场景（ID 以 `custom_` 开头） |

---

## 难度等级

| 等级 | 说明 | 对应难度 |
|------|------|----------|
| A1 | 入门级 | primary |
| A2 | 基础级 | primary |
| B1 | 中级 | medium |
| B2 | 中高级 | medium |
| C1 | 高级 | advanced |
| C2 | 精通级 | advanced |

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在（如会话未找到） |
| 500 | 服务器内部错误 |

### 错误响应格式

```json
{
  "detail": "Session not found"
}
```

---

## 部署说明

### 生产环境

1. 配置 Nginx 反向代理，将域名指向端口 8600
2. 配置 SSL 证书（小程序要求 HTTPS）
3. 设置环境变量

**Nginx 配置示例:**

```nginx
server {
    listen 443 ssl;
    server_name www.minirice.xyz;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8600;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket 支持（如需要）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
}
```

### 微信小程序域名配置

在微信公众平台配置以下域名：

- **request 合法域名:** `https://www.minirice.xyz`
- **uploadFile 合法域名:** `https://www.minirice.xyz`
- **downloadFile 合法域名:** `https://www.minirice.xyz`

---

## 文件结构

```
LangCoach/
├── src/
│   └── api/
│       └── miniprogram_api.py    # 统一 API 服务
├── run_miniprogram_api.sh        # 启动脚本
└── MINIPROGRAM_API.md            # 本文档

LangCoach-MiniProgram/
├── app.js                        # 小程序入口
├── utils/
│   └── api.js                    # API 请求封装
└── pages/
    └── chat/
        └── chat.js               # 对话页面
```

---

## 更新日志

### v1.1.0 (2026-01-25)

- 新增自定义场景功能
  - `/api/custom-scenario/extract` - 从用户输入提取场景信息
  - `/api/custom-scenario/generate` - 生成自定义场景 prompt
- 自定义场景支持：
  - 智能角色提取（AI 角色、用户角色）
  - 自动难度调整（根据场景内容）
  - 无限对话轮数
  - 动态 prompt 生成

### v1.0.0 (2024-01-21)

- 整合 Chat API 和 Speech API 为统一服务
- 统一端口 8600
- 支持对话管理、语音识别、语音合成、词典查询
- 兼容旧版 Speech API 接口
