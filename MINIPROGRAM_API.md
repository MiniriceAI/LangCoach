# LangCoach Mini Program API 文档

## 概述

本文档描述 LangCoach 小程序后端 API 的架构设计和使用方法。API 服务整合了对话管理、语音识别、语音合成、词典查询等功能，通过单一端口（8600）对外提供服务。

## 架构设计

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    WeChat Mini Program                          │
│                   (LangCoach-MiniProgram)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Home Tab  │  │  Chat Tab   │  │ Profile Tab │             │
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
│  │  │ Chat API   │ │ Speech API │ │ Dict API   │            │  │
│  │  │ /api/chat/*│ │/api/trans* │ │/api/dict*  │            │  │
│  │  └─────┬──────┘ └─────┬──────┘ └────────────┘            │  │
│  │        │              │                                   │  │
│  │        ▼              ▼                                   │  │
│  │  ┌────────────┐ ┌────────────┐                           │  │
│  │  │  Session   │ │  TTS/STT   │                           │  │
│  │  │  Manager   │ │  Services  │                           │  │
│  │  └─────┬──────┘ └────────────┘                           │  │
│  └────────┼──────────────────────────────────────────────────┘  │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              LLM Agent Layer                              │  │
│  │  ┌────────────────┐  ┌────────────────┐                  │  │
│  │  │ ScenarioAgent  │  │  VocabAgent    │                  │  │
│  │  │ (job_interview │  │  (vocabulary)  │                  │  │
│  │  │  hotel_checkin │  │                │                  │  │
│  │  │  renting, etc) │  │                │                  │  │
│  │  └───────┬────────┘  └────────────────┘                  │  │
│  └──────────┼────────────────────────────────────────────────┘  │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    LLM Factory                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │  │
│  │  │  Ollama  │  │ DeepSeek │  │  OpenAI  │               │  │
│  │  │ (GLM-4)  │  │          │  │          │               │  │
│  │  └──────────┘  └──────────┘  └──────────┘               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 统一 API | `src/api/miniprogram_api.py` | 整合所有小程序接口 |
| 场景代理 | `src/agents/scenario_agent.py` | 处理场景对话逻辑 |
| TTS 服务 | `src/tts/service.py` | 文字转语音 |
| STT 服务 | `src/stt/service.py` | 语音转文字 |
| LLM 工厂 | `src/agents/llm_factory.py` | 管理多 LLM 提供者 |

---

## 快速开始

### 启动服务

```bash
cd /workspace/LangCoach

# 方式 1: 使用启动脚本（推荐）
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

### v1.0.0 (2024-01-21)

- 整合 Chat API 和 Speech API 为统一服务
- 统一端口 8600
- 支持对话管理、语音识别、语音合成、词典查询
- 兼容旧版 Speech API 接口
