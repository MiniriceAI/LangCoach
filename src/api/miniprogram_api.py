#!/usr/bin/env python3
"""
LangCoach Mini Program Unified API

统一的 FastAPI 服务，整合所有小程序需要的接口：
- 对话管理 (Chat)
- 语音识别 (STT)
- 语音合成 (TTS)
- 词典查询 (Dictionary)
- 用户认证 (Auth)

Usage:
    ./run_miniprogram_api.sh
    # 或
    uvicorn src.api.miniprogram_api:app --host 0.0.0.0 --port 8600
"""

import io
import os
import uuid
import logging
import tempfile
import base64
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Global State
# ============================================================

# Session storage (in production, use Redis)
_sessions: Dict[str, Dict[str, Any]] = {}

# Lazy loaded services
_tts_service = None
_stt_service = None
_agents: Dict[str, Any] = {}

# ============================================================
# Service Loaders
# ============================================================

def get_tts_service():
    """Lazy load TTS service."""
    global _tts_service
    if _tts_service is None:
        from src.tts.service import initialize_tts_service
        _tts_service = initialize_tts_service()
    return _tts_service


def get_stt_service():
    """Lazy load STT service."""
    global _stt_service
    if _stt_service is None:
        from src.stt.service import initialize_stt_service
        _stt_service = initialize_stt_service()
    return _stt_service


def get_agent(scenario: str):
    """Lazy load scenario agent."""
    global _agents
    if scenario not in _agents:
        from src.agents.scenario_agent import ScenarioAgent
        _agents[scenario] = ScenarioAgent(scenario)
        logger.info(f"Loaded agent for scenario: {scenario}")
    return _agents[scenario]


# ============================================================
# Constants
# ============================================================

AVAILABLE_SCENARIOS = ["job_interview", "hotel_checkin", "renting", "salary_negotiation"]

DEFAULT_GREETINGS = {
    "job_interview": "Hello! I'm your interviewer today. Please have a seat and let's begin. Could you start by telling me a little about yourself?",
    "hotel_checkin": "Good evening! Welcome to our hotel. I'll be helping you with check-in today. May I have your name and reservation details, please?",
    "renting": "Hi there! I'm the property manager. I understand you're interested in renting this apartment. Would you like me to show you around first?",
    "salary_negotiation": "Thank you for coming in today. We've reviewed your application and would like to discuss the compensation package. What are your salary expectations?",
    "default": "Hi there! I'm your English practice partner. What would you like to talk about today?"
}

LEVEL_TO_DIFFICULTY = {
    "A1": "primary", "A2": "primary",
    "B1": "medium", "B2": "medium",
    "C1": "advanced", "C2": "advanced",
}

EDGE_TTS_VOICES = {
    "Ceylia": "en-US-JennyNeural",
    "Tifa": "en-US-AriaNeural",
    "default": "en-US-JennyNeural"
}

# ============================================================
# Pydantic Models
# ============================================================

# --- Chat Models ---
class ChatStartRequest(BaseModel):
    """开始对话请求"""
    scenario: Optional[Dict[str, Any]] = None
    level: str = "B1"
    turns: int = 20


class ChatStartResponse(BaseModel):
    """开始对话响应"""
    session_id: str
    greeting: str
    scenario: str
    level: str
    max_turns: int


class ChatMessageRequest(BaseModel):
    """发送消息请求"""
    session_id: str
    message: str


class ChatMessageResponse(BaseModel):
    """消息响应"""
    reply: str
    feedback: Optional[str] = None
    session_ended: bool = False
    current_turn: int = 0
    report: Optional[Dict[str, Any]] = None


class ChatRateRequest(BaseModel):
    """评分请求"""
    session_id: str
    rating: int
    feedback: Optional[str] = None


class ChatFeedbackRequest(BaseModel):
    """消息反馈请求"""
    session_id: str
    message_id: str
    feedback: str


# --- Speech Models ---
class SynthesizeRequest(BaseModel):
    """语音合成请求"""
    text: str
    speaker: str = "Ceylia"
    fast_mode: bool = True


class TranscribeResponse(BaseModel):
    """语音识别响应"""
    text: str
    language: str = "en"


# --- Dictionary Models ---
class DictionaryResponse(BaseModel):
    """词典查询响应"""
    word: str
    phonetic: Optional[str] = None
    definition: Optional[str] = None


# ============================================================
# Session Management
# ============================================================

def create_session(scenario: str, level: str, turns: int) -> Dict[str, Any]:
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "scenario": scenario,
        "level": level,
        "difficulty": LEVEL_TO_DIFFICULTY.get(level, "medium"),
        "max_turns": turns,
        "current_turn": 0,
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "ended": False,
        "rating": None,
    }
    _sessions[session_id] = session
    logger.info(f"Created session: {session_id} for scenario: {scenario}")
    return session


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session by ID."""
    return _sessions.get(session_id)


# ============================================================
# FastAPI App
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("=" * 50)
    logger.info("Starting LangCoach Mini Program API...")
    logger.info("=" * 50)

    preload = os.getenv("PRELOAD_MODELS", "false").lower() == "true"
    if preload:
        logger.info("Pre-loading services...")

        # 1. 预加载 STT 服务 (Whisper-large-v3 + 4bit)
        logger.info("[1/3] Loading STT service (Whisper-large-v3)...")
        try:
            get_stt_service()
            logger.info("[1/3] STT service loaded successfully")
        except Exception as e:
            logger.error(f"[1/3] Failed to load STT: {e}")

        # 2. TTS 使用 Edge-TTS 快速模式，无需预加载本地模型
        # 如果需要使用本地 Orpheus TTS，取消下面的注释
        # logger.info("[2/3] Loading TTS service (Orpheus)...")
        # try:
        #     get_tts_service()
        #     logger.info("[2/3] TTS service loaded successfully")
        # except Exception as e:
        #     logger.error(f"[2/3] Failed to load TTS: {e}")
        logger.info("[2/3] TTS: Using Edge-TTS (no preload needed)")

        # 3. 预加载 LLM Agent (Ollama + GLM-4-9B)
        logger.info("[3/3] Loading LLM Agents (Ollama + GLM-4-9B)...")
        try:
            # 预加载第一个场景的 Agent，这会初始化 LLM 连接
            get_agent(AVAILABLE_SCENARIOS[0])
            logger.info("[3/3] LLM Agent loaded successfully")
        except Exception as e:
            logger.error(f"[3/3] Failed to load LLM Agent: {e}")

        logger.info("=" * 50)
        logger.info("All services loaded. API is ready!")
        logger.info("=" * 50)
    else:
        logger.info("Preload disabled. Services will load on first request.")

    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="LangCoach Mini Program API",
    description="统一的小程序后端 API，提供对话、语音、词典等功能",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Health & Info Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """健康检查"""
    global _tts_service, _stt_service, _agents
    return {
        "status": "healthy",
        "service": "langcoach-miniprogram-api",
        "timestamp": datetime.now().isoformat(),
        "sessions_count": len(_sessions),
        "services": {
            "stt": {
                "status": "loaded" if (_stt_service is not None and _stt_service.is_initialized) else "not_loaded",
                "model": "unsloth/whisper-large-v3"
            },
            "tts": {
                "status": "ready",
                "mode": "edge-tts",
                "note": "Edge-TTS (Microsoft Azure) - no preload needed"
            },
            "llm": {
                "status": "loaded" if len(_agents) > 0 else "not_loaded",
                "agents_loaded": list(_agents.keys()),
                "model": "hf.co/unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL"
            }
        }
    }


@app.get("/api/scenarios")
async def list_scenarios():
    """获取可用场景列表"""
    scenarios = []
    for scenario_id in AVAILABLE_SCENARIOS:
        scenarios.append({
            "id": scenario_id,
            "title": scenario_id.replace("_", " ").title(),
            "greeting": DEFAULT_GREETINGS.get(scenario_id, DEFAULT_GREETINGS["default"]),
        })
    return {"scenarios": scenarios}


@app.get("/api/speakers")
async def list_speakers():
    """获取可用的 TTS 语音角色"""
    return {"speakers": list(EDGE_TTS_VOICES.keys())}


# ============================================================
# Chat Endpoints
# ============================================================

@app.post("/api/chat/start", response_model=ChatStartResponse)
async def chat_start(request: ChatStartRequest):
    """开始新的对话会话"""
    try:
        # 确定场景
        scenario = "default"
        if request.scenario:
            scenario = request.scenario.get("scenario") or request.scenario.get("id") or "default"

        # 验证场景
        if scenario not in AVAILABLE_SCENARIOS and scenario != "default":
            logger.warning(f"Unknown scenario: {scenario}, using job_interview")
            scenario = "job_interview"

        # 创建会话
        session = create_session(scenario, request.level, request.turns)

        # 获取开场白
        greeting = DEFAULT_GREETINGS.get(scenario, DEFAULT_GREETINGS["default"])

        # 尝试使用 Agent 获取开场白
        if scenario in AVAILABLE_SCENARIOS:
            try:
                agent = get_agent(scenario)
                from src.agents.conversation_config import create_config
                config = create_config(
                    turns=request.turns,
                    difficulty=LEVEL_TO_DIFFICULTY.get(request.level, "medium")
                )
                greeting = agent.start_new_session(session_id=session["id"], config=config)
            except Exception as e:
                logger.error(f"Agent init failed: {e}, using default greeting")

        # 保存开场白到会话
        session["messages"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": datetime.now().isoformat()
        })

        return ChatStartResponse(
            session_id=session["id"],
            greeting=greeting,
            scenario=scenario,
            level=request.level,
            max_turns=request.turns
        )

    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/message", response_model=ChatMessageResponse)
async def chat_message(request: ChatMessageRequest):
    """发送消息并获取 AI 回复"""
    try:
        session = get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session["ended"]:
            raise HTTPException(status_code=400, detail="Session has ended")

        # 保存用户消息
        session["messages"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })

        # 获取 AI 回复
        scenario = session["scenario"]
        reply = "I understand. Could you tell me more about that?"

        if scenario in AVAILABLE_SCENARIOS:
            try:
                agent = get_agent(scenario)
                reply = agent.chat_with_history(request.message, session_id=request.session_id)
            except Exception as e:
                logger.error(f"Agent error: {e}")

        # 更新轮数
        session["current_turn"] += 1

        # 保存 AI 回复
        session["messages"].append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat()
        })

        # 检查是否结束
        session_ended = session["current_turn"] >= session["max_turns"]
        report = None

        if session_ended:
            session["ended"] = True
            report = {
                "grammarScore": 85,
                "vocabularyScore": 78,
                "fluencyScore": 82,
                "totalTurns": session["current_turn"],
                "tips": [
                    "Try using more complex sentence structures",
                    "Good use of vocabulary!",
                    "Practice speaking more fluently"
                ]
            }

        return ChatMessageResponse(
            reply=reply,
            feedback=None,
            session_ended=session_ended,
            current_turn=session["current_turn"],
            report=report
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/rate")
async def chat_rate(request: ChatRateRequest):
    """评价会话"""
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["rating"] = request.rating
    session["user_feedback"] = request.feedback
    logger.info(f"Session {request.session_id} rated: {request.rating}")

    return {"success": True, "message": "Rating submitted"}


@app.post("/api/chat/feedback")
async def chat_feedback(request: ChatFeedbackRequest):
    """消息反馈（点赞/踩）"""
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Feedback: session={request.session_id}, msg={request.message_id}, type={request.feedback}")
    return {"success": True, "message": "Feedback submitted"}


# ============================================================
# Speech Endpoints
# ============================================================

@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None)
):
    """语音转文字"""
    import librosa

    try:
        service = get_stt_service()

        # 保存上传的文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # 加载并转录
            audio_data, sr = librosa.load(tmp_path, sr=16000)
            result = service.transcribe(
                audio=audio_data,
                sample_rate=sr,
                language=language,
                task="transcribe"
            )

            logger.info(f"Transcribed: {result['text'][:50]}...")
            return TranscribeResponse(
                text=result["text"],
                language=result.get("language", "en")
            )
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/synthesize")
async def synthesize(request: SynthesizeRequest):
    """文字转语音（返回 base64 编码的音频）"""
    try:
        if request.fast_mode:
            # 使用 Edge-TTS 快速模式
            return await _synthesize_edge_tts(request.text, request.speaker)
        else:
            # 使用本地 TTS 模型
            return await _synthesize_local(request.text, request.speaker)

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _synthesize_edge_tts(text: str, speaker: str) -> JSONResponse:
    """使用 Edge-TTS 合成语音"""
    try:
        import edge_tts

        voice = EDGE_TTS_VOICES.get(speaker, EDGE_TTS_VOICES["default"])
        logger.info(f"[TTS] Edge-TTS voice: {voice}, text: {text[:30]}...")

        communicate = edge_tts.Communicate(text, voice)
        buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

        audio_bytes = buffer.getvalue()

        return JSONResponse({
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "sample_rate": 24000,
            "speaker": speaker,
            "text": text,
            "format": "mp3"
        })

    except ImportError:
        raise HTTPException(status_code=500, detail="edge-tts not installed")


async def _synthesize_local(text: str, speaker: str) -> JSONResponse:
    """使用本地 TTS 模型合成语音"""
    import soundfile as sf

    service = get_tts_service()
    result = service.synthesize(text=text, speaker=speaker)

    buffer = io.BytesIO()
    sf.write(buffer, result["audio"], result["sample_rate"], format="WAV")
    audio_bytes = buffer.getvalue()

    return JSONResponse({
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "sample_rate": result["sample_rate"],
        "speaker": result["speaker"],
        "text": result["text"],
        "format": "wav"
    })


# ============================================================
# Dictionary Endpoint
# ============================================================

# 简单词典（生产环境应使用真实词典 API）
SIMPLE_DICTIONARY = {
    "hello": {"phonetic": "/həˈloʊ/", "definition": "used as a greeting"},
    "interview": {"phonetic": "/ˈɪntərˌvjuː/", "definition": "a formal meeting for assessment"},
    "salary": {"phonetic": "/ˈsæləri/", "definition": "fixed regular payment for work"},
    "experience": {"phonetic": "/ɪkˈspɪriəns/", "definition": "practical contact with events"},
    "hotel": {"phonetic": "/hoʊˈtel/", "definition": "an establishment providing lodging"},
    "apartment": {"phonetic": "/əˈpɑːrtmənt/", "definition": "a self-contained housing unit"},
    "rent": {"phonetic": "/rent/", "definition": "payment for use of property"},
    "negotiate": {"phonetic": "/nɪˈɡoʊʃieɪt/", "definition": "to discuss to reach an agreement"},
}


@app.get("/api/dictionary", response_model=DictionaryResponse)
async def dictionary_lookup(word: str):
    """查询单词释义"""
    word_lower = word.lower().strip()

    if word_lower in SIMPLE_DICTIONARY:
        entry = SIMPLE_DICTIONARY[word_lower]
        return DictionaryResponse(
            word=word,
            phonetic=entry["phonetic"],
            definition=entry["definition"]
        )

    return DictionaryResponse(
        word=word,
        phonetic=None,
        definition="Definition not available. Try an online dictionary."
    )


# ============================================================
# Auth Endpoint (Placeholder)
# ============================================================

@app.post("/api/auth/wechat")
async def wechat_auth(code: str = Form(...)):
    """微信登录（占位实现）"""
    # 生产环境应验证 code 并调用微信 API
    token = str(uuid.uuid4())
    logger.info(f"WeChat auth: code={code[:10]}..., token={token[:10]}...")

    return {
        "token": token,
        "user_id": f"user_{token[:8]}",
        "expires_in": 7200
    }


# ============================================================
# Legacy Endpoints (兼容旧接口)
# ============================================================

@app.get("/speakers")
async def legacy_speakers():
    """兼容旧的 speakers 接口"""
    return await list_speakers()


@app.post("/transcribe", response_model=TranscribeResponse)
async def legacy_transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: Optional[str] = Form("transcribe")
):
    """兼容旧的 transcribe 接口"""
    return await transcribe(audio=audio, language=language)


@app.post("/synthesize")
async def legacy_synthesize(request: SynthesizeRequest):
    """兼容旧的 synthesize 接口（返回流）"""
    try:
        if request.fast_mode:
            import edge_tts
            voice = EDGE_TTS_VOICES.get(request.speaker, EDGE_TTS_VOICES["default"])
            communicate = edge_tts.Communicate(request.text, voice)
            buffer = io.BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])

            buffer.seek(0)
            return StreamingResponse(
                buffer,
                media_type="audio/mpeg",
                headers={"Content-Disposition": f'attachment; filename="speech.mp3"'}
            )
        else:
            import soundfile as sf
            service = get_tts_service()
            result = service.synthesize(text=request.text, speaker=request.speaker)

            buffer = io.BytesIO()
            sf.write(buffer, result["audio"], result["sample_rate"], format="WAV")
            buffer.seek(0)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": f'attachment; filename="speech.wav"'}
            )

    except Exception as e:
        logger.error(f"Legacy synthesize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/json")
async def legacy_synthesize_json(request: SynthesizeRequest):
    """兼容旧的 synthesize/json 接口"""
    return await synthesize(request)


# ============================================================
# Main Entry
# ============================================================

def main():
    """Run the API server."""
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8600"))

    logger.info(f"Starting API on {host}:{port}")

    uvicorn.run(
        "src.api.miniprogram_api:app",
        host=host,
        port=port,
        reload=False
    )


if __name__ == "__main__":
    main()
