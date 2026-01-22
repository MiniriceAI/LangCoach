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

# Import configuration
from src.api.config import config

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

# Audio file cache (临时音频文件存储)
_audio_cache: Dict[str, str] = {}

# Lazy loaded services
_tts_service = None
_stt_service = None
_llm_service = None
_agents: Dict[str, Any] = {}

# Service status tracking
_service_status = {
    "stt": {"status": "not_loaded", "model": config.service.stt_model},
    "tts": {"status": "loaded", "provider": "Edge-TTS"},
    "llm": {"status": "not_loaded", "model": config.service.ollama_model.split("/")[-1], "provider": "Ollama"}
}

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
        _service_status["stt"]["status"] = "loaded"
    return _stt_service





def get_scenario_context(scenario: str, difficulty: str) -> str:
    """获取场景上下文提示"""
    import os
    
    # 尝试读取prompt文件
    prompt_file = config.get_prompt_path(scenario)
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            
            # 在prompt中添加难度级别说明
            difficulty_instruction = f"\n\n**Additional Instructions:**\nUse {difficulty} level English. Keep responses short and conversational (1-2 sentences max). Focus on interactive dialogue, not long explanations."
            
            return prompt_content + difficulty_instruction
        except Exception as e:
            logger.error(f"Failed to load prompt file {prompt_file}: {e}")
    
    # 回退到默认context - 基于配置构建
    contexts = {}
    for scenario_id in config.content.available_scenarios + ["default"]:
        if scenario_id == "job_interview":
            contexts[scenario_id] = f"You are a professional job interviewer. Conduct a {difficulty} level interview. Keep responses short and interactive. Ask one question at a time."
        elif scenario_id == "hotel_checkin":
            contexts[scenario_id] = f"You are a hotel receptionist helping with check-in. Use {difficulty} level English. Keep responses brief and professional."
        elif scenario_id == "renting":
            contexts[scenario_id] = f"You are a property manager showing an apartment. Use {difficulty} level English. Keep responses short and conversational. Focus on one topic at a time."
        elif scenario_id == "salary_negotiation":
            contexts[scenario_id] = f"You are an HR manager discussing salary. Use {difficulty} level English. Keep responses brief and professional."
        else:
            contexts[scenario_id] = f"You are an English conversation partner. Use {difficulty} level English. Keep responses short and encouraging."
    return contexts.get(scenario, contexts["default"])


def get_llm_service():
    """获取LLM服务实例"""
    global _llm_service
    if _llm_service is None:
        try:
            import requests
            # 测试Ollama连接
            response = requests.get(config.get_ollama_url("api/tags"), timeout=config.service.ollama_timeout)
            if response.status_code == 200:
                from langchain_ollama import ChatOllama
                _llm_service = ChatOllama(
                    model=config.service.ollama_model,
                    base_url=config.service.ollama_base_url,
                    temperature=config.service.ollama_temperature,
                    num_predict=config.service.ollama_num_predict,
                    stop=config.service.ollama_stop_tokens
                )
                _service_status["llm"]["status"] = "loaded"
                logger.info("LLM service (Ollama + GLM-4-9B) loaded successfully")
            else:
                logger.error("Ollama服务不可用")
                _service_status["llm"]["status"] = "error"
        except Exception as e:
            logger.error(f"LLM service loading failed: {e}")
            _service_status["llm"]["status"] = "error"
    return _llm_service


def get_agent(scenario: str):
    """Lazy load scenario agent."""
    global _agents
    if scenario not in _agents:
        from src.agents.scenario_agent import ScenarioAgent
        _agents[scenario] = ScenarioAgent(scenario)
        logger.info(f"Loaded agent for scenario: {scenario}")
    return _agents[scenario]


# ============================================================
# Audio File Management
# ============================================================

async def generate_audio_url(text: str, speaker: str = None, fast_mode: bool = True) -> Optional[str]:
    """生成文本对应的音频URL"""
    if speaker is None:
        speaker = config.service.tts_default_speaker
    try:
        # 生成音频文件名
        audio_id = str(uuid.uuid4())
        filename = f"{audio_id}.mp3"
        
        if fast_mode:
            # 使用 Edge-TTS
            audio_data = await _generate_edge_tts_audio_async(text, speaker)
        else:
            # 使用本地 TTS
            audio_data = _generate_local_tts_audio(text, speaker)
        
        if audio_data:
            # 将音频数据保存到临时文件
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            # 存储到缓存
            _audio_cache[audio_id] = file_path
            
            # 返回访问URL
            return f"/api/audio/{audio_id}"
        
        return None
    except Exception as e:
        logger.error(f"Failed to generate audio URL: {e}")
        return None


async def _generate_edge_tts_audio_async(text: str, speaker: str) -> Optional[bytes]:
    """异步使用Edge-TTS生成音频数据"""
    try:
        import edge_tts
        
        voice = config.get_edge_tts_voice(speaker)
        logger.info(f"[TTS] Edge-TTS voice: {voice}, text: {text[:30]}...")
        
        communicate = edge_tts.Communicate(text, voice)
        buffer = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Edge-TTS generation failed: {e}")
        return None


def _generate_local_tts_audio(text: str, speaker: str) -> Optional[bytes]:
    """使用本地TTS生成音频数据"""
    try:
        import soundfile as sf
        
        service = get_tts_service()
        result = service.synthesize(text=text, speaker=speaker)
        
        buffer = io.BytesIO()
        sf.write(buffer, result["audio"], result["sample_rate"], format="MP3")
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Local TTS generation failed: {e}")
        return None


# ============================================================
# Constants (moved to config.py)
# ============================================================
# All constants are now managed in src.api.config

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
    audio_url: Optional[str] = None
    scenario: str
    level: str
    max_turns: int


class ChatMessageRequest(BaseModel):
    """发送消息请求"""
    session_id: str
    message: str


class ChatTips(BaseModel):
    """对话提示"""
    english: str = ""
    chinese: str = ""


class ChatMessageResponse(BaseModel):
    """消息响应"""
    reply: str
    audio_url: Optional[str] = None
    feedback: Optional[str] = None
    session_ended: bool = False
    current_turn: int = 0
    report: Optional[Dict[str, Any]] = None
    chat_tips: Optional[ChatTips] = None


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
    speaker: str = None  # Will use config default if None
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
        "difficulty": config.get_difficulty_for_level(level),
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


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    解析LLM响应，分离直接回复和对话提示

    LLM响应格式:
    [直接回复内容 - 1-2句话]

    **对话提示:**
    [英文示例]
    [中文示例]

    返回:
    {
        "direct_response": "直接回复内容",
        "chat_tips": {
            "english": "英文示例",
            "chinese": "中文示例"
        }
    }
    """
    import re

    result = {
        "direct_response": response_text.strip(),
        "chat_tips": None
    }

    # 分离对话提示部分 - 查找 **对话提示:** 标记
    tips_pattern = r'\*\*对话提示[：:]\*\*\s*\n(.+?)$'
    tips_match = re.search(tips_pattern, response_text, re.DOTALL | re.IGNORECASE)

    if tips_match:
        # 获取对话提示内容
        tips_content = tips_match.group(1).strip()

        # 移除对话提示部分，获取直接回复
        direct_response = response_text[:tips_match.start()].strip()

        # 解析对话提示内容（英文和中文）
        tips_lines = [line.strip() for line in tips_content.split('\n') if line.strip()]

        english_tip = ""
        chinese_tip = ""

        # 第一行通常是英文，第二行是中文
        if len(tips_lines) >= 1:
            english_tip = tips_lines[0]
        if len(tips_lines) >= 2:
            chinese_tip = tips_lines[1]

        # 如果没有找到中文，尝试从所有行中检测
        if not chinese_tip:
            for line in tips_lines:
                if re.search(r'[\u4e00-\u9fff]', line):
                    chinese_tip = line
                    break

        if english_tip or chinese_tip:
            result["chat_tips"] = {
                "english": english_tip,
                "chinese": chinese_tip
            }

        result["direct_response"] = direct_response

    # 清理直接回复中的任何格式标记
    direct_response = result["direct_response"]

    # 移除 **LangCoach:** 前缀（如果有）
    direct_response = re.sub(r'^\*\*LangCoach[：:]\*\*\s*', '', direct_response)

    # 移除其他可能的格式标记
    direct_response = re.sub(r'^\*\*[^*]+\*\*\s*', '', direct_response)

    # 清理多余的空白
    result["direct_response"] = direct_response.strip()

    return result


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

        # 3. 预加载 LLM 服务 (Ollama + GLM-4-9B)
        logger.info("[3/3] Loading LLM service (Ollama + GLM-4-9B)...")
        try:
            get_llm_service()
            if _service_status["llm"]["status"] == "loaded":
                logger.info("[3/3] LLM service loaded successfully")
            else:
                logger.error("[3/3] LLM service failed to load")
        except Exception as e:
            logger.error(f"[3/3] Failed to load LLM service: {e}")

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
    allow_origins=config.service.cors_origins,
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
    return {
        "status": "healthy",
        "service": "langcoach-miniprogram-api",
        "timestamp": datetime.now().isoformat(),
        "sessions_count": len(_sessions),
        "services": {
            "stt": _service_status["stt"],
            "tts": _service_status["tts"],
            "llm": _service_status["llm"]
        }
    }


@app.get("/api/scenarios")
async def list_scenarios():
    """获取可用场景列表"""
    scenarios = []
    for scenario_id in config.content.available_scenarios:
        scenarios.append({
            "id": scenario_id,
            "title": scenario_id.replace("_", " ").title(),
            "greeting": config.content.default_greetings.get(scenario_id, config.content.default_greetings["default"]),
        })
    return {"scenarios": scenarios}


@app.get("/api/speakers")
async def list_speakers():
    """获取可用的 TTS 语音角色"""
    return {"speakers": list(config.service.edge_tts_voices.keys())}


@app.get("/api/audio/{audio_id}")
async def get_audio_file(audio_id: str):
    """获取音频文件"""
    try:
        if audio_id not in _audio_cache:
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        file_path = _audio_cache[audio_id]
        
        if not os.path.exists(file_path):
            # 清理失效的缓存条目
            del _audio_cache[audio_id]
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # 返回音频文件流
        def iterfile(file_path: str):
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        
        return StreamingResponse(
            iterfile(file_path),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename={audio_id}.mp3",
                "Cache-Control": f"max-age={config.service.audio_cache_hours * 3600}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving audio file {audio_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


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
        if not config.is_scenario_available(scenario):
            logger.warning(f"Unknown scenario: {scenario}, using job_interview")
            scenario = "job_interview"

        # 创建会话
        session = create_session(scenario, request.level, request.turns)

        # 获取开场白
        greeting = config.content.default_greetings.get(scenario, config.content.default_greetings["default"])

        # 尝试使用 Agent 获取开场白
        if scenario in config.content.available_scenarios:
            try:
                agent = get_agent(scenario)
                from src.agents.conversation_config import create_config
                agent_config = create_config(
                    turns=request.turns,
                    difficulty=config.get_difficulty_for_level(request.level)
                )
                greeting = agent.start_new_session(session_id=session["id"], config=agent_config)
            except Exception as e:
                logger.error(f"Agent init failed: {e}, using default greeting")

        # 保存开场白到会话
        session["messages"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": datetime.now().isoformat()
        })

        # 生成开场白音频URL
        audio_url = await generate_audio_url(greeting, speaker=config.service.tts_default_speaker, fast_mode=True)

        return ChatStartResponse(
            session_id=session["id"],
            greeting=greeting,
            audio_url=audio_url,
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
        chat_tips = None

        # 使用 LLM 生成回复
        try:
            llm = get_llm_service()
            if llm:
                # 构建完整的对话上下文
                context = get_scenario_context(scenario, session["difficulty"])

                # 构建对话历史
                conversation_history = ""
                recent_messages = session["messages"][-config.service.max_recent_messages:]  # 使用配置的数量

                for msg in recent_messages:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    conversation_history += f"{role}: {msg['content']}\n"

                # 构建完整prompt
                full_prompt = f"""{context}

**Conversation History:**
{conversation_history}
Human: {request.message}
Assistant:"""

                logger.info(f"Sending prompt to LLM (length: {len(full_prompt)} chars)")

                # 调用LLM
                response = llm.invoke(full_prompt)
                raw_reply = response.content.strip()

                # 解析LLM响应，分离直接回复和对话提示
                parsed = parse_llm_response(raw_reply)
                reply = parsed["direct_response"]

                if parsed["chat_tips"]:
                    chat_tips = ChatTips(
                        english=parsed["chat_tips"]["english"],
                        chinese=parsed["chat_tips"]["chinese"]
                    )

                # 限制回复长度
                if len(reply) > config.service.max_reply_length:
                    # 截取到第一个完整句子结束
                    sentences = reply.split('. ')
                    reply = sentences[0]
                    if not reply.endswith('.'):
                        reply += '.'
                    if len(reply) > config.service.max_reply_sentences:
                        reply = reply[:config.service.max_reply_sentences] + "..."

                logger.info(f"LLM generated reply: {reply[:100]}...")
                if chat_tips:
                    logger.info(f"Chat tips: EN={chat_tips.english[:50]}..., CN={chat_tips.chinese[:50]}...")
            else:
                logger.warning("LLM service not available, using fallback reply")
        except Exception as e:
            logger.error(f"LLM inference error: {e}")

        # 更新轮数
        session["current_turn"] += 1

        # 保存 AI 回复到 chat history（只保存直接回复，不包含对话提示）
        session["messages"].append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat()
        })

        # 生成音频URL (只对直接回复生成TTS)
        audio_url = await generate_audio_url(reply, speaker=config.service.tts_default_speaker, fast_mode=True)

        # 检查是否结束
        session_ended = session["current_turn"] >= session["max_turns"]
        report = None

        if session_ended:
            session["ended"] = True
            scores = config.generate_random_scores()
            tips = config.get_random_tips(3)
            report = {
                "grammarScore": scores["grammarScore"],
                "vocabularyScore": scores["vocabularyScore"],
                "fluencyScore": scores["fluencyScore"],
                "totalTurns": session["current_turn"],
                "tips": tips
            }

        return ChatMessageResponse(
            reply=reply,
            audio_url=audio_url,
            feedback=None,
            session_ended=session_ended,
            current_turn=session["current_turn"],
            report=report,
            chat_tips=chat_tips
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
            audio_data, sr = librosa.load(tmp_path, sr=config.service.stt_sample_rate)
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
        speaker = request.speaker or config.service.tts_default_speaker
        if request.fast_mode:
            # 使用 Edge-TTS 快速模式
            return await _synthesize_edge_tts(request.text, speaker)
        else:
            # 使用本地 TTS 模型
            return await _synthesize_local(request.text, speaker)

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _synthesize_edge_tts(text: str, speaker: str) -> JSONResponse:
    """使用 Edge-TTS 合成语音"""
    try:
        import edge_tts

        voice = config.get_edge_tts_voice(speaker)
        logger.info(f"[TTS] Edge-TTS voice: {voice}, text: {text[:30]}...")

        communicate = edge_tts.Communicate(text, voice)
        buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

        audio_bytes = buffer.getvalue()

        return JSONResponse({
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "sample_rate": config.service.tts_sample_rate,
            "speaker": speaker,
            "text": text,
            "format": config.service.tts_format
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

@app.get("/api/dictionary", response_model=DictionaryResponse)
async def dictionary_lookup(word: str):
    """查询单词释义"""
    word_lower = word.lower().strip()

    if word_lower in config.content.simple_dictionary:
        entry = config.content.simple_dictionary[word_lower]
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
            voice = config.get_edge_tts_voice(request.speaker or config.service.tts_default_speaker)
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

    logger.info(f"Starting API on {config.service.api_host}:{config.service.api_port}")

    uvicorn.run(
        "src.api.miniprogram_api:app",
        host=config.service.api_host,
        port=config.service.api_port,
        reload=False
    )


if __name__ == "__main__":
    main()
