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
import json
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

# Custom scenario prompts cache (临时自定义场景prompt存储)
_custom_scenario_prompts: Dict[str, str] = {}

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


def get_custom_scenario_context(scenario_id: str, difficulty: str) -> Optional[str]:
    """获取自定义场景的上下文"""
    if scenario_id in _custom_scenario_prompts:
        prompt_content = _custom_scenario_prompts[scenario_id]
        difficulty_instruction = f"\n\n**Additional Instructions:**\nUse {difficulty} level English. Keep responses short and conversational (1-2 sentences max). Focus on interactive dialogue, not long explanations."
        return prompt_content + difficulty_instruction
    return None


def get_scenario_context(scenario: str, difficulty: str) -> str:
    """获取场景上下文提示"""
    import os

    # 首先检查是否是自定义场景
    if scenario.startswith("custom_"):
        custom_context = get_custom_scenario_context(scenario, difficulty)
        if custom_context:
            return custom_context

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
    """获取LLM服务实例，根据LLM_PROVIDER配置选择提供商"""
    global _llm_service
    if _llm_service is None:
        try:
            import os
            
            # 读取LLM提供商配置
            llm_provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
            
            if llm_provider == "deepseek":
                # 从环境变量或.env文件获取API密钥
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if api_key:
                    from langchain_openai import ChatOpenAI
                    _llm_service = ChatOpenAI(
                        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                        openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                        openai_api_key=api_key,
                        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192")),
                        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.8")),
                    )
                    _service_status["llm"]["status"] = "loaded"
                    _service_status["llm"]["provider"] = "DeepSeek"
                    _service_status["llm"]["model"] = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
                    logger.info(f"LLM service loaded successfully - Provider: DeepSeek")
                else:
                    logger.error("DEEPSEEK_API_KEY not found in environment variables or .env file")
                    _service_status["llm"]["status"] = "error"
                    _service_status["llm"]["provider"] = "Error"
                    _service_status["llm"]["model"] = "API key missing"
            
            elif llm_provider == "openai":
                # 从环境变量或.env文件获取API密钥
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    from langchain_openai import ChatOpenAI
                    _llm_service = ChatOpenAI(
                        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        openai_api_key=api_key,
                        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "8192")),
                        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.8")),
                    )
                    _service_status["llm"]["status"] = "loaded"
                    _service_status["llm"]["provider"] = "OpenAI"
                    _service_status["llm"]["model"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    logger.info(f"LLM service loaded successfully - Provider: OpenAI")
                else:
                    logger.error("OPENAI_API_KEY not found in environment variables or .env file")
                    _service_status["llm"]["status"] = "error"
                    _service_status["llm"]["provider"] = "Error"
                    _service_status["llm"]["model"] = "API key missing"
            
            elif llm_provider == "ollama":
                import requests
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                try:
                    # 测试Ollama连接
                    response = requests.get(f"{ollama_url}/api/tags", timeout=int(os.getenv("OLLAMA_TIMEOUT", "5")))
                    if response.status_code == 200:
                        from langchain_ollama import ChatOllama
                        _llm_service = ChatOllama(
                            model=os.getenv("OLLAMA_MODEL", "hf.co/unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL"),
                            base_url=ollama_url,
                            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.8")),
                            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
                        )
                        _service_status["llm"]["status"] = "loaded"
                        _service_status["llm"]["provider"] = "Ollama"
                        _service_status["llm"]["model"] = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/GLM-4-9B-0414-GGUF:Q8_K_XL")
                        logger.info(f"LLM service loaded successfully - Provider: Ollama")
                    else:
                        logger.error("Ollama service unavailable")
                        _service_status["llm"]["status"] = "error"
                        _service_status["llm"]["provider"] = "Error"
                        _service_status["llm"]["model"] = "Service unavailable"
                except Exception as e:
                    logger.error(f"Failed to connect to Ollama: {e}")
                    _service_status["llm"]["status"] = "error"
                    _service_status["llm"]["provider"] = "Error"
                    _service_status["llm"]["model"] = "Connection failed"
            else:
                logger.error(f"Unsupported LLM provider: {llm_provider}")
                _service_status["llm"]["status"] = "error"
                _service_status["llm"]["provider"] = "Error"
                _service_status["llm"]["model"] = f"Unsupported provider: {llm_provider}"
            
        except Exception as e:
            logger.error(f"LLM service loading failed: {e}")
            _service_status["llm"]["status"] = "error"
            _service_status["llm"]["provider"] = "Error"
            _service_status["llm"]["model"] = "Unknown"
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
# Prompt Loading Functions
# ============================================================

def load_prompt_file(filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompts_dir = os.getenv("PROMPTS_DIR", "prompts")
    if not os.path.isabs(prompts_dir):
        # 相对路径，相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompts_dir = os.path.join(project_root, prompts_dir)
    
    prompt_path = os.path.join(prompts_dir, filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load prompt file {filename}: {e}")
        raise


def load_difficulty_instructions() -> Dict[str, str]:
    """Load difficulty instructions from JSON file."""
    import json
    prompts_dir = os.getenv("PROMPTS_DIR", "prompts")
    if not os.path.isabs(prompts_dir):
        # 相对路径，相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        prompts_dir = os.path.join(project_root, prompts_dir)
    
    instructions_path = os.path.join(prompts_dir, "difficulty_instructions.json")
    try:
        with open(instructions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load difficulty instructions: {e}")
        # Fallback to defaults
        return {
            "easy": "Use simple vocabulary and short sentences",
            "medium": "Use everyday vocabulary and moderate complexity",
            "hard": "Use advanced vocabulary and complex structures"
        }


# ============================================================
# Audio File Management
# ============================================================

async def generate_audio_url(text: str, speaker: str = None, fast_mode: bool = True, speaking_speed: str = "medium") -> Optional[str]:
    """生成文本对应的音频URL

    Args:
        text: 要合成的文本
        speaker: 说话人
        fast_mode: 是否使用快速模式(Edge-TTS)
        speaking_speed: 语速 (slow/medium/fast)
    """
    if speaker is None:
        speaker = config.service.tts_default_speaker
    try:
        # 生成音频文件名
        audio_id = str(uuid.uuid4())
        filename = f"{audio_id}.mp3"

        if fast_mode:
            # 使用 Edge-TTS
            audio_data = await _generate_edge_tts_audio_async(text, speaker, speaking_speed)
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


async def _generate_edge_tts_audio_async(text: str, speaker: str, speaking_speed: str = "medium") -> Optional[bytes]:
    """异步使用Edge-TTS生成音频数据

    Args:
        text: 要合成的文本
        speaker: 说话人
        speaking_speed: 语速 (slow/medium/fast)
    """
    try:
        import edge_tts

        voice = config.get_edge_tts_voice(speaker)

        # 将speaking_speed转换为Edge-TTS的rate参数
        # Edge-TTS rate: -50% to +100% (default 0%)
        rate_map = {
            "slow": "-30%",    # 慢速：减少30%
            "medium": "+0%",   # 正常速度
            "fast": "+30%"     # 快速：增加30%
        }
        rate = rate_map.get(speaking_speed, "+0%")

        logger.info(f"[TTS] Edge-TTS voice: {voice}, rate: {rate}, text: {text[:30]}...")

        communicate = edge_tts.Communicate(text, voice, rate=rate)
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


class ChatTips(BaseModel):
    """对话提示"""
    english: str = ""
    chinese: str = ""


class ChatStartResponse(BaseModel):
    """开始对话响应"""
    session_id: str
    greeting: str
    audio_url: Optional[str] = None
    scenario: str
    level: str
    max_turns: int
    chat_tips: Optional[ChatTips] = None  # 添加开场白的对话提示


class ChatMessageRequest(BaseModel):
    """发送消息请求"""
    session_id: str
    message: str


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


# --- Custom Scenario Models ---
class CustomScenarioExtractRequest(BaseModel):
    """自定义场景提取请求"""
    user_input: str  # 用户输入的场景描述，如"小学三年级学生，去超市买文具"


class CustomScenarioExtractResponse(BaseModel):
    """自定义场景提取响应"""
    ai_role: str  # AI扮演的角色
    ai_role_cn: str  # AI角色中文
    user_role: str  # 用户扮演的角色
    user_role_cn: str  # 用户角色中文
    goal: str  # 对话目标
    goal_cn: str  # 目标中文
    challenge: str  # 挑战
    challenge_cn: str  # 挑战中文
    greeting: str  # 随机开场白
    difficulty_level: str  # 难度级别: easy, medium, hard
    speaking_speed: str  # 语速: slow, medium, fast
    vocabulary: str  # 词汇难度: simple, medium, advanced
    scenario_summary: str  # 场景摘要
    scenario_summary_cn: str  # 场景摘要中文


class CustomScenarioGenerateRequest(BaseModel):
    """自定义场景生成prompt请求"""
    scenario_info: CustomScenarioExtractResponse
    user_input: str  # 原始用户输入


class CustomScenarioGenerateResponse(BaseModel):
    """自定义场景生成prompt响应"""
    scenario_id: str  # 临时场景ID
    prompt_content: str  # 生成的prompt内容
    greeting: str  # 开场白
    audio_url: Optional[str] = None  # 开场白音频URL


class RandomScenarioResponse(BaseModel):
    """随机场景生成响应（仅返回场景描述）"""
    scenario_description: str  # 场景描述，如"小学生去参观动物园"


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
    解析LLM响应，支持JSON格式和旧的文本格式

    JSON格式:
    {
        "direct_response": "直接回复内容",
        "hints_english": "英文提示",
        "hints_chinese": "中文提示"
    }

    旧文本格式:
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
    import json

    result = {
        "direct_response": response_text.strip(),
        "chat_tips": None
    }

    # 尝试解析JSON格式
    try:
        # 查找JSON对象
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group()
            parsed_json = json.loads(json_str)

            # 检查是否包含必需的字段
            if "direct_response" in parsed_json:
                result["direct_response"] = parsed_json["direct_response"].strip()

                # 提取提示信息
                if "hints_english" in parsed_json or "hints_chinese" in parsed_json:
                    result["chat_tips"] = {
                        "english": parsed_json.get("hints_english", "").strip(),
                        "chinese": parsed_json.get("hints_chinese", "").strip()
                    }

                logger.info(f"Successfully parsed JSON response")
                return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}, falling back to text parsing")
    except Exception as e:
        logger.warning(f"Error parsing JSON: {e}, falling back to text parsing")

    # 回退到旧的文本格式解析
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

    # 移除所有行中的 **LangCoach:** 前缀（不仅仅是开头）
    # 匹配行首的 **LangCoach:** 或 **LangCoach：** (支持中英文冒号)
    direct_response = re.sub(r'^\*\*LangCoach[：:]\*\*\s*', '', direct_response, flags=re.MULTILINE)

    # 移除可能的其他格式标记（如 **Teacher:** 等）
    direct_response = re.sub(r'^\*\*[A-Za-z]+[：:]\*\*\s*', '', direct_response, flags=re.MULTILINE)

    # 清理多余的空白和换行
    direct_response = direct_response.strip()
    result["direct_response"] = direct_response

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
        custom_greeting = None

        if request.scenario:
            scenario = request.scenario.get("scenario") or request.scenario.get("id") or "default"
            # 检查是否有自定义场景的开场白
            custom_greeting = request.scenario.get("greeting")

        # 验证场景 - 自定义场景以 custom_ 开头
        is_custom_scenario = scenario.startswith("custom_")
        if not is_custom_scenario and not config.is_scenario_available(scenario):
            logger.warning(f"Unknown scenario: {scenario}, using job_interview")
            scenario = "job_interview"

        # 创建会话 - 自定义场景不限制轮数
        max_turns = 9999 if is_custom_scenario else request.turns
        session = create_session(scenario, request.level, max_turns)

        # 获取开场白
        if custom_greeting:
            # 使用自定义场景的开场白
            greeting = custom_greeting
        elif is_custom_scenario:
            # 自定义场景但没有提供开场白
            greeting = "Hello! I'm ready to help you practice. What would you like to talk about?"
        else:
            greeting = config.content.default_greetings.get(scenario, config.content.default_greetings["default"])

        # 尝试使用 Agent 获取开场白（仅对非自定义场景）
        if not is_custom_scenario and scenario in config.content.available_scenarios:
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

        # 获取语速设置（从scenario配置中获取，如果有的话）
        speaking_speed = "medium"  # 默认值
        if request.scenario and "scenarioInfo" in request.scenario:
            speaking_speed = request.scenario["scenarioInfo"].get("speaking_speed", "medium")
        elif request.scenario and "speaking_speed" in request.scenario:
            speaking_speed = request.scenario.get("speaking_speed", "medium")

        # 保存语速到会话中，供后续消息使用
        session["speaking_speed"] = speaking_speed

        # 生成开场白音频URL（带语速控制）
        audio_url = await generate_audio_url(
            greeting,
            speaker=config.service.tts_default_speaker,
            fast_mode=True,
            speaking_speed=speaking_speed
        )

        # 生成开场白的对话提示
        chat_tips = await generate_startup_tips(greeting, scenario, session["difficulty"])

        return ChatStartResponse(
            session_id=session["id"],
            greeting=greeting,
            audio_url=audio_url,
            scenario=scenario,
            level=request.level,
            max_turns=request.turns,
            chat_tips=chat_tips
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

        # 生成音频URL (只对直接回复生成TTS，使用会话中保存的语速)
        speaking_speed = session.get("speaking_speed", "medium")
        audio_url = await generate_audio_url(
            reply,
            speaker=config.service.tts_default_speaker,
            fast_mode=True,
            speaking_speed=speaking_speed
        )

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
# Custom Scenario Endpoints
# ============================================================

# Load prompts from configuration files
_SCENARIO_EXTRACT_PROMPT = None
_SCENARIO_PROMPT_TEMPLATE = None
_DIFFICULTY_INSTRUCTIONS = None
_RANDOM_SCENARIO_GENERATOR_PROMPT = None


def get_scenario_extract_prompt() -> str:
    """Lazy load scenario extraction prompt."""
    global _SCENARIO_EXTRACT_PROMPT
    if _SCENARIO_EXTRACT_PROMPT is None:
        _SCENARIO_EXTRACT_PROMPT = load_prompt_file("custom_scenario_extract_prompt.txt")
    return _SCENARIO_EXTRACT_PROMPT


def get_scenario_prompt_template() -> str:
    """Lazy load scenario prompt template."""
    global _SCENARIO_PROMPT_TEMPLATE
    if _SCENARIO_PROMPT_TEMPLATE is None:
        _SCENARIO_PROMPT_TEMPLATE = load_prompt_file("custom_scenario_template.txt")
    return _SCENARIO_PROMPT_TEMPLATE


def get_difficulty_instructions() -> Dict[str, str]:
    """Lazy load difficulty instructions."""
    global _DIFFICULTY_INSTRUCTIONS
    if _DIFFICULTY_INSTRUCTIONS is None:
        _DIFFICULTY_INSTRUCTIONS = load_difficulty_instructions()
    return _DIFFICULTY_INSTRUCTIONS


def get_random_scenario_generator_prompt() -> str:
    """Lazy load random scenario generator prompt."""
    global _RANDOM_SCENARIO_GENERATOR_PROMPT
    if _RANDOM_SCENARIO_GENERATOR_PROMPT is None:
        _RANDOM_SCENARIO_GENERATOR_PROMPT = load_prompt_file("random_scenario_generator_prompt.txt")
    return _RANDOM_SCENARIO_GENERATOR_PROMPT


async def generate_startup_tips(greeting: str, scenario: str, difficulty: str) -> Optional[ChatTips]:
    """生成开场白的对话提示"""
    try:
        llm = get_llm_service()
        if not llm:
            return None

        # 加载提示生成prompt
        prompt_template = load_prompt_file("startup_tips_generator_prompt.txt")
        prompt = prompt_template.format(
            scenario=scenario,
            difficulty=difficulty,
            greeting=greeting
        )

        logger.info(f"Generating startup tips for greeting: {greeting[:50]}...")

        # 调用LLM
        response = llm.invoke(prompt)
        raw_response = response.content.strip()

        # 解析JSON响应
        import json
        import re

        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            json_str = json_match.group()
            tips_data = json.loads(json_str)

            return ChatTips(
                english=tips_data.get("hints_english", ""),
                chinese=tips_data.get("hints_chinese", "")
            )

        return None

    except Exception as e:
        logger.error(f"Error generating startup tips: {e}")
        return None
    return _DIFFICULTY_INSTRUCTIONS


@app.post("/api/custom-scenario/extract", response_model=CustomScenarioExtractResponse)
async def extract_custom_scenario(request: CustomScenarioExtractRequest):
    """从用户输入中提取自定义场景信息"""
    try:
        llm = get_llm_service()
        if not llm:
            raise HTTPException(status_code=503, detail="LLM service not available")

        # 构建提取prompt
        prompt_template = get_scenario_extract_prompt()
        prompt = prompt_template.format(user_input=request.user_input)

        logger.info(f"Extracting scenario info from: {request.user_input}")

        # 调用LLM
        response = llm.invoke(prompt)
        raw_response = response.content.strip()

        logger.info(f"LLM response: {raw_response[:500]}...")

        # 解析JSON响应
        import json
        import re

        # 尝试提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            json_str = json_match.group()
            scenario_info = json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response")

        # 确保所有字段都存在，提供默认值
        defaults = {
            "ai_role": "conversation partner",
            "ai_role_cn": "对话伙伴",
            "user_role": "English learner",
            "user_role_cn": "英语学习者",
            "goal": "Practice English conversation",
            "goal_cn": "练习英语对话",
            "challenge": "Maintain natural conversation flow",
            "challenge_cn": "保持自然的对话流程",
            "greeting": "Hello! How can I help you today?",
            "difficulty_level": "medium",
            "speaking_speed": "medium",
            "vocabulary": "medium",
            "scenario_summary": request.user_input,
            "scenario_summary_cn": request.user_input
        }

        for key, default_value in defaults.items():
            if key not in scenario_info or not scenario_info[key]:
                scenario_info[key] = default_value

        return CustomScenarioExtractResponse(**scenario_info)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        # 返回基于用户输入的默认响应
        return CustomScenarioExtractResponse(
            ai_role="conversation partner",
            ai_role_cn="对话伙伴",
            user_role="English learner",
            user_role_cn="英语学习者",
            goal="Practice English conversation based on: " + request.user_input,
            goal_cn="基于以下场景练习英语对话：" + request.user_input,
            challenge="Communicate effectively in this scenario",
            challenge_cn="在此场景中有效沟通",
            greeting="Hello! I'm ready to help you practice. What would you like to talk about?",
            difficulty_level="medium",
            speaking_speed="medium",
            vocabulary="medium",
            scenario_summary=request.user_input,
            scenario_summary_cn=request.user_input
        )
    except Exception as e:
        logger.error(f"Error extracting scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/custom-scenario/generate", response_model=CustomScenarioGenerateResponse)
async def generate_custom_scenario_prompt(request: CustomScenarioGenerateRequest):
    """生成自定义场景的prompt"""
    try:
        scenario_info = request.scenario_info

        # 生成唯一的场景ID
        scenario_id = f"custom_{uuid.uuid4().hex[:8]}"

        # 获取难度说明
        difficulty_instructions_map = get_difficulty_instructions()
        difficulty_instructions = difficulty_instructions_map.get(
            scenario_info.difficulty_level,
            difficulty_instructions_map["medium"]
        )

        # 生成prompt内容
        prompt_template = get_scenario_prompt_template()
        prompt_content = prompt_template.format(
            scenario_summary=scenario_info.scenario_summary,
            ai_role=scenario_info.ai_role,
            user_role=scenario_info.user_role,
            difficulty_level=scenario_info.difficulty_level,
            speaking_speed=scenario_info.speaking_speed,
            vocabulary=scenario_info.vocabulary,
            difficulty_instructions=difficulty_instructions
        )

        # 存储到临时缓存
        _custom_scenario_prompts[scenario_id] = prompt_content

        logger.info(f"Generated custom scenario prompt: {scenario_id}")

        # 生成开场白音频（带语速控制）
        audio_url = await generate_audio_url(
            scenario_info.greeting,
            speaker=config.service.tts_default_speaker,
            fast_mode=True,
            speaking_speed=scenario_info.speaking_speed
        )

        return CustomScenarioGenerateResponse(
            scenario_id=scenario_id,
            prompt_content=prompt_content,
            greeting=scenario_info.greeting,
            audio_url=audio_url
        )

    except Exception as e:
        logger.error(f"Error generating scenario prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/custom-scenario/random", response_model=RandomScenarioResponse)
async def generate_random_scenario():
    """使用LLM或预设场景生成随机场景描述

    根据配置 RANDOM_SCENARIO_MODE 决定使用哪种方式：
    - preset (默认): 从预设的100个场景中随机选择
    - llm: 使用LLM生成新的场景描述
    """
    try:
        # 检查配置的模式
        mode = config.content.random_scenario_mode.lower()

        if mode == "llm":
            # 使用LLM生成
            return await _generate_random_scenario_llm()
        else:
            # 使用预设场景（默认）
            return await _generate_random_scenario_preset()

    except Exception as e:
        logger.error(f"Error generating random scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_random_scenario_preset() -> RandomScenarioResponse:
    """从预设场景中随机选择一个"""
    import json
    import random

    try:
        # 加载预设场景文件
        preset_file = os.path.join(config.content.prompts_dir, "preset_scenarios.json")

        # 如果是相对路径，解析为绝对路径
        if not os.path.isabs(preset_file):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            preset_file = os.path.join(project_root, preset_file)

        with open(preset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取场景列表
        scenarios = data.get("scenarios", [])
        if not scenarios:
            raise ValueError("No scenarios found in preset file")

        # 随机选择一个场景
        selected_scenario = random.choice(scenarios)
        scenario_description = selected_scenario["scenario_description"]

        logger.info(f"Selected preset scenario: {scenario_description}")

        return RandomScenarioResponse(scenario_description=scenario_description)

    except FileNotFoundError:
        logger.error(f"Preset scenarios file not found: {preset_file}")
        # 返回默认场景
        return RandomScenarioResponse(scenario_description="在咖啡店点餐")
    except Exception as e:
        logger.error(f"Error loading preset scenarios: {e}")
        # 返回默认场景
        return RandomScenarioResponse(scenario_description="在咖啡店点餐")


async def _generate_random_scenario_llm() -> RandomScenarioResponse:
    """使用LLM生成随机场景描述"""
    try:
        llm = get_llm_service()
        if not llm:
            raise HTTPException(status_code=503, detail="LLM service not available")

        # 加载随机场景生成prompt
        prompt = get_random_scenario_generator_prompt()

        logger.info("Generating random scenario description using LLM...")

        # 调用LLM生成随机场景描述
        response = llm.invoke(prompt)
        raw_response = response.content.strip()

        logger.info(f"LLM random scenario response: {raw_response[:200]}...")

        # 解析JSON响应
        import json
        import re

        # 尝试提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            json_str = json_match.group()
            scenario_data = json.loads(json_str)

            scenario_description = scenario_data.get("scenario_description", "")

            if not scenario_description:
                raise ValueError("No scenario_description found in response")

            logger.info(f"Generated random scenario: {scenario_description}")

            return RandomScenarioResponse(scenario_description=scenario_description)
        else:
            raise ValueError("No valid JSON found in response")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        # 返回默认场景描述
        return RandomScenarioResponse(scenario_description="在咖啡店点餐")
    except Exception as e:
        logger.error(f"Error generating LLM scenario: {e}")
        raise


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
