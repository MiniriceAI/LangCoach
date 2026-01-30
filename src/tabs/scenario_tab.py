# tabs/scenario_tab.py

import gradio as gr
import numpy as np
import io
import base64
import os
import requests
from typing import Optional, Tuple
from agents.scenario_agent import ScenarioAgent
from agents.conversation_config import (
    ConversationConfig,
    DifficultyLevel,
    TurnOption,
    create_config,
)
from utils.logger import LOG

# Speech API 配置（通过 HTTP 调用独立的 Speech API 服务）
SPEECH_API_URL = os.getenv("SPEECH_API_URL", "http://localhost:8600")

# 支持的 Speaker 列表
SPEAKER_CHOICES = [
    ("Ceylia", "Ceylia"),
    ("Tifa", "Tifa"),
]


def extract_english_response(bot_message: str) -> str:
    """
    从 AI 回复中提取纯英文部分（排除对话提示）
    
    AI 回复格式通常是:
    - 英文回复
    - 对话提示:
    - 英文提示
    - 中文翻译
    
    我们只需要第一部分的英文回复用于 TTS
    """
    if not bot_message:
        return ""
    
    import re
    
    # 第一步：移除 "LangCoach:" 或 "**LangCoach:**" 前缀
    text = bot_message.strip()
    prefix_patterns = [
        r'^\*\*LangCoach:\*\*\s*',
        r'^LangCoach:\s*',
        r'^\*\*LangCoach：\*\*\s*',
        r'^LangCoach：\s*',
    ]
    for prefix in prefix_patterns:
        text = re.sub(prefix, '', text)
    
    # 第二步：按照"对话提示"或"Dialogue hint"分割，只取之前的部分
    separators = [
        r'\n\n\*\*对话提示[：:]\*\*',
        r'\n\n对话提示[：:]',
        r'\n对话提示[：:]',
        r'\n\nDialogue [Hh]int[：:]',
        r'\n\n\*\*对话提示',
    ]
    
    for sep in separators:
        parts = re.split(sep, text, maxsplit=1)
        if len(parts) > 1:
            english_part = parts[0].strip()
            if english_part:
                return english_part
    
    # 如果没有找到分隔符，返回整个消息的第一段（到双换行符为止）
    paragraphs = text.split('\n\n')
    if paragraphs:
        return paragraphs[0].strip()
    
    return text.strip()


def synthesize_speech(text: str, speaker: str, fast_mode: bool = True) -> Optional[Tuple[int, np.ndarray]]:
    """
    通过 Speech API 合成语音

    Args:
        text: 要合成的文本
        speaker: 说话人（Ceylia 或 Tifa）
        fast_mode: 使用Edge-TTS快速模式（默认开启）

    Returns:
        (sample_rate, audio_array) 或 None（如果失败）
    """
    try:
        mode_str = "fast" if fast_mode else "orpheus"
        LOG.info(f"[TTS] Calling Speech API ({mode_str}) for speaker: {speaker}, text: {text[:30]}...")
        
        response = requests.post(
            f"{SPEECH_API_URL}/synthesize/json",
            json={"text": text, "speaker": speaker, "fast_mode": fast_mode},
            timeout=120 if not fast_mode else 30  # fast mode需要较少时间
        )
        
        if response.status_code != 200:
            LOG.error(f"[TTS] API error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        audio_format = result.get("format", "wav")
        
        # 解码 base64 音频
        audio_bytes = base64.b64decode(result["audio_base64"])
        
        if audio_format == "mp3":
            # MP3格式（Edge-TTS返回）- 需要使用pydub或其他方式解码
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                sample_rate = audio_segment.frame_rate
                # 转换为numpy数组
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                samples = samples / 32768.0  # 归一化到 [-1, 1]
                # 如果是立体声，转换为单声道
                if audio_segment.channels == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                LOG.info(f"[TTS] Successfully synthesized (MP3): {len(samples)} samples at {sample_rate}Hz")
                return (sample_rate, samples)
            except ImportError:
                LOG.warning("[TTS] pydub not installed, returning raw MP3 bytes")
                # 如果没有pydub，直接返回MP3字节供Gradio处理
                return (24000, audio_bytes)
        else:
            # WAV格式（Orpheus返回）- 使用标准库 wave
            import wave
            
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                # 读取原始音频数据
                raw_data = wav_file.readframes(n_frames)
                
                # 根据样本宽度解析数据
                if sample_width == 2:  # 16-bit
                    audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:  # 32-bit
                    audio_data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:  # 8-bit
                    audio_data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                
                # 如果是立体声，转换为单声道
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            LOG.info(f"[TTS] Successfully synthesized (WAV): {len(audio_data)} samples at {sample_rate}Hz")
            return (sample_rate, audio_data)
        
    except requests.exceptions.ConnectionError:
        LOG.error(f"[TTS] Cannot connect to Speech API at {SPEECH_API_URL}. Is it running?")
        return None
    except requests.exceptions.Timeout:
        LOG.error(f"[TTS] Request timed out. TTS model may be loading, please try again.")
        return None
    except Exception as e:
        LOG.error(f"[TTS] 语音合成失败: {e}")
        return None


def transcribe_audio(audio_data: Tuple[int, np.ndarray]) -> Optional[str]:
    """
    通过 Speech API 转录语音

    Args:
        audio_data: (sample_rate, audio_array)

    Returns:
        转录的文本或 None（如果失败）
    """
    try:
        sample_rate, audio = audio_data
        
        # 确保音频是 float32 格式
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            # 如果是整数格式，归一化到 [-1, 1]
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0
        
        LOG.info(f"[STT] Calling Speech API for transcription")
        
        # 将音频转换为 WAV bytes（使用标准库 wave）
        import wave
        buffer = io.BytesIO()
        
        # 转换为 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        buffer.seek(0)
        
        # 发送文件 - 参数名必须是 "audio" 与 API 定义匹配
        files = {"audio": ("audio.wav", buffer, "audio/wav")}
        # 增加超时时间：(连接超时, 读取超时) - Whisper 模型首次加载或处理长音频需要更多时间
        response = requests.post(
            f"{SPEECH_API_URL}/transcribe",
            files=files,
            timeout=(10, 120)  # 连接10秒，读取120秒
        )
        
        if response.status_code != 200:
            LOG.error(f"[STT] API error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        text = result.get("text", "")
        
        LOG.info(f"[STT] Transcription result: {text[:50]}...")
        return text
        
    except requests.exceptions.ConnectionError:
        LOG.error(f"[STT] Cannot connect to Speech API at {SPEECH_API_URL}. Is it running?")
        return None
    except Exception as e:
        LOG.error(f"[STT] 语音转录失败: {e}")
        return None


# 初始化场景代理
agents = {
    "job_interview": ScenarioAgent("job_interview"),
    "hotel_checkin": ScenarioAgent("hotel_checkin"),
    "renting": ScenarioAgent("renting"),
    "salary_negotiation": ScenarioAgent("salary_negotiation"),
    # 可以根据需要添加更多场景代理
}

# 难度级别选项
DIFFICULTY_CHOICES = [
    ("初级 (A1/A2)", "primary"),
    ("中级 (B1/B2)", "medium"),
    ("高级 (C1/C2)", "advanced"),
]

# 对话轮数选项
TURNS_CHOICES = [
    ("简短 (10轮)", 10),
    ("标准 (20轮)", 20),
    ("扩展 (30轮)", 30),
    ("深度 (50轮)", 50),
]


def get_page_desc(scenario):
    try:
        with open(f"content/page/{scenario}.md", "r", encoding="utf-8") as file:
            scenario_intro = file.read().strip()
        return scenario_intro
    except FileNotFoundError:
        LOG.error(f"场景介绍文件 content/page/{scenario}.md 未找到！")
        return "场景介绍文件未找到。"


def build_config_from_ui(turns: int, difficulty: str) -> ConversationConfig:
    """从 UI 控件值构建会话配置。"""
    return create_config(turns=turns, difficulty=difficulty)


# 获取场景介绍并启动新会话的函数
def start_new_scenario_chatbot(scenario, turns, difficulty):
    """
    切换场景时启动新的聊天会话，清除之前的聊天历史。

    参数:
        scenario: 场景名称
        turns: 对话轮数
        difficulty: 难度级别

    返回:
        list: 包含初始AI消息的消息列表，用于重置聊天界面
    """
    LOG.info(f"[Scenario] Switching to scenario: {scenario}, turns={turns}, difficulty={difficulty}")

    # 创建配置
    config = build_config_from_ui(turns, difficulty)

    # 启动新场景的会话并清除历史
    # 这会清除当前场景的会话历史，确保每次切换场景都是全新的会话
    initial_ai_message = agents[scenario].start_new_session(config=config)

    # Gradio 6.0.0 使用字典格式的消息
    # 返回新的消息列表会替换聊天机器人中的所有历史消息
    # 这应该清除 ChatInterface 中显示的旧消息
    LOG.debug(f"[Scenario] Returning new initial message for chatbot reset")
    return [{"role": "assistant", "content": initial_ai_message}]


def create_scenario_tab():
    with gr.Tab("场景"):  # 场景标签
        gr.Markdown("## 选择一个场景完成目标和挑战")  # 场景选择说明

        with gr.Row():
            # 左侧边栏：配置选项
            with gr.Column(scale=1):
                gr.Markdown("### 会话设置")

                # 场景选择
                scenario_radio = gr.Radio(
                    choices=[
                        ("求职面试", "job_interview"),
                        ("酒店入住", "hotel_checkin"),
                        ("租房", "renting"),
                        ("薪资谈判", "salary_negotiation"),
                    ],
                    label="场景",
                    value="job_interview",
                )

                # 难度级别下拉菜单
                difficulty_dropdown = gr.Dropdown(
                    choices=DIFFICULTY_CHOICES,
                    label="难度级别",
                    value="medium",
                    info="选择语言难度：初级(A1/A2)、中级(B1/B2)、高级(C1/C2)",
                )

                # 对话轮数滑块
                turns_slider = gr.Slider(
                    minimum=10,
                    maximum=50,
                    step=10,
                    value=20,
                    label="对话轮数",
                    info="选择对话轮数，完成后会收到反馈",
                )

                # Speaker 选择（TTS 语音）
                gr.Markdown("### 语音设置")
                speaker_dropdown = gr.Dropdown(
                    choices=SPEAKER_CHOICES,
                    label="TTS 语音角色",
                    value="Ceylia",
                    info="选择 AI 回复的语音角色",
                )

                # TTS 开关
                tts_enabled = gr.Checkbox(
                    label="启用语音播放",
                    value=True,  # 默认开启
                    info="开启后 AI 回复会自动生成并播放语音",
                )

                # 开始新会话按钮
                start_btn = gr.Button("开始新会话", variant="primary")

                # 场景介绍
                scenario_intro = gr.Markdown()

            # 右侧：聊天界面
            with gr.Column(scale=2):
                # 使用 State 来跟踪当前场景
                current_scenario_state = gr.State(value=None)
                # 存储最后一条 AI 消息用于 TTS
                last_ai_message_state = gr.State(value="")

                scenario_chatbot = gr.Chatbot(
                    placeholder="<strong>你的英语私教 LangCoach</strong><br><br>选择场景后开始对话吧！",
                    height=450,
                    value=None,
                )

                # TTS 播放区域：播放按钮 + 音频播放器
                with gr.Row():
                    tts_play_btn = gr.Button("🔊 播放 AI 语音", variant="secondary", scale=1)
                    audio_output = gr.Audio(
                        label="AI 语音",
                        type="numpy",
                        autoplay=True,
                        scale=3,
                        elem_id="ai_audio_player",
                    )
                
                # JavaScript 强制自动播放（绕过浏览器限制）
                gr.HTML("""
                <script>
                // 监听音频元素变化，尝试自动播放
                const observer = new MutationObserver((mutations) => {
                    const audioContainer = document.getElementById('ai_audio_player');
                    if (audioContainer) {
                        const audio = audioContainer.querySelector('audio');
                        if (audio && audio.src && audio.paused) {
                            audio.play().catch(e => console.log('Autoplay blocked:', e));
                        }
                    }
                });
                
                // 开始观察
                setTimeout(() => {
                    const target = document.getElementById('ai_audio_player');
                    if (target) {
                        observer.observe(target, { childList: true, subtree: true, attributes: true });
                    }
                }, 1000);
                </script>
                """)

                # 手动创建聊天输入框和发送按钮，放在同一行
                with gr.Row():
                    scenario_input = gr.Textbox(
                        placeholder="输入你的消息...",
                        label="消息",
                        scale=7,
                        container=False,
                    )
                    scenario_submit_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)

                # 语音输入区域
                with gr.Row():
                    audio_input = gr.Audio(
                        label="🎤 语音输入（录音后点击发送语音）",
                        sources=["microphone"],
                        type="numpy",
                        scale=3,
                    )
                    voice_submit_btn = gr.Button("发送语音", variant="secondary", scale=1, min_width=80)

        # 更新场景介绍并在场景变化时启动新会话
        def on_scenario_change(scenario, current_state, turns, difficulty):
            """处理场景切换，重置聊天界面"""
            LOG.info(f"[Scenario] Scenario changed from {current_state} to: {scenario}")

            # 如果场景发生变化，清除之前场景的会话历史
            if current_state and current_state != scenario and current_state in agents:
                LOG.debug(f"[Scenario] Clearing previous scenario history: {current_state}")
                agents[current_state].start_new_session()  # 清除之前场景的历史

            # 启动新场景的会话
            intro = get_page_desc(scenario)
            new_chat_history = start_new_scenario_chatbot(scenario, turns, difficulty)

            # 获取初始 AI 消息
            initial_ai_message = ""
            if new_chat_history and len(new_chat_history) > 0:
                initial_ai_message = new_chat_history[0].get("content", "")

            LOG.debug(f"[Scenario] Returning intro and new chat history for scenario: {scenario}, history: {new_chat_history}")
            # 直接返回新消息列表，这会替换 chatbot 中的所有旧消息
            # 同时清空音频输出，保存初始 AI 消息
            return intro, new_chat_history, scenario, None, initial_ai_message

        scenario_radio.change(
            fn=on_scenario_change,
            inputs=[scenario_radio, current_scenario_state, turns_slider, difficulty_dropdown],
            outputs=[scenario_intro, scenario_chatbot, current_scenario_state, audio_output, last_ai_message_state],
        )

        # 开始新会话按钮点击事件
        def on_start_new_session(scenario, turns, difficulty):
            """手动开始新会话"""
            LOG.info(f"[Scenario] Starting new session: {scenario}, turns={turns}, difficulty={difficulty}")
            intro = get_page_desc(scenario)
            new_chat_history = start_new_scenario_chatbot(scenario, turns, difficulty)

            # 获取初始 AI 消息
            initial_ai_message = ""
            if new_chat_history and len(new_chat_history) > 0:
                initial_ai_message = new_chat_history[0].get("content", "")

            return intro, new_chat_history, None, initial_ai_message

        start_btn.click(
            fn=on_start_new_session,
            inputs=[scenario_radio, turns_slider, difficulty_dropdown],
            outputs=[scenario_intro, scenario_chatbot, audio_output, last_ai_message_state],
        )

        # TTS 播放按钮点击事件
        def on_tts_play(last_message, speaker):
            """点击播放按钮时生成 TTS"""
            if not last_message:
                LOG.warning("[Scenario] No AI message to play")
                return None

            # 提取纯英文回复部分（排除对话提示）
            english_response = extract_english_response(last_message)
            LOG.info(f"[Scenario] Playing TTS for: {english_response[:50]}...")
            audio_result = synthesize_speech(english_response, speaker)
            return audio_result

        tts_play_btn.click(
            fn=on_tts_play,
            inputs=[last_ai_message_state, speaker_dropdown],
            outputs=[audio_output],
        )

        # 处理用户消息的函数（支持 TTS）
        def on_message_submit(user_input, chat_history, scenario, speaker, enable_tts):
            """处理用户提交的消息"""
            if not user_input or not user_input.strip():
                return chat_history or [], "", None, ""

            LOG.debug(f"[Scenario] User message submitted for scenario: {scenario}")

            # 确保 chat_history 不为 None
            if chat_history is None:
                chat_history = []

            # 创建新的聊天历史列表（避免直接修改原列表）
            new_chat_history = list(chat_history) if chat_history else []

            # 添加用户消息到聊天历史
            new_chat_history.append({"role": "user", "content": user_input.strip()})

            # 获取AI回复
            bot_message = agents[scenario].chat_with_history(user_input.strip())

            # 添加AI回复到聊天历史
            new_chat_history.append({"role": "assistant", "content": bot_message})

            LOG.debug(f"[Scenario] Chat history updated, length: {len(new_chat_history)}")

            # 如果启用 TTS，自动生成语音（只对英文回复部分）
            audio_result = None
            if enable_tts:
                english_response = extract_english_response(bot_message)
                LOG.info(f"[Scenario] Auto-generating TTS for: {english_response[:50]}...")
                audio_result = synthesize_speech(english_response, speaker)

            return new_chat_history, "", audio_result, bot_message

        # 处理语音输入的函数
        def on_voice_submit(audio_data, chat_history, scenario, speaker, enable_tts):
            """处理语音输入"""
            if audio_data is None:
                return chat_history or [], None, None, ""

            LOG.info("[Scenario] Processing voice input...")

            # 转录语音
            transcribed_text = transcribe_audio(audio_data)
            if not transcribed_text:
                LOG.warning("[Scenario] Voice transcription failed or empty")
                return chat_history or [], None, None, ""

            LOG.info(f"[Scenario] Transcribed text: {transcribed_text}")

            # 使用转录的文本进行对话
            new_chat_history, _, audio_result, bot_message = on_message_submit(
                transcribed_text, chat_history, scenario, speaker, enable_tts
            )

            # 清空音频输入
            return new_chat_history, None, audio_result, bot_message

        # 绑定文本提交事件
        scenario_submit_btn.click(
            fn=on_message_submit,
            inputs=[scenario_input, scenario_chatbot, scenario_radio, speaker_dropdown, tts_enabled],
            outputs=[scenario_chatbot, scenario_input, audio_output, last_ai_message_state],
        )

        # 也支持回车键提交
        scenario_input.submit(
            fn=on_message_submit,
            inputs=[scenario_input, scenario_chatbot, scenario_radio, speaker_dropdown, tts_enabled],
            outputs=[scenario_chatbot, scenario_input, audio_output, last_ai_message_state],
        )

        # 绑定语音提交事件
        voice_submit_btn.click(
            fn=on_voice_submit,
            inputs=[audio_input, scenario_chatbot, scenario_radio, speaker_dropdown, tts_enabled],
            outputs=[scenario_chatbot, audio_input, audio_output, last_ai_message_state],
        )
