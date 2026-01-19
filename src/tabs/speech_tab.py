# tabs/speech_tab.py
"""
Speech Tab for LangCoach

This tab provides voice-based interaction with the AI tutor,
supporting both speech input (STT) and speech output (TTS).
"""

import gradio as gr
import numpy as np
import tempfile
import os
from typing import Optional, Tuple, List
from utils.logger import LOG

# Lazy load speech services to avoid slow startup
_tts_service = None
_stt_service = None


def get_tts_service():
    """Lazy load TTS service."""
    global _tts_service
    if _tts_service is None:
        try:
            from tts.service import initialize_tts_service
            _tts_service = initialize_tts_service()
            LOG.info("TTS service initialized")
        except Exception as e:
            LOG.error(f"Failed to initialize TTS service: {e}")
            raise
    return _tts_service


def get_stt_service():
    """Lazy load STT service."""
    global _stt_service
    if _stt_service is None:
        try:
            from stt.service import initialize_stt_service
            _stt_service = initialize_stt_service()
            LOG.info("STT service initialized")
        except Exception as e:
            LOG.error(f"Failed to initialize STT service: {e}")
            raise
    return _stt_service


# Supported speakers
SPEAKER_CHOICES = [
    ("Ceylia (女声)", "Ceylia"),
    ("Tifa (女声)", "Tifa"),
]


def transcribe_audio(audio_input: Tuple[int, np.ndarray]) -> str:
    """
    Transcribe audio input to text.

    Args:
        audio_input: Tuple of (sample_rate, audio_data)

    Returns:
        Transcribed text
    """
    if audio_input is None:
        return ""

    try:
        sample_rate, audio_data = audio_input

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            # Normalize if int16
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0

        # Handle stereo audio
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        stt = get_stt_service()
        result = stt.transcribe(audio_data, sample_rate)

        LOG.info(f"Transcribed: {result['text'][:50]}...")
        return result["text"]

    except Exception as e:
        LOG.error(f"Transcription error: {e}")
        return f"[转录错误: {str(e)}]"


def synthesize_speech(text: str, speaker: str) -> Optional[Tuple[int, np.ndarray]]:
    """
    Synthesize speech from text.

    Args:
        text: Text to synthesize
        speaker: Speaker name

    Returns:
        Tuple of (sample_rate, audio_data) or None
    """
    if not text or not text.strip():
        return None

    try:
        tts = get_tts_service()
        result = tts.synthesize(text, speaker)

        LOG.info(f"Synthesized speech for '{speaker}': {text[:30]}...")
        return (result["sample_rate"], result["audio"])

    except Exception as e:
        LOG.error(f"Synthesis error: {e}")
        return None


def process_voice_message(
    audio_input: Tuple[int, np.ndarray],
    chat_history: List[dict],
    speaker: str,
    scenario_agent
) -> Tuple[List[dict], Optional[Tuple[int, np.ndarray]], str]:
    """
    Process voice input: transcribe -> get AI response -> synthesize.

    Args:
        audio_input: Audio from microphone
        chat_history: Current chat history
        speaker: TTS speaker
        scenario_agent: The scenario agent for conversation

    Returns:
        Tuple of (updated_history, response_audio, transcribed_text)
    """
    if audio_input is None:
        return chat_history or [], None, ""

    # Transcribe user speech
    user_text = transcribe_audio(audio_input)
    if not user_text or user_text.startswith("[转录错误"):
        return chat_history or [], None, user_text

    # Update chat history with user message
    if chat_history is None:
        chat_history = []
    new_history = list(chat_history)
    new_history.append({"role": "user", "content": user_text})

    # Get AI response
    try:
        ai_response = scenario_agent.chat_with_history(user_text)
        new_history.append({"role": "assistant", "content": ai_response})
    except Exception as e:
        LOG.error(f"AI response error: {e}")
        ai_response = f"抱歉，发生了错误: {str(e)}"
        new_history.append({"role": "assistant", "content": ai_response})

    # Synthesize AI response
    response_audio = synthesize_speech(ai_response, speaker)

    return new_history, response_audio, user_text


def create_speech_tab():
    """Create the speech interaction tab."""
    from agents.scenario_agent import ScenarioAgent
    from agents.conversation_config import create_config

    # Initialize scenario agents
    agents = {
        "job_interview": ScenarioAgent("job_interview"),
        "hotel_checkin": ScenarioAgent("hotel_checkin"),
        "renting": ScenarioAgent("renting"),
        "salary_negotiation": ScenarioAgent("salary_negotiation"),
    }

    with gr.Tab("语音对话"):
        gr.Markdown("## 语音交互模式")
        gr.Markdown("使用麦克风进行语音对话，AI 会用语音回复你。")

        with gr.Row():
            # Left sidebar: Settings
            with gr.Column(scale=1):
                gr.Markdown("### 设置")

                # Scenario selection
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

                # Speaker selection
                speaker_dropdown = gr.Dropdown(
                    choices=SPEAKER_CHOICES,
                    label="AI 语音",
                    value="Ceylia",
                    info="选择 AI 的语音角色",
                )

                # Difficulty
                difficulty_dropdown = gr.Dropdown(
                    choices=[
                        ("初级 (A1/A2)", "primary"),
                        ("中级 (B1/B2)", "medium"),
                        ("高级 (C1/C2)", "advanced"),
                    ],
                    label="难度级别",
                    value="medium",
                )

                # Start new session button
                start_btn = gr.Button("开始新会话", variant="primary")

                # Service status
                with gr.Accordion("服务状态", open=False):
                    status_text = gr.Markdown("点击检查状态")
                    check_status_btn = gr.Button("检查服务状态", size="sm")

            # Right side: Chat and audio
            with gr.Column(scale=2):
                # Chat display
                speech_chatbot = gr.Chatbot(
                    placeholder="<strong>语音对话模式</strong><br><br>点击麦克风开始录音，或输入文字发送。",
                    height=400,
                    value=None,
                )

                # Audio output for AI response
                audio_output = gr.Audio(
                    label="AI 语音回复",
                    autoplay=True,
                    visible=True,
                )

                # Audio input
                gr.Markdown("### 语音输入")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="点击录音",
                )

                # Text input as alternative
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="或者输入文字...",
                        label="文字输入",
                        scale=9,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                # Transcription display
                transcription_display = gr.Textbox(
                    label="语音识别结果",
                    interactive=False,
                    visible=True,
                )

        # State for current scenario
        current_scenario_state = gr.State(value="job_interview")

        # Event handlers
        def on_start_session(scenario, difficulty):
            """Start a new conversation session."""
            config = create_config(turns=20, difficulty=difficulty)
            initial_message = agents[scenario].start_new_session(config=config)
            return [{"role": "assistant", "content": initial_message}], scenario

        def on_scenario_change(scenario, current_state, difficulty):
            """Handle scenario change."""
            if current_state and current_state != scenario and current_state in agents:
                agents[current_state].start_new_session()

            config = create_config(turns=20, difficulty=difficulty)
            initial_message = agents[scenario].start_new_session(config=config)
            return [{"role": "assistant", "content": initial_message}], scenario

        def on_voice_submit(audio, history, speaker, scenario):
            """Handle voice input submission."""
            if audio is None:
                return history or [], None, ""

            new_history, response_audio, transcription = process_voice_message(
                audio, history, speaker, agents[scenario]
            )
            return new_history, response_audio, transcription

        def on_text_submit(text, history, speaker, scenario):
            """Handle text input submission."""
            if not text or not text.strip():
                return history or [], None, ""

            if history is None:
                history = []
            new_history = list(history)
            new_history.append({"role": "user", "content": text.strip()})

            # Get AI response
            try:
                ai_response = agents[scenario].chat_with_history(text.strip())
                new_history.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                ai_response = f"抱歉，发生了错误: {str(e)}"
                new_history.append({"role": "assistant", "content": ai_response})

            # Synthesize response
            response_audio = synthesize_speech(ai_response, speaker)

            return new_history, response_audio, ""

        def check_service_status():
            """Check if speech services are available."""
            global _tts_service, _stt_service

            tts_status = "已加载" if _tts_service and _tts_service.is_initialized else "未加载"
            stt_status = "已加载" if _stt_service and _stt_service.is_initialized else "未加载"

            return f"""
**TTS 服务**: {tts_status}
- 模型: unsloth/orpheus-3b-0.1-ft + LoRA
- 量化: 4bit

**STT 服务**: {stt_status}
- 模型: unsloth/whisper-large-v3
- 量化: 4bit
"""

        # Bind events
        start_btn.click(
            fn=on_start_session,
            inputs=[scenario_radio, difficulty_dropdown],
            outputs=[speech_chatbot, current_scenario_state],
        )

        scenario_radio.change(
            fn=on_scenario_change,
            inputs=[scenario_radio, current_scenario_state, difficulty_dropdown],
            outputs=[speech_chatbot, current_scenario_state],
        )

        # Voice input - process when recording stops
        audio_input.stop_recording(
            fn=on_voice_submit,
            inputs=[audio_input, speech_chatbot, speaker_dropdown, scenario_radio],
            outputs=[speech_chatbot, audio_output, transcription_display],
        )

        # Text input
        send_btn.click(
            fn=on_text_submit,
            inputs=[text_input, speech_chatbot, speaker_dropdown, scenario_radio],
            outputs=[speech_chatbot, audio_output, text_input],
        )

        text_input.submit(
            fn=on_text_submit,
            inputs=[text_input, speech_chatbot, speaker_dropdown, scenario_radio],
            outputs=[speech_chatbot, audio_output, text_input],
        )

        # Status check
        check_status_btn.click(
            fn=check_service_status,
            outputs=[status_text],
        )
