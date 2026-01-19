# TTS Module for LangCoach
# Multi-speaker TTS training and inference using Orpheus model

from .inference import InferenceConfig, MultiSpeakerTTSInference
from .service import TTSService, TTSServiceConfig, get_tts_service, initialize_tts_service

__all__ = [
    "InferenceConfig",
    "MultiSpeakerTTSInference",
    "TTSService",
    "TTSServiceConfig",
    "get_tts_service",
    "initialize_tts_service",
]
