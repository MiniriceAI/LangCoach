# STT Module for LangCoach
# Speech-to-Text using Whisper model with 4bit quantization

from .inference import STTConfig, WhisperSTTInference, get_stt_instance
from .service import STTService, STTServiceConfig, get_stt_service, initialize_stt_service

__all__ = [
    "STTConfig",
    "WhisperSTTInference",
    "get_stt_instance",
    "STTService",
    "STTServiceConfig",
    "get_stt_service",
    "initialize_stt_service",
]
