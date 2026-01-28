#!/usr/bin/env python3
"""
STT Service for LangCoach

This module provides a service wrapper around the STT inference engine,
using Whisper-large-v3 with 4bit quantization.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from .inference import STTConfig, WhisperSTTInference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default model
DEFAULT_MODEL = "unsloth/whisper-large-v3"


@dataclass
class STTServiceConfig:
    """Configuration for STT service."""
    model_name: str = DEFAULT_MODEL
    device: str = "cuda"

    # Always use 4bit quantization for service
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Default inference parameters
    language: Optional[str] = None  # Auto-detect
    task: str = "transcribe"

    # Audio settings
    sample_rate: int = 16000


class STTService:
    """
    STT Service using Whisper with 4bit quantization.

    This service wraps the WhisperSTTInference class and provides
    a simplified interface for speech-to-text transcription.
    """

    def __init__(self, config: Optional[STTServiceConfig] = None):
        self.config = config or STTServiceConfig()
        self._engine: Optional[WhisperSTTInference] = None
        self._initialized = False

    def initialize(self):
        """Initialize the STT engine. Call this before using the service."""
        if self._initialized:
            logger.info("STT service already initialized")
            return

        logger.info("Initializing STT service...")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"4bit quantization: {self.config.load_in_4bit}")

        inference_config = STTConfig(
            model_name=self.config.model_name,
            device=self.config.device,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            language=self.config.language,
            task=self.config.task,
            sample_rate=self.config.sample_rate
        )

        self._engine = WhisperSTTInference(inference_config)
        self._initialized = True
        logger.info("STT service initialized successfully")

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            language: Language code (e.g., "en", "zh"). Auto-detect if None
            task: "transcribe" or "translate"

        Returns:
            Dictionary with:
                - text: transcribed text
                - language: detected or specified language
        """
        if not self._initialized:
            raise RuntimeError("STT service not initialized. Call initialize() first.")

        logger.info(f"Transcribing audio (sample_rate={sample_rate}, language={language or 'auto'})")

        result = self._engine.transcribe(
            audio=audio,
            sample_rate=sample_rate,
            language=language,
            task=task
        )

        return result

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code
            task: "transcribe" or "translate"

        Returns:
            Dictionary with transcription results
        """
        if not self._initialized:
            raise RuntimeError("STT service not initialized. Call initialize() first.")

        logger.info(f"Transcribing file: {audio_path}")

        return self._engine.transcribe_file(audio_path, language, task)


# Global service instance
_stt_service: Optional[STTService] = None


def get_stt_service(config: Optional[STTServiceConfig] = None) -> STTService:
    """
    Get or create global STT service instance.

    Args:
        config: Optional configuration. Only used if service not yet created.

    Returns:
        STTService instance
    """
    global _stt_service

    if _stt_service is None:
        _stt_service = STTService(config)

    return _stt_service


def initialize_stt_service(config: Optional[STTServiceConfig] = None) -> STTService:
    """
    Initialize the global STT service.

    Args:
        config: Optional configuration

    Returns:
        Initialized STTService instance
    """
    service = get_stt_service(config)
    service.initialize()
    return service
