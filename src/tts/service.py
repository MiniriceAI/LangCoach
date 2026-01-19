#!/usr/bin/env python3
"""
TTS Service for LangCoach

This module provides a service wrapper around the TTS inference engine,
supporting multiple speakers with 4bit quantization.

The service loads the base model (unsloth/orpheus-3b-0.1-ft) with 4bit quantization
and applies LoRA adapters for multi-speaker support.
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np

from .inference import InferenceConfig, MultiSpeakerTTSInference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default paths
DEFAULT_BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"
DEFAULT_LORA_PATH = "/workspace/FineTunedModels/unsloth-orpheus-3b-0.1-ft-Ceylia-Tifa"

# Supported speakers from the fine-tuned model
SUPPORTED_SPEAKERS = ["Ceylia", "Tifa"]


@dataclass
class TTSServiceConfig:
    """Configuration for TTS service."""
    base_model: str = DEFAULT_BASE_MODEL
    lora_path: str = DEFAULT_LORA_PATH
    device: str = "cuda"

    # Always use 4bit quantization for service
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Generation parameters - optimized for speed
    max_new_tokens: int = 600  # Reduced for faster generation
    temperature: float = 0.4  # Lower for faster, more deterministic
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Output settings
    sample_rate: int = 24000


class TTSService:
    """
    TTS Service with multi-speaker support.

    This service wraps the MultiSpeakerTTSInference class and provides
    a simplified interface for text-to-speech synthesis.
    """

    def __init__(self, config: Optional[TTSServiceConfig] = None):
        self.config = config or TTSServiceConfig()
        self._engine: Optional[MultiSpeakerTTSInference] = None
        self._initialized = False

    def initialize(self):
        """Initialize the TTS engine. Call this before using the service."""
        if self._initialized:
            logger.info("TTS service already initialized")
            return

        logger.info("Initializing TTS service...")
        logger.info(f"Base model: {self.config.base_model}")
        logger.info(f"LoRA path: {self.config.lora_path}")
        logger.info(f"4bit quantization: {self.config.load_in_4bit}")

        inference_config = InferenceConfig(
            model_path=self.config.lora_path,
            use_lora=True,
            device=self.config.device,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            sample_rate=self.config.sample_rate
        )

        self._engine = MultiSpeakerTTSInference(inference_config)
        self._initialized = True
        logger.info("TTS service initialized successfully")

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    def get_supported_speakers(self) -> List[str]:
        """Get list of supported speakers."""
        if self._engine:
            return self._engine.get_supported_speakers()
        return SUPPORTED_SPEAKERS

    def synthesize(
        self,
        text: str,
        speaker: str = "Ceylia",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker: Speaker name (Ceylia or Tifa)
            temperature: Override default temperature
            top_p: Override default top_p

        Returns:
            Dictionary with:
                - audio: numpy array of audio samples
                - sample_rate: sample rate of the audio
                - speaker: speaker used
                - text: input text
        """
        if not self._initialized:
            raise RuntimeError("TTS service not initialized. Call initialize() first.")

        if speaker not in SUPPORTED_SPEAKERS:
            logger.warning(f"Speaker '{speaker}' not in supported list: {SUPPORTED_SPEAKERS}")

        logger.info(f"Synthesizing speech for speaker '{speaker}': {text[:50]}...")

        audio = self._engine.synthesize(
            text=text,
            speaker=speaker,
            temperature=temperature,
            top_p=top_p
        )

        return {
            "audio": audio,
            "sample_rate": self.config.sample_rate,
            "speaker": speaker,
            "text": text
        }

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        speaker: str = "Ceylia",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Synthesize speech and save to file.

        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            speaker: Speaker name
            temperature: Override default temperature
            top_p: Override default top_p

        Returns:
            Path to saved audio file
        """
        result = self.synthesize(text, speaker, temperature, top_p)

        self._engine.save_audio(result["audio"], output_path)

        return output_path


# Global service instance
_tts_service: Optional[TTSService] = None


def get_tts_service(config: Optional[TTSServiceConfig] = None) -> TTSService:
    """
    Get or create global TTS service instance.

    Args:
        config: Optional configuration. Only used if service not yet created.

    Returns:
        TTSService instance
    """
    global _tts_service

    if _tts_service is None:
        _tts_service = TTSService(config)

    return _tts_service


def initialize_tts_service(config: Optional[TTSServiceConfig] = None) -> TTSService:
    """
    Initialize the global TTS service.

    Args:
        config: Optional configuration

    Returns:
        Initialized TTSService instance
    """
    service = get_tts_service(config)
    service.initialize()
    return service
