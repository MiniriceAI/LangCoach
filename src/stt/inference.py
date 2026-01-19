#!/usr/bin/env python3
"""
STT Inference Script for LangCoach

This script provides Speech-to-Text inference using Whisper-large-v3 model
with 4bit quantization for efficient inference.

Usage:
    # Basic usage
    python inference.py --audio_path audio.wav

    # With 4bit quantization (default)
    python inference.py --audio_path audio.wav --load_in_4bit

    # Specify language
    python inference.py --audio_path audio.wav --language en
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class STTConfig:
    """Configuration for STT inference."""
    model_name: str = "unsloth/whisper-large-v3"
    device: str = "cuda"

    # Quantization settings
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Inference parameters
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # "transcribe" or "translate"

    # Audio settings
    sample_rate: int = 16000


class WhisperSTTInference:
    """Whisper-based Speech-to-Text inference engine with quantization support."""

    def __init__(self, config: STTConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model with quantization."""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration, BitsAndBytesConfig

        logger.info(f"Loading Whisper model: {self.config.model_name}")
        logger.info(f"Quantization: 4bit={self.config.load_in_4bit}, 8bit={self.config.load_in_8bit}")

        # Load processor
        self.processor = WhisperProcessor.from_pretrained(self.config.model_name)

        # Configure quantization
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        logger.info("Whisper model loaded successfully")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: Optional[str] = None,
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            language: Language code (e.g., "en", "zh"). Auto-detect if None
            task: "transcribe" or "translate"
            return_timestamps: Whether to return word-level timestamps

        Returns:
            Dictionary with transcription results
        """
        import librosa

        # Resample if needed
        if sample_rate != self.config.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.config.sample_rate
            )

        # Process audio
        input_features = self.processor(
            audio,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt"
        ).input_features

        # Move to device and convert dtype to match model
        input_features = input_features.to(self.model.device)
        # 对于量化模型，需要将输入转换为 float16 以匹配模型权重
        if self.config.load_in_4bit or self.config.load_in_8bit:
            input_features = input_features.to(torch.float16)

        # Prepare generation kwargs
        generate_kwargs = {
            "task": task or self.config.task,
        }

        if language or self.config.language:
            generate_kwargs["language"] = language or self.config.language

        if return_timestamps:
            generate_kwargs["return_timestamps"] = True

        # Generate
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                **generate_kwargs
            )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        result = {
            "text": transcription.strip(),
            "language": language or self.config.language or "auto"
        }

        logger.info(f"Transcription: {result['text'][:50]}...")

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
        import librosa

        logger.info(f"Loading audio file: {audio_path}")

        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)

        return self.transcribe(audio, sr, language, task)


# Global instance for service use
_stt_instance: Optional[WhisperSTTInference] = None


def get_stt_instance(config: Optional[STTConfig] = None) -> WhisperSTTInference:
    """Get or create global STT instance."""
    global _stt_instance

    if _stt_instance is None:
        if config is None:
            config = STTConfig()
        _stt_instance = WhisperSTTInference(config)

    return _stt_instance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Whisper STT Inference for LangCoach"
    )

    parser.add_argument(
        "--audio_path",
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--model_name",
        default="unsloth/whisper-large-v3",
        help="Whisper model name"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., 'en', 'zh'). Auto-detect if not specified"
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task type"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model with 4bit quantization (default: True)"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model with 8bit quantization"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable quantization"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine quantization settings
    load_in_4bit = args.load_in_4bit and not args.no_quantization and not args.load_in_8bit
    load_in_8bit = args.load_in_8bit and not args.no_quantization

    config = STTConfig(
        model_name=args.model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        language=args.language,
        task=args.task
    )

    stt = WhisperSTTInference(config)

    # Transcribe
    result = stt.transcribe_file(args.audio_path, args.language, args.task)

    print(f"\nTranscription Result:")
    print(f"  Language: {result['language']}")
    print(f"  Text: {result['text']}")


if __name__ == "__main__":
    main()
