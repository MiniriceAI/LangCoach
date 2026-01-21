"""
TTS Evaluator Module

Evaluates Text-to-Speech performance including:
- Synthesis latency
- Audio quality metrics
- Real-time factor (RTF)
"""

import os
import sys
import io
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from .base import BaseEvaluator, Timer


class TTSEvaluator(BaseEvaluator):
    """
    Evaluator for TTS performance.

    Supports two modes:
    - Local: Uses local Orpheus TTS model
    - API: Uses Speech API endpoint
    - Fast: Uses Edge-TTS (Microsoft Azure)
    """

    def __init__(
        self,
        mode: str = "api",
        speaker: str = "Ceylia",
        fast_mode: bool = True,
        api_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TTS evaluator.

        Args:
            mode: "local", "api", or "fast"
            speaker: Speaker name for synthesis
            fast_mode: Use Edge-TTS fast mode (only for API mode)
            api_url: Speech API URL (for API mode)
            config: Additional configuration
        """
        super().__init__(config)
        self.mode = mode
        self.speaker = speaker
        self.fast_mode = fast_mode
        self.api_url = api_url or os.getenv("SPEECH_API_URL", "http://localhost:8600")
        self._service = None

    def initialize(self):
        """Initialize the TTS service."""
        if self._initialized:
            return

        if self.mode == "local":
            from tts.service import initialize_tts_service
            self._service = initialize_tts_service()

        self._initialized = True

    def get_provider_info(self) -> Dict[str, str]:
        """Get provider information."""
        if self.mode == "fast" or (self.mode == "api" and self.fast_mode):
            return {
                "provider": "edge-tts",
                "model": "Microsoft Azure TTS",
            }
        return {
            "provider": "orpheus",
            "model": "unsloth/orpheus-3b-0.1-ft",
        }

    def evaluate_single(self, sample: Any) -> Dict[str, Any]:
        """
        Evaluate a single sample.

        Args:
            sample: BenchmarkSample or dict with 'user_input' key,
                   or string text to synthesize

        Returns:
            Evaluation result dictionary
        """
        # Extract text to synthesize
        if hasattr(sample, 'user_input'):
            text = sample.user_input
        elif isinstance(sample, dict):
            text = sample.get('user_input', sample.get('text', str(sample)))
        else:
            text = str(sample)

        try:
            if self.mode == "local":
                return self._evaluate_local(text)
            else:
                return self._evaluate_api(text)

        except Exception as e:
            return {
                "success": False,
                "latency": 0.0,
                "output": None,
                "error": str(e),
            }

    def _evaluate_local(self, text: str) -> Dict[str, Any]:
        """Evaluate using local TTS service."""
        with Timer() as timer:
            result = self._service.synthesize(
                text=text,
                speaker=self.speaker
            )

        audio = result["audio"]
        sample_rate = result["sample_rate"]

        # Calculate metrics
        audio_duration = len(audio) / sample_rate
        rtf = timer.elapsed / audio_duration if audio_duration > 0 else 0

        return {
            "success": True,
            "latency": timer.elapsed,
            "output": {
                "audio_length": len(audio),
                "sample_rate": sample_rate,
                "duration_seconds": audio_duration,
            },
            "extra": {
                "audio_duration": audio_duration,
                "rtf": rtf,  # Real-time factor
                "text_length": len(text),
                "chars_per_second": len(text) / timer.elapsed if timer.elapsed > 0 else 0,
            }
        }

    def _evaluate_api(self, text: str) -> Dict[str, Any]:
        """Evaluate using Speech API."""
        import requests

        with Timer() as timer:
            response = requests.post(
                f"{self.api_url}/synthesize/json",
                json={
                    "text": text,
                    "speaker": self.speaker,
                    "fast_mode": self.fast_mode
                },
                timeout=120
            )

        if response.status_code != 200:
            return {
                "success": False,
                "latency": timer.elapsed,
                "output": None,
                "error": f"API error: {response.status_code} - {response.text}",
            }

        result = response.json()
        sample_rate = result.get("sample_rate", 24000)

        # Estimate audio duration from base64 size
        import base64
        audio_bytes = base64.b64decode(result["audio_base64"])
        # Rough estimate: assume 16-bit audio
        estimated_samples = len(audio_bytes) // 2
        audio_duration = estimated_samples / sample_rate

        rtf = timer.elapsed / audio_duration if audio_duration > 0 else 0

        return {
            "success": True,
            "latency": timer.elapsed,
            "output": {
                "audio_size_bytes": len(audio_bytes),
                "sample_rate": sample_rate,
                "format": result.get("format", "wav"),
                "duration_seconds": audio_duration,
            },
            "extra": {
                "audio_duration": audio_duration,
                "rtf": rtf,
                "text_length": len(text),
                "chars_per_second": len(text) / timer.elapsed if timer.elapsed > 0 else 0,
            }
        }

    def cleanup(self):
        """Clean up resources."""
        self._service = None
        self._initialized = False


class TTSComparisonEvaluator:
    """
    Evaluator for comparing TTS modes.

    Compares local Orpheus TTS vs Edge-TTS (fast mode).
    """

    def __init__(
        self,
        modes: List[str] = None,
        speaker: str = "Ceylia",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize comparison evaluator.

        Args:
            modes: List of modes to compare ("local", "fast")
            speaker: Speaker name
            config: Additional configuration
        """
        self.modes = modes or ["fast", "local"]
        self.speaker = speaker
        self.config = config or {}
        self._evaluators: Dict[str, TTSEvaluator] = {}

    def initialize(self):
        """Initialize all evaluators."""
        for mode in self.modes:
            fast_mode = mode == "fast"
            evaluator = TTSEvaluator(
                mode="api" if mode == "fast" else mode,
                speaker=self.speaker,
                fast_mode=fast_mode,
                config=self.config
            )
            try:
                evaluator.initialize()
                self._evaluators[mode] = evaluator
            except Exception as e:
                print(f"Warning: Failed to initialize TTS {mode}: {e}")

    def evaluate(
        self,
        samples: List[Any],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all modes with the same samples.

        Args:
            samples: List of samples to evaluate
            progress_callback: Optional callback(mode, current, total)

        Returns:
            Dictionary with results for each mode
        """
        if not self._evaluators:
            self.initialize()

        results = {}
        for mode, evaluator in self._evaluators.items():
            def mode_progress(current, total):
                if progress_callback:
                    progress_callback(mode, current, total)

            result = evaluator.evaluate(samples, mode_progress)
            results[mode] = result

        return results

    def cleanup(self):
        """Clean up all evaluators."""
        for evaluator in self._evaluators.values():
            evaluator.cleanup()
        self._evaluators.clear()
