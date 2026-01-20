"""
STT Evaluator Module

Evaluates Speech-to-Text performance including:
- Transcription latency
- Word Error Rate (WER) if reference available
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


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = words in reference
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


class STTEvaluator(BaseEvaluator):
    """
    Evaluator for STT performance.

    Supports two modes:
    - Local: Uses local Whisper model
    - API: Uses Speech API endpoint
    """

    def __init__(
        self,
        mode: str = "api",
        language: Optional[str] = "en",
        api_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize STT evaluator.

        Args:
            mode: "local" or "api"
            language: Language code for transcription
            api_url: Speech API URL (for API mode)
            config: Additional configuration
        """
        super().__init__(config)
        self.mode = mode
        self.language = language
        self.api_url = api_url or os.getenv("SPEECH_API_URL", "http://localhost:8301")
        self._service = None

    def initialize(self):
        """Initialize the STT service."""
        if self._initialized:
            return

        if self.mode == "local":
            from stt.service import initialize_stt_service
            self._service = initialize_stt_service()

        self._initialized = True

    def get_provider_info(self) -> Dict[str, str]:
        """Get provider information."""
        return {
            "provider": "whisper",
            "model": "unsloth/whisper-large-v3",
        }

    def evaluate_single(self, sample: Any) -> Dict[str, Any]:
        """
        Evaluate a single sample.

        For STT evaluation, we need audio data. This can come from:
        1. A pre-recorded audio file (sample.audio_path)
        2. Synthesized audio from text (for testing pipeline)

        Args:
            sample: BenchmarkSample with audio_path or audio data

        Returns:
            Evaluation result dictionary
        """
        try:
            # Get audio data
            audio_data, sample_rate, reference_text = self._get_audio_data(sample)

            if audio_data is None:
                return {
                    "success": False,
                    "latency": 0.0,
                    "output": None,
                    "error": "No audio data available for STT evaluation",
                }

            if self.mode == "local":
                return self._evaluate_local(audio_data, sample_rate, reference_text)
            else:
                return self._evaluate_api(audio_data, sample_rate, reference_text)

        except Exception as e:
            return {
                "success": False,
                "latency": 0.0,
                "output": None,
                "error": str(e),
            }

    def _get_audio_data(self, sample: Any) -> tuple:
        """
        Get audio data from sample.

        Returns:
            (audio_array, sample_rate, reference_text)
        """
        reference_text = None

        # Check for audio path
        if hasattr(sample, 'audio_path') and sample.audio_path:
            import librosa
            audio, sr = librosa.load(sample.audio_path, sr=16000)
            reference_text = getattr(sample, 'expected_transcription', None)
            return audio, sr, reference_text

        # Check for audio data in dict
        if isinstance(sample, dict):
            if 'audio' in sample:
                audio = sample['audio']
                sr = sample.get('sample_rate', 16000)
                reference_text = sample.get('expected_transcription')
                return audio, sr, reference_text
            if 'audio_path' in sample:
                import librosa
                audio, sr = librosa.load(sample['audio_path'], sr=16000)
                reference_text = sample.get('expected_transcription')
                return audio, sr, reference_text

        # For benchmark samples without audio, we can synthesize test audio
        # using TTS first (this tests the full pipeline)
        if hasattr(sample, 'user_input') or (isinstance(sample, dict) and 'user_input' in sample):
            text = sample.user_input if hasattr(sample, 'user_input') else sample['user_input']
            audio, sr = self._synthesize_test_audio(text)
            return audio, sr, text

        return None, None, None

    def _synthesize_test_audio(self, text: str) -> tuple:
        """
        Synthesize test audio using Edge-TTS for STT evaluation.

        This allows testing STT without pre-recorded audio files.
        """
        import requests
        import base64

        try:
            response = requests.post(
                f"{self.api_url}/synthesize/json",
                json={"text": text, "speaker": "Ceylia", "fast_mode": True},
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"TTS API error: {response.status_code}")

            result = response.json()
            audio_bytes = base64.b64decode(result["audio_base64"])

            # Convert MP3 to numpy array
            from pydub import AudioSegment
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

            # Resample to 16kHz for Whisper
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0

            return samples, 16000

        except Exception as e:
            # Fallback: generate simple sine wave as placeholder
            print(f"Warning: Could not synthesize test audio: {e}")
            duration = 2.0
            sr = 16000
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            return audio, sr

    def _evaluate_local(self, audio: np.ndarray, sample_rate: int, reference: Optional[str]) -> Dict[str, Any]:
        """Evaluate using local STT service."""
        audio_duration = len(audio) / sample_rate

        with Timer() as timer:
            result = self._service.transcribe(
                audio=audio,
                sample_rate=sample_rate,
                language=self.language
            )

        transcription = result.get("text", "")
        rtf = timer.elapsed / audio_duration if audio_duration > 0 else 0

        extra = {
            "audio_duration": audio_duration,
            "rtf": rtf,
            "transcription_length": len(transcription),
        }

        # Calculate WER if reference available
        if reference:
            wer = calculate_wer(reference, transcription)
            extra["wer"] = wer

        return {
            "success": True,
            "latency": timer.elapsed,
            "output": {
                "text": transcription,
                "language": result.get("language", self.language),
            },
            "extra": extra,
        }

    def _evaluate_api(self, audio: np.ndarray, sample_rate: int, reference: Optional[str]) -> Dict[str, Any]:
        """Evaluate using Speech API."""
        import requests
        import wave

        audio_duration = len(audio) / sample_rate

        # Convert to WAV bytes
        buffer = io.BytesIO()
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)

        with Timer() as timer:
            response = requests.post(
                f"{self.api_url}/transcribe",
                files={"audio": ("audio.wav", buffer, "audio/wav")},
                data={"language": self.language} if self.language else {},
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
        transcription = result.get("text", "")
        rtf = timer.elapsed / audio_duration if audio_duration > 0 else 0

        extra = {
            "audio_duration": audio_duration,
            "rtf": rtf,
            "transcription_length": len(transcription),
        }

        # Calculate WER if reference available
        if reference:
            wer = calculate_wer(reference, transcription)
            extra["wer"] = wer

        return {
            "success": True,
            "latency": timer.elapsed,
            "output": {
                "text": transcription,
                "language": result.get("language", self.language),
            },
            "extra": extra,
        }

    def cleanup(self):
        """Clean up resources."""
        self._service = None
        self._initialized = False
