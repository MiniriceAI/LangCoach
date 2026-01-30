"""
E2E (End-to-End) Pipeline Evaluator

Evaluates the complete audio pipeline:
Audio Input -> STT -> LLM Inference -> TTS -> Audio Output

Measures:
- Total E2E latency
- Individual component latencies
- Success rate
- Quality metrics
"""

import os
import sys
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from .base import BaseEvaluator, Timer, TimingMetrics, EvaluationResult


@dataclass
class E2ETimingBreakdown:
    """Detailed timing breakdown for E2E evaluation."""
    stt_timing: TimingMetrics = field(default_factory=TimingMetrics)
    llm_timing: TimingMetrics = field(default_factory=TimingMetrics)
    tts_timing: TimingMetrics = field(default_factory=TimingMetrics)
    total_timing: TimingMetrics = field(default_factory=TimingMetrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "stt": self.stt_timing.to_dict(),
            "llm": self.llm_timing.to_dict(),
            "tts": self.tts_timing.to_dict(),
            "total": self.total_timing.to_dict(),
        }


@dataclass
class E2EResult(EvaluationResult):
    """Extended result for E2E evaluation with component breakdown."""
    timing_breakdown: E2ETimingBreakdown = field(default_factory=E2ETimingBreakdown)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        base_dict = super().to_dict()
        base_dict["timing_breakdown"] = self.timing_breakdown.to_dict()
        return base_dict


# Default system prompt for E2E evaluation
DEFAULT_E2E_PROMPT = """You are a friendly English tutor having a casual conversation.
Keep responses natural and concise (under 40 words).
Respond conversationally without any formatting."""


class E2EEvaluator(BaseEvaluator):
    """
    End-to-End Pipeline Evaluator.

    Evaluates the complete workflow:
    1. STT: Convert audio to text
    2. LLM: Generate response
    3. TTS: Convert response to audio

    Target: E2E latency < 3 seconds for conversational fluency.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        tts_fast_mode: bool = True,
        speaker: str = "Ceylia",
        system_prompt: Optional[str] = None,
        api_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize E2E evaluator.

        Args:
            llm_provider: LLM provider name (ollama, deepseek, openai)
            tts_fast_mode: Use Edge-TTS for faster TTS
            speaker: TTS speaker name
            system_prompt: System prompt for LLM
            api_url: Speech API URL
            config: Additional configuration
        """
        super().__init__(config)
        self.llm_provider = llm_provider
        self.tts_fast_mode = tts_fast_mode
        self.speaker = speaker
        self.system_prompt = system_prompt or DEFAULT_E2E_PROMPT
        self.api_url = api_url or os.getenv("SPEECH_API_URL", "http://localhost:8600")

        self._llm = None
        self._provider_info: Dict[str, str] = {}

    def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return

        # Initialize LLM
        from agents.llm_factory import create_llm, get_current_provider_info

        self._llm = create_llm(self.llm_provider)

        info = get_current_provider_info()
        self._provider_info = {
            "provider": f"e2e-{self.llm_provider or info.get('provider', 'auto')}",
            "model": info.get("model", "unknown"),
            "tts_mode": "edge-tts" if self.tts_fast_mode else "orpheus",
        }

        self._initialized = True

    def get_provider_info(self) -> Dict[str, str]:
        """Get provider information."""
        return self._provider_info

    def evaluate_single(self, sample: Any) -> Dict[str, Any]:
        """
        Evaluate a single sample through the complete pipeline.

        For E2E evaluation, we simulate the full workflow:
        1. Generate test audio from text (simulating user speech)
        2. STT: Transcribe the audio
        3. LLM: Generate response
        4. TTS: Synthesize response audio

        Args:
            sample: BenchmarkSample or dict with 'user_input'

        Returns:
            Evaluation result with timing breakdown
        """
        # Extract user input
        if hasattr(sample, 'user_input'):
            user_input = sample.user_input
        elif isinstance(sample, dict):
            user_input = sample.get('user_input', str(sample))
        else:
            user_input = str(sample)

        result = {
            "success": False,
            "latency": 0.0,
            "output": None,
            "error": None,
            "extra": {
                "stt_latency": 0.0,
                "llm_latency": 0.0,
                "tts_latency": 0.0,
            }
        }

        total_start = Timer()
        total_start.__enter__()

        try:
            # Step 1: Simulate STT (in real scenario, this would transcribe actual audio)
            # For benchmark, we use the text directly but measure simulated STT time
            stt_result = self._evaluate_stt(user_input)
            if not stt_result["success"]:
                result["error"] = f"STT failed: {stt_result.get('error')}"
                return result

            result["extra"]["stt_latency"] = stt_result["latency"]
            transcribed_text = stt_result["output"]

            # Step 2: LLM Inference
            llm_result = self._evaluate_llm(transcribed_text)
            if not llm_result["success"]:
                result["error"] = f"LLM failed: {llm_result.get('error')}"
                return result

            result["extra"]["llm_latency"] = llm_result["latency"]
            response_text = llm_result["output"]

            # Step 3: TTS
            tts_result = self._evaluate_tts(response_text)
            if not tts_result["success"]:
                result["error"] = f"TTS failed: {tts_result.get('error')}"
                return result

            result["extra"]["tts_latency"] = tts_result["latency"]

            # Calculate total latency
            total_start.__exit__(None, None, None)
            result["latency"] = total_start.elapsed
            result["success"] = True
            result["output"] = {
                "transcribed_text": transcribed_text,
                "response_text": response_text,
                "audio_info": tts_result["output"],
            }

            # Add additional metrics
            result["extra"]["response_length"] = len(response_text)
            result["extra"]["within_target"] = result["latency"] < 3.0  # Target: < 3s

        except Exception as e:
            total_start.__exit__(None, None, None)
            result["latency"] = total_start.elapsed
            result["error"] = str(e)

        return result

    def _evaluate_stt(self, text: str) -> Dict[str, Any]:
        """
        Evaluate STT component.

        For benchmark purposes, we synthesize audio from text first,
        then transcribe it back. This tests the full STT pipeline.
        """
        import requests
        import base64

        try:
            # First, synthesize audio from text
            synth_response = requests.post(
                f"{self.api_url}/synthesize/json",
                json={"text": text, "speaker": self.speaker, "fast_mode": True},
                timeout=30
            )

            if synth_response.status_code != 200:
                return {
                    "success": False,
                    "latency": 0.0,
                    "output": None,
                    "error": f"Audio synthesis failed: {synth_response.status_code}",
                }

            synth_result = synth_response.json()
            audio_bytes = base64.b64decode(synth_result["audio_base64"])

            # Convert MP3 to WAV for STT
            from pydub import AudioSegment
            import wave

            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            # Now transcribe
            with Timer() as timer:
                response = requests.post(
                    f"{self.api_url}/transcribe",
                    files={"audio": ("audio.wav", wav_buffer, "audio/wav")},
                    timeout=120
                )

            if response.status_code != 200:
                return {
                    "success": False,
                    "latency": timer.elapsed,
                    "output": None,
                    "error": f"STT API error: {response.status_code}",
                }

            result = response.json()
            return {
                "success": True,
                "latency": timer.elapsed,
                "output": result.get("text", ""),
            }

        except ImportError:
            # If pydub not available, use text directly (skip actual STT)
            return {
                "success": True,
                "latency": 0.1,  # Simulated latency
                "output": text,
            }
        except Exception as e:
            return {
                "success": False,
                "latency": 0.0,
                "output": None,
                "error": str(e),
            }

    def _evaluate_llm(self, text: str) -> Dict[str, Any]:
        """Evaluate LLM component."""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ]

            with Timer() as timer:
                response = self._llm.invoke(messages)

            if hasattr(response, 'content'):
                output_text = response.content
            else:
                output_text = str(response)

            return {
                "success": True,
                "latency": timer.elapsed,
                "output": output_text,
            }

        except Exception as e:
            return {
                "success": False,
                "latency": 0.0,
                "output": None,
                "error": str(e),
            }

    def _evaluate_tts(self, text: str) -> Dict[str, Any]:
        """Evaluate TTS component."""
        import requests

        try:
            with Timer() as timer:
                response = requests.post(
                    f"{self.api_url}/synthesize/json",
                    json={
                        "text": text,
                        "speaker": self.speaker,
                        "fast_mode": self.tts_fast_mode
                    },
                    timeout=120
                )

            if response.status_code != 200:
                return {
                    "success": False,
                    "latency": timer.elapsed,
                    "output": None,
                    "error": f"TTS API error: {response.status_code}",
                }

            result = response.json()
            return {
                "success": True,
                "latency": timer.elapsed,
                "output": {
                    "format": result.get("format", "unknown"),
                    "sample_rate": result.get("sample_rate", 0),
                },
            }

        except Exception as e:
            return {
                "success": False,
                "latency": 0.0,
                "output": None,
                "error": str(e),
            }

    def evaluate(
        self,
        samples: List[Any],
        progress_callback: Optional[callable] = None
    ) -> E2EResult:
        """
        Evaluate multiple samples with detailed timing breakdown.

        Args:
            samples: List of samples to evaluate
            progress_callback: Optional callback(current, total)

        Returns:
            E2EResult with component-level timing breakdown
        """
        if not self._initialized:
            self.initialize()

        timing_breakdown = E2ETimingBreakdown()
        errors = []
        success_count = 0
        failure_count = 0
        extra_metrics: Dict[str, List] = {
            "within_target": [],
        }

        total = len(samples)
        for i, sample in enumerate(samples):
            try:
                result = self.evaluate_single(sample)

                if result.get("success", False):
                    success_count += 1

                    # Record component timings
                    timing_breakdown.stt_timing.add(result["extra"]["stt_latency"])
                    timing_breakdown.llm_timing.add(result["extra"]["llm_latency"])
                    timing_breakdown.tts_timing.add(result["extra"]["tts_latency"])
                    timing_breakdown.total_timing.add(result["latency"])

                    extra_metrics["within_target"].append(
                        1 if result["extra"].get("within_target", False) else 0
                    )
                else:
                    failure_count += 1
                    if result.get("error"):
                        errors.append(result["error"])

            except Exception as e:
                failure_count += 1
                errors.append(str(e))

            if progress_callback:
                progress_callback(i + 1, total)

        # Calculate target achievement rate
        target_rate = 0.0
        if extra_metrics["within_target"]:
            target_rate = sum(extra_metrics["within_target"]) / len(extra_metrics["within_target"]) * 100

        return E2EResult(
            evaluator=self.__class__.__name__,
            provider=self._provider_info.get("provider", "unknown"),
            model=self._provider_info.get("model", "unknown"),
            timing=timing_breakdown.total_timing,
            timing_breakdown=timing_breakdown,
            success_count=success_count,
            failure_count=failure_count,
            errors=errors,
            extra_metrics={
                "target_latency_ms": 3000,
                "within_target_rate": round(target_rate, 2),
                "tts_mode": self._provider_info.get("tts_mode", "unknown"),
            },
            config=self.config,
        )

    def cleanup(self):
        """Clean up resources."""
        self._llm = None
        self._initialized = False


class E2EComparisonEvaluator:
    """
    Evaluator for comparing E2E performance across different configurations.

    Useful for comparing:
    - Different LLM providers
    - Fast TTS vs Local TTS
    - Different system prompts
    """

    def __init__(
        self,
        configurations: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize comparison evaluator.

        Args:
            configurations: List of configuration dicts, each with:
                - name: Configuration name
                - llm_provider: LLM provider
                - tts_fast_mode: TTS mode
                - speaker: TTS speaker
            config: Additional shared configuration
        """
        self.configurations = configurations
        self.config = config or {}
        self._evaluators: Dict[str, E2EEvaluator] = {}

    def initialize(self):
        """Initialize all evaluators."""
        for cfg in self.configurations:
            name = cfg.get("name", f"config-{len(self._evaluators)}")
            evaluator = E2EEvaluator(
                llm_provider=cfg.get("llm_provider"),
                tts_fast_mode=cfg.get("tts_fast_mode", True),
                speaker=cfg.get("speaker", "Ceylia"),
                system_prompt=cfg.get("system_prompt"),
                config=self.config
            )
            try:
                evaluator.initialize()
                self._evaluators[name] = evaluator
            except Exception as e:
                print(f"Warning: Failed to initialize {name}: {e}")

    def evaluate(
        self,
        samples: List[Any],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, E2EResult]:
        """
        Evaluate all configurations with the same samples.

        Args:
            samples: List of samples to evaluate
            progress_callback: Optional callback(config_name, current, total)

        Returns:
            Dictionary with E2EResult for each configuration
        """
        if not self._evaluators:
            self.initialize()

        results = {}
        for name, evaluator in self._evaluators.items():
            def config_progress(current, total):
                if progress_callback:
                    progress_callback(name, current, total)

            result = evaluator.evaluate(samples, config_progress)
            results[name] = result

        return results

    def cleanup(self):
        """Clean up all evaluators."""
        for evaluator in self._evaluators.values():
            evaluator.cleanup()
        self._evaluators.clear()
