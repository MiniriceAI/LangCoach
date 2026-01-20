"""
LLM Evaluator Module

Evaluates LLM inference performance including:
- Response latency (Time to First Token and Total Time)
- Response quality
- Token throughput
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from .base import BaseEvaluator, Timer, TimingMetrics


# Default system prompt for evaluation (casual conversation style)
DEFAULT_SYSTEM_PROMPT = """You are a friendly English tutor having a casual conversation with a student.
Keep your responses natural, concise (under 50 words), and conversational.
Respond as if you're chatting with a friend who is practicing English.
Do not include any formatting, bullet points, or structured responses.
Just have a natural conversation."""


class LLMEvaluator(BaseEvaluator):
    """
    Evaluator for LLM inference performance.

    Supports multiple providers: ollama, deepseek, openai
    Configurable via environment variables or constructor parameters.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LLM evaluator.

        Args:
            provider: LLM provider name (ollama, deepseek, openai).
                     If None, uses environment variable or auto-selects.
            model: Model name. If None, uses provider default.
            system_prompt: System prompt for conversations.
            config: Additional configuration.
        """
        super().__init__(config)
        self._provider_name = provider
        self._model_name = model
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._llm = None
        self._provider_info: Dict[str, str] = {}

    def initialize(self):
        """Initialize the LLM."""
        if self._initialized:
            return

        # Import here to avoid circular imports
        from agents.llm_factory import create_llm, get_current_provider_info

        # Create LLM instance
        self._llm = create_llm(self._provider_name)

        # Get provider info
        info = get_current_provider_info()
        self._provider_info = {
            "provider": self._provider_name or info.get("provider", "unknown"),
            "model": self._model_name or info.get("model", "unknown"),
        }

        self._initialized = True

    def get_provider_info(self) -> Dict[str, str]:
        """Get provider information."""
        return self._provider_info

    def evaluate_single(self, sample: Any) -> Dict[str, Any]:
        """
        Evaluate a single sample.

        Args:
            sample: BenchmarkSample or dict with 'user_input' key

        Returns:
            Evaluation result dictionary
        """
        # Extract user input
        if hasattr(sample, 'user_input'):
            user_input = sample.user_input
        elif isinstance(sample, dict):
            user_input = sample.get('user_input', str(sample))
        else:
            user_input = str(sample)

        try:
            # Build messages
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_input},
            ]

            # Time the inference
            with Timer() as timer:
                response = self._llm.invoke(messages)

            # Extract response content
            if hasattr(response, 'content'):
                output_text = response.content
            else:
                output_text = str(response)

            # Calculate metrics
            output_length = len(output_text)
            word_count = len(output_text.split())

            return {
                "success": True,
                "latency": timer.elapsed,
                "output": output_text,
                "extra": {
                    "output_length": output_length,
                    "word_count": word_count,
                    "chars_per_second": output_length / timer.elapsed if timer.elapsed > 0 else 0,
                }
            }

        except Exception as e:
            return {
                "success": False,
                "latency": 0.0,
                "output": None,
                "error": str(e),
            }

    def cleanup(self):
        """Clean up resources."""
        self._llm = None
        self._initialized = False


class LLMComparisonEvaluator:
    """
    Evaluator for comparing multiple LLM providers.

    Runs the same benchmark across different providers and generates
    comparison metrics.
    """

    def __init__(
        self,
        providers: List[str],
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize comparison evaluator.

        Args:
            providers: List of provider names to compare
            system_prompt: System prompt for all providers
            config: Additional configuration
        """
        self.providers = providers
        self.system_prompt = system_prompt
        self.config = config or {}
        self._evaluators: Dict[str, LLMEvaluator] = {}

    def initialize(self):
        """Initialize all evaluators."""
        for provider in self.providers:
            evaluator = LLMEvaluator(
                provider=provider,
                system_prompt=self.system_prompt,
                config=self.config
            )
            try:
                evaluator.initialize()
                self._evaluators[provider] = evaluator
            except Exception as e:
                print(f"Warning: Failed to initialize {provider}: {e}")

    def evaluate(
        self,
        samples: List[Any],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all providers with the same samples.

        Args:
            samples: List of samples to evaluate
            progress_callback: Optional callback(provider, current, total)

        Returns:
            Dictionary with results for each provider
        """
        if not self._evaluators:
            self.initialize()

        results = {}
        for provider, evaluator in self._evaluators.items():
            def provider_progress(current, total):
                if progress_callback:
                    progress_callback(provider, current, total)

            result = evaluator.evaluate(samples, provider_progress)
            results[provider] = result

        return results

    def cleanup(self):
        """Clean up all evaluators."""
        for evaluator in self._evaluators.values():
            evaluator.cleanup()
        self._evaluators.clear()
