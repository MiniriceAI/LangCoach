"""
Base Evaluator Module

Provides base classes and common utilities for all evaluators.
"""

import time
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class TimingMetrics:
    """Timing metrics for a single evaluation run."""
    # Individual latencies in seconds
    latencies: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of measurements."""
        return len(self.latencies)

    @property
    def mean(self) -> float:
        """Mean latency in seconds."""
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)

    @property
    def median(self) -> float:
        """Median latency in seconds."""
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies)

    @property
    def min(self) -> float:
        """Minimum latency in seconds."""
        if not self.latencies:
            return 0.0
        return min(self.latencies)

    @property
    def max(self) -> float:
        """Maximum latency in seconds."""
        if not self.latencies:
            return 0.0
        return max(self.latencies)

    @property
    def std(self) -> float:
        """Standard deviation of latencies."""
        if len(self.latencies) < 2:
            return 0.0
        return statistics.stdev(self.latencies)

    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        return self.median

    @property
    def p90(self) -> float:
        """90th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.9)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p95(self) -> float:
        """95th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99(self) -> float:
        """99th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def add(self, latency: float):
        """Add a latency measurement."""
        self.latencies.append(latency)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "count": self.count,
            "mean_ms": round(self.mean * 1000, 2),
            "median_ms": round(self.median * 1000, 2),
            "min_ms": round(self.min * 1000, 2),
            "max_ms": round(self.max * 1000, 2),
            "std_ms": round(self.std * 1000, 2),
            "p50_ms": round(self.p50 * 1000, 2),
            "p90_ms": round(self.p90 * 1000, 2),
            "p95_ms": round(self.p95 * 1000, 2),
            "p99_ms": round(self.p99 * 1000, 2),
        }


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""
    # Evaluator name
    evaluator: str
    # Provider/model being evaluated
    provider: str
    model: str
    # Timing metrics
    timing: TimingMetrics
    # Success/failure counts
    success_count: int = 0
    failure_count: int = 0
    # Error messages
    errors: List[str] = field(default_factory=list)
    # Additional metrics (e.g., accuracy, quality scores)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        """Total number of evaluations."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "evaluator": self.evaluator,
            "provider": self.provider,
            "model": self.model,
            "timing": self.timing.to_dict(),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_count": self.total_count,
            "success_rate": round(self.success_rate, 2),
            "errors": self.errors[:10],  # Limit errors in report
            "extra_metrics": self.extra_metrics,
            "timestamp": self.timestamp,
            "config": self.config,
        }


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.perf_counter()
        return end - self.start_time


class BaseEvaluator(ABC):
    """
    Base class for all evaluators.

    Subclasses must implement:
    - evaluate_single(): Evaluate a single sample
    - get_provider_info(): Get provider/model information
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.

        Args:
            config: Optional configuration dictionary
                - warmup_runs: Number of warmup runs to exclude GPU model loading time (default: 1)
                - skip_warmup: Skip warmup entirely (default: False)
        """
        self.config = config or {}
        self._initialized = False
        self._warmed_up = False
        self._warmup_runs = self.config.get("warmup_runs", 1)
        self._skip_warmup = self.config.get("skip_warmup", False)

    @abstractmethod
    def initialize(self):
        """Initialize the evaluator (load models, etc.)."""
        pass

    def warmup(self, sample: Any) -> None:
        """
        Perform warmup runs to exclude GPU model loading time.
        
        This is important for local models (STT/LLM/TTS) where the first
        inference includes model loading overhead.
        
        Args:
            sample: A sample to use for warmup
        """
        if self._warmed_up or self._skip_warmup:
            return
            
        if not self._initialized:
            self.initialize()
            
        for i in range(self._warmup_runs):
            try:
                # Run inference without recording metrics
                self.evaluate_single(sample)
            except Exception:
                pass  # Ignore warmup errors
                
        self._warmed_up = True

    @abstractmethod
    def evaluate_single(self, sample: Any) -> Dict[str, Any]:
        """
        Evaluate a single sample.

        Args:
            sample: The sample to evaluate

        Returns:
            Dictionary with:
                - success: bool
                - latency: float (seconds)
                - output: Any (the result)
                - error: Optional[str]
                - extra: Optional[Dict] (additional metrics)
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, str]:
        """
        Get provider information.

        Returns:
            Dictionary with 'provider' and 'model' keys
        """
        pass

    def evaluate(
        self,
        samples: List[Any],
        progress_callback: Optional[callable] = None
    ) -> EvaluationResult:
        """
        Evaluate multiple samples.

        Args:
            samples: List of samples to evaluate
            progress_callback: Optional callback(current, total) for progress

        Returns:
            EvaluationResult with aggregated metrics
        """
        if not self._initialized:
            self.initialize()

        # Perform warmup to exclude GPU model loading time
        if samples and not self._warmed_up and not self._skip_warmup:
            self.warmup(samples[0])

        provider_info = self.get_provider_info()
        timing = TimingMetrics()
        errors = []
        success_count = 0
        failure_count = 0
        extra_metrics: Dict[str, List] = {}

        total = len(samples)
        for i, sample in enumerate(samples):
            try:
                result = self.evaluate_single(sample)

                if result.get("success", False):
                    success_count += 1
                    timing.add(result.get("latency", 0.0))
                else:
                    failure_count += 1
                    if result.get("error"):
                        errors.append(result["error"])

                # Collect extra metrics
                if result.get("extra"):
                    for key, value in result["extra"].items():
                        if key not in extra_metrics:
                            extra_metrics[key] = []
                        extra_metrics[key].append(value)

            except Exception as e:
                failure_count += 1
                errors.append(str(e))

            if progress_callback:
                progress_callback(i + 1, total)

        # Aggregate extra metrics
        aggregated_extra = {}
        for key, values in extra_metrics.items():
            if all(isinstance(v, (int, float)) for v in values):
                aggregated_extra[f"{key}_mean"] = statistics.mean(values)
                aggregated_extra[f"{key}_min"] = min(values)
                aggregated_extra[f"{key}_max"] = max(values)

        return EvaluationResult(
            evaluator=self.__class__.__name__,
            provider=provider_info.get("provider", "unknown"),
            model=provider_info.get("model", "unknown"),
            timing=timing,
            success_count=success_count,
            failure_count=failure_count,
            errors=errors,
            extra_metrics=aggregated_extra,
            config=self.config,
        )

    def cleanup(self):
        """Clean up resources. Override if needed."""
        pass
