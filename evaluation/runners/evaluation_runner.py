"""
Evaluation Runner

Main entry point for running evaluations.
Supports running individual module evaluations or full E2E pipeline.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add paths
eval_path = Path(__file__).parent.parent
src_path = eval_path.parent / "src"
sys.path.insert(0, str(eval_path))
sys.path.insert(0, str(src_path))

from benchmark.dataset import BenchmarkDataset, BenchmarkSample
from evaluators.base import EvaluationResult
from evaluators.llm_evaluator import LLMEvaluator
from evaluators.tts_evaluator import TTSEvaluator
from evaluators.stt_evaluator import STTEvaluator
from evaluators.e2e_evaluator import E2EEvaluator, E2EResult


class EvaluationRunner:
    """
    Main evaluation runner.

    Supports running:
    - Individual module evaluations (STT, LLM, TTS)
    - Full E2E pipeline evaluation
    - Custom evaluation configurations
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize evaluation runner.

        Args:
            output_dir: Directory for saving results
            config: Additional configuration
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation/reports/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._dataset = None

    def load_dataset(self, n_samples: Optional[int] = None) -> List[BenchmarkSample]:
        """
        Load benchmark dataset.

        Args:
            n_samples: Number of samples to load. If None, loads all.

        Returns:
            List of BenchmarkSample objects
        """
        if self._dataset is None:
            self._dataset = BenchmarkDataset()

        return self._dataset.get_samples(n_samples)

    def run_llm_evaluation(
        self,
        n_samples: Optional[int] = None,
        provider: Optional[str] = None,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Run LLM evaluation.

        Args:
            n_samples: Number of samples to evaluate
            provider: LLM provider name
            verbose: Print progress

        Returns:
            EvaluationResult
        """
        samples = self.load_dataset(n_samples)

        if verbose:
            print(f"\n{'='*60}")
            print(f"LLM Evaluation")
            print(f"Provider: {provider or 'auto'}")
            print(f"Samples: {len(samples)}")
            print(f"{'='*60}\n")

        evaluator = LLMEvaluator(provider=provider, config=self.config)
        evaluator.initialize()

        def progress(current, total):
            if verbose:
                print(f"\rProgress: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)

        result = evaluator.evaluate(samples, progress)

        if verbose:
            print(f"\n\nResults:")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Mean Latency: {result.timing.mean*1000:.0f}ms")
            print(f"  P95 Latency: {result.timing.p95*1000:.0f}ms")

        evaluator.cleanup()
        return result

    def run_tts_evaluation(
        self,
        n_samples: Optional[int] = None,
        fast_mode: bool = True,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Run TTS evaluation.

        Args:
            n_samples: Number of samples to evaluate
            fast_mode: Use Edge-TTS fast mode
            verbose: Print progress

        Returns:
            EvaluationResult
        """
        samples = self.load_dataset(n_samples)

        if verbose:
            print(f"\n{'='*60}")
            print(f"TTS Evaluation")
            print(f"Mode: {'Edge-TTS (fast)' if fast_mode else 'Orpheus (local)'}")
            print(f"Samples: {len(samples)}")
            print(f"{'='*60}\n")

        evaluator = TTSEvaluator(
            mode="api",
            fast_mode=fast_mode,
            config=self.config
        )
        evaluator.initialize()

        def progress(current, total):
            if verbose:
                print(f"\rProgress: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)

        result = evaluator.evaluate(samples, progress)

        if verbose:
            print(f"\n\nResults:")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Mean Latency: {result.timing.mean*1000:.0f}ms")
            print(f"  P95 Latency: {result.timing.p95*1000:.0f}ms")

        evaluator.cleanup()
        return result

    def run_stt_evaluation(
        self,
        n_samples: Optional[int] = None,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Run STT evaluation.

        Args:
            n_samples: Number of samples to evaluate
            verbose: Print progress

        Returns:
            EvaluationResult
        """
        samples = self.load_dataset(n_samples)

        if verbose:
            print(f"\n{'='*60}")
            print(f"STT Evaluation")
            print(f"Samples: {len(samples)}")
            print(f"{'='*60}\n")

        evaluator = STTEvaluator(mode="api", config=self.config)
        evaluator.initialize()

        def progress(current, total):
            if verbose:
                print(f"\rProgress: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)

        result = evaluator.evaluate(samples, progress)

        if verbose:
            print(f"\n\nResults:")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Mean Latency: {result.timing.mean*1000:.0f}ms")
            print(f"  P95 Latency: {result.timing.p95*1000:.0f}ms")
            if "wer_mean" in result.extra_metrics:
                print(f"  Mean WER: {result.extra_metrics['wer_mean']:.2%}")

        evaluator.cleanup()
        return result

    def run_e2e_evaluation(
        self,
        n_samples: Optional[int] = None,
        llm_provider: Optional[str] = None,
        tts_fast_mode: bool = True,
        verbose: bool = True
    ) -> E2EResult:
        """
        Run E2E pipeline evaluation.

        Args:
            n_samples: Number of samples to evaluate
            llm_provider: LLM provider name
            tts_fast_mode: Use Edge-TTS fast mode
            verbose: Print progress

        Returns:
            E2EResult with timing breakdown
        """
        samples = self.load_dataset(n_samples)

        if verbose:
            print(f"\n{'='*60}")
            print(f"E2E Pipeline Evaluation")
            print(f"LLM Provider: {llm_provider or 'auto'}")
            print(f"TTS Mode: {'Edge-TTS (fast)' if tts_fast_mode else 'Orpheus (local)'}")
            print(f"Samples: {len(samples)}")
            print(f"Target Latency: < 3000ms")
            print(f"{'='*60}\n")

        evaluator = E2EEvaluator(
            llm_provider=llm_provider,
            tts_fast_mode=tts_fast_mode,
            config=self.config
        )
        evaluator.initialize()

        def progress(current, total):
            if verbose:
                print(f"\rProgress: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)

        result = evaluator.evaluate(samples, progress)

        if verbose:
            print(f"\n\nResults:")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Within Target (<3s): {result.extra_metrics.get('within_target_rate', 0):.1f}%")
            print(f"\n  Timing Breakdown:")
            print(f"    STT Mean: {result.timing_breakdown.stt_timing.mean*1000:.0f}ms")
            print(f"    LLM Mean: {result.timing_breakdown.llm_timing.mean*1000:.0f}ms")
            print(f"    TTS Mean: {result.timing_breakdown.tts_timing.mean*1000:.0f}ms")
            print(f"    Total Mean: {result.timing_breakdown.total_timing.mean*1000:.0f}ms")
            print(f"    Total P95: {result.timing_breakdown.total_timing.p95*1000:.0f}ms")

        evaluator.cleanup()
        return result

    def run_full_evaluation(
        self,
        n_samples: Optional[int] = None,
        llm_provider: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite (all modules + E2E).

        Args:
            n_samples: Number of samples to evaluate
            llm_provider: LLM provider name
            verbose: Print progress

        Returns:
            Dictionary with all results
        """
        results = {}

        # LLM Evaluation
        results["llm"] = self.run_llm_evaluation(n_samples, llm_provider, verbose)

        # TTS Evaluation (fast mode)
        results["tts_fast"] = self.run_tts_evaluation(n_samples, fast_mode=True, verbose=verbose)

        # STT Evaluation
        results["stt"] = self.run_stt_evaluation(n_samples, verbose)

        # E2E Evaluation
        results["e2e"] = self.run_e2e_evaluation(n_samples, llm_provider, verbose=verbose)

        return results

    def save_results(
        self,
        results: Dict[str, Any],
        name: Optional[str] = None
    ) -> str:
        """
        Save evaluation results to file.

        Args:
            results: Results dictionary
            name: Optional name for the results file

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json" if name else f"evaluation_{timestamp}.json"
        filepath = self.output_dir / filename

        # Convert results to serializable format
        serializable = {}
        for key, value in results.items():
            if hasattr(value, 'to_dict'):
                serializable[key] = value.to_dict()
            elif isinstance(value, dict):
                serializable[key] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        return str(filepath)


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="LangCoach Evaluation Runner")
    parser.add_argument(
        "--module",
        choices=["llm", "tts", "stt", "e2e", "all"],
        default="all",
        help="Module to evaluate"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all 100)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider (ollama, deepseek, openai)"
    )
    parser.add_argument(
        "--tts-mode",
        choices=["fast", "local"],
        default="fast",
        help="TTS mode (fast=Edge-TTS, local=Orpheus)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    runner = EvaluationRunner(output_dir=args.output)
    verbose = not args.quiet

    if args.module == "llm":
        result = runner.run_llm_evaluation(args.samples, args.provider, verbose)
        results = {"llm": result}
    elif args.module == "tts":
        result = runner.run_tts_evaluation(args.samples, args.tts_mode == "fast", verbose)
        results = {"tts": result}
    elif args.module == "stt":
        result = runner.run_stt_evaluation(args.samples, verbose)
        results = {"stt": result}
    elif args.module == "e2e":
        result = runner.run_e2e_evaluation(
            args.samples, args.provider, args.tts_mode == "fast", verbose
        )
        results = {"e2e": result}
    else:
        results = runner.run_full_evaluation(args.samples, args.provider, verbose)

    # Save results
    filepath = runner.save_results(results, args.module)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
