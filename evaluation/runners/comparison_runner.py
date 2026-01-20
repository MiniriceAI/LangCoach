"""
Comparison Runner

Specialized runner for A/B testing and comparison evaluations.
Supports comparing different LLM providers, TTS modes, etc.
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
from evaluators.llm_evaluator import LLMEvaluator, LLMComparisonEvaluator
from evaluators.tts_evaluator import TTSEvaluator, TTSComparisonEvaluator
from evaluators.e2e_evaluator import E2EEvaluator, E2EComparisonEvaluator, E2EResult


class ComparisonRunner:
    """
    Runner for comparison evaluations.

    Supports:
    - LLM provider comparison (e.g., DeepSeek vs Ollama)
    - TTS mode comparison (Edge-TTS vs Orpheus)
    - E2E configuration comparison
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize comparison runner.

        Args:
            output_dir: Directory for saving results
            config: Additional configuration
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation/reports/comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._dataset = None

    def load_dataset(self, n_samples: Optional[int] = None) -> List[BenchmarkSample]:
        """Load benchmark dataset."""
        if self._dataset is None:
            self._dataset = BenchmarkDataset()
        return self._dataset.get_samples(n_samples)

    def compare_llm_providers(
        self,
        providers: List[str],
        n_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple LLM providers.

        Args:
            providers: List of provider names (e.g., ["ollama", "deepseek"])
            n_samples: Number of samples to evaluate
            verbose: Print progress

        Returns:
            Dictionary mapping provider name to EvaluationResult
        """
        samples = self.load_dataset(n_samples)

        if verbose:
            print(f"\n{'='*60}")
            print(f"LLM Provider Comparison")
            print(f"Providers: {', '.join(providers)}")
            print(f"Samples: {len(samples)}")
            print(f"{'='*60}\n")

        evaluator = LLMComparisonEvaluator(providers=providers, config=self.config)

        def progress(provider, current, total):
            if verbose:
                print(f"\r[{provider}] Progress: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)

        results = evaluator.evaluate(samples, progress)

        if verbose:
            print(f"\n\n{'='*60}")
            print("Comparison Results:")
            print(f"{'='*60}")
            for provider, result in results.items():
                print(f"\n{provider}:")
                print(f"  Success Rate: {result.success_rate:.1f}%")
                print(f"  Mean Latency: {result.timing.mean*1000:.0f}ms")
                print(f"  P95 Latency: {result.timing.p95*1000:.0f}ms")

            # Print winner
            if len(results) > 1:
                fastest = min(results.items(), key=lambda x: x[1].timing.mean)
                print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1].timing.mean*1000:.0f}ms mean)")

        evaluator.cleanup()
        return results

    def compare_tts_modes(
        self,
        modes: List[str] = None,
        n_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, EvaluationResult]:
        """
        Compare TTS modes.

        Args:
            modes: List of modes (e.g., ["fast", "local"])
            n_samples: Number of samples to evaluate
            verbose: Print progress

        Returns:
            Dictionary mapping mode to EvaluationResult
        """
        modes = modes or ["fast", "local"]
        samples = self.load_dataset(n_samples)

        if verbose:
            print(f"\n{'='*60}")
            print(f"TTS Mode Comparison")
            print(f"Modes: {', '.join(modes)}")
            print(f"Samples: {len(samples)}")
            print(f"{'='*60}\n")

        evaluator = TTSComparisonEvaluator(modes=modes, config=self.config)

        def progress(mode, current, total):
            if verbose:
                print(f"\r[{mode}] Progress: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)

        results = evaluator.evaluate(samples, progress)

        if verbose:
            print(f"\n\n{'='*60}")
            print("Comparison Results:")
            print(f"{'='*60}")
            for mode, result in results.items():
                print(f"\n{mode}:")
                print(f"  Success Rate: {result.success_rate:.1f}%")
                print(f"  Mean Latency: {result.timing.mean*1000:.0f}ms")
                print(f"  P95 Latency: {result.timing.p95*1000:.0f}ms")
                if "rtf_mean" in result.extra_metrics:
                    print(f"  Mean RTF: {result.extra_metrics['rtf_mean']:.2f}")

            if len(results) > 1:
                fastest = min(results.items(), key=lambda x: x[1].timing.mean)
                print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1].timing.mean*1000:.0f}ms mean)")

        evaluator.cleanup()
        return results

    def compare_e2e_configurations(
        self,
        configurations: List[Dict[str, Any]],
        n_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, E2EResult]:
        """
        Compare E2E pipeline configurations.

        Args:
            configurations: List of configuration dicts with:
                - name: Configuration name
                - llm_provider: LLM provider
                - tts_fast_mode: TTS mode
            n_samples: Number of samples to evaluate
            verbose: Print progress

        Returns:
            Dictionary mapping config name to E2EResult
        """
        samples = self.load_dataset(n_samples)

        if verbose:
            print(f"\n{'='*60}")
            print(f"E2E Configuration Comparison")
            print(f"Configurations: {len(configurations)}")
            print(f"Samples: {len(samples)}")
            print(f"Target Latency: < 3000ms")
            print(f"{'='*60}\n")

        evaluator = E2EComparisonEvaluator(configurations=configurations, config=self.config)

        def progress(config_name, current, total):
            if verbose:
                print(f"\r[{config_name}] Progress: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)

        results = evaluator.evaluate(samples, progress)

        if verbose:
            print(f"\n\n{'='*60}")
            print("Comparison Results:")
            print(f"{'='*60}")
            for name, result in results.items():
                print(f"\n{name}:")
                print(f"  Success Rate: {result.success_rate:.1f}%")
                print(f"  Within Target (<3s): {result.extra_metrics.get('within_target_rate', 0):.1f}%")
                print(f"  Mean Total Latency: {result.timing_breakdown.total_timing.mean*1000:.0f}ms")
                print(f"  P95 Total Latency: {result.timing_breakdown.total_timing.p95*1000:.0f}ms")
                print(f"  Breakdown:")
                print(f"    STT: {result.timing_breakdown.stt_timing.mean*1000:.0f}ms")
                print(f"    LLM: {result.timing_breakdown.llm_timing.mean*1000:.0f}ms")
                print(f"    TTS: {result.timing_breakdown.tts_timing.mean*1000:.0f}ms")

            if len(results) > 1:
                fastest = min(results.items(), key=lambda x: x[1].timing_breakdown.total_timing.mean)
                best_target = max(results.items(), key=lambda x: x[1].extra_metrics.get('within_target_rate', 0))
                print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1].timing_breakdown.total_timing.mean*1000:.0f}ms mean)")
                print(f"🎯 Best Target Rate: {best_target[0]} ({best_target[1].extra_metrics.get('within_target_rate', 0):.1f}%)")

        evaluator.cleanup()
        return results

    def run_standard_comparison(
        self,
        n_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run standard comparison suite.

        Compares:
        - LLM: DeepSeek vs Ollama (if both available)
        - TTS: Edge-TTS vs Orpheus
        - E2E: Multiple configurations

        Args:
            n_samples: Number of samples to evaluate
            verbose: Print progress

        Returns:
            Dictionary with all comparison results
        """
        results = {}

        # Check available LLM providers
        try:
            from agents.llm_factory import list_available_providers
            available_providers = list_available_providers()
        except:
            available_providers = []

        # LLM Comparison
        llm_providers = [p for p in ["deepseek", "ollama"] if p in available_providers]
        if len(llm_providers) >= 2:
            results["llm_comparison"] = self.compare_llm_providers(
                llm_providers, n_samples, verbose
            )
        elif len(llm_providers) == 1:
            if verbose:
                print(f"\nSkipping LLM comparison: only {llm_providers[0]} available")

        # TTS Comparison (fast mode only for now, local requires GPU)
        results["tts_comparison"] = self.compare_tts_modes(
            ["fast"], n_samples, verbose
        )

        # E2E Comparison
        e2e_configs = []
        for provider in llm_providers[:2]:  # Max 2 providers
            e2e_configs.append({
                "name": f"{provider}-fast",
                "llm_provider": provider,
                "tts_fast_mode": True,
            })

        if e2e_configs:
            results["e2e_comparison"] = self.compare_e2e_configurations(
                e2e_configs, n_samples, verbose
            )

        return results

    def save_results(
        self,
        results: Dict[str, Any],
        name: Optional[str] = None
    ) -> str:
        """
        Save comparison results to file.

        Args:
            results: Results dictionary
            name: Optional name for the results file

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json" if name else f"comparison_{timestamp}.json"
        filepath = self.output_dir / filename

        # Convert results to serializable format
        serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable[key] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            elif hasattr(value, 'to_dict'):
                serializable[key] = value.to_dict()
            else:
                serializable[key] = value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        return str(filepath)


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="LangCoach Comparison Runner")
    parser.add_argument(
        "--type",
        choices=["llm", "tts", "e2e", "all"],
        default="all",
        help="Type of comparison to run"
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=None,
        help="LLM providers to compare (e.g., deepseek ollama)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all 100)"
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

    runner = ComparisonRunner(output_dir=args.output)
    verbose = not args.quiet

    if args.type == "llm":
        providers = args.providers or ["deepseek", "ollama"]
        results = {"llm": runner.compare_llm_providers(providers, args.samples, verbose)}
    elif args.type == "tts":
        results = {"tts": runner.compare_tts_modes(["fast", "local"], args.samples, verbose)}
    elif args.type == "e2e":
        providers = args.providers or ["deepseek", "ollama"]
        configs = [
            {"name": f"{p}-fast", "llm_provider": p, "tts_fast_mode": True}
            for p in providers
        ]
        results = {"e2e": runner.compare_e2e_configurations(configs, args.samples, verbose)}
    else:
        results = runner.run_standard_comparison(args.samples, verbose)

    # Save results
    filepath = runner.save_results(results, args.type)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
