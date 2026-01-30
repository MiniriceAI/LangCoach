#!/usr/bin/env python3
"""
LangCoach Evaluation CLI

Main entry point for running evaluations from command line.

Usage:
    # Run full evaluation
    python -m evaluation.run_eval

    # Run specific module
    python -m evaluation.run_eval --module llm

    # Run with limited samples
    python -m evaluation.run_eval --samples 10

    # Run comparison
    python -m evaluation.run_eval --compare --providers deepseek ollama

    # Generate report
    python -m evaluation.run_eval --report html
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add paths
eval_path = Path(__file__).parent
project_root = eval_path.parent
src_path = project_root / "src"

sys.path.insert(0, str(eval_path))
sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


def main():
    parser = argparse.ArgumentParser(
        description="LangCoach Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation suite
  python -m evaluation.run_eval

  # Run only LLM evaluation with 10 samples
  python -m evaluation.run_eval --module llm -n 10

  # Compare LLM providers
  python -m evaluation.run_eval --compare --providers deepseek ollama

  # Run E2E evaluation and generate HTML report
  python -m evaluation.run_eval --module e2e --report html

  # Quick test with 5 samples
  python -m evaluation.run_eval -n 5 --quick
        """
    )

    # Module selection
    parser.add_argument(
        "--module", "-m",
        choices=["llm", "tts", "stt", "e2e", "quality", "all"],
        default="all",
        help="Module to evaluate (default: all)"
    )

    # Sample configuration
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all 100)"
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: run with 5 samples only"
    )

    # Provider configuration
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="LLM provider (ollama, deepseek, openai)"
    )

    # Quality evaluation options
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        help="Judge LLM provider for quality evaluation (openai, anthropic, ollama)"
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model for quality evaluation (e.g., gpt-4o, claude-3-opus)"
    )

    parser.add_argument(
        "--eval-date",
        type=str,
        default=None,
        help="Date for quality evaluation (YYYY-MM-DD). Default: yesterday"
    )

    parser.add_argument(
        "--dqi-threshold",
        type=float,
        default=85.0,
        help="DQI threshold for quality alerts (default: 85.0)"
    )

    # Comparison mode
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Run comparison evaluation"
    )

    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=None,
        help="Providers to compare (e.g., deepseek ollama)"
    )

    # TTS configuration
    parser.add_argument(
        "--tts-mode",
        choices=["fast", "local"],
        default="fast",
        help="TTS mode (fast=Edge-TTS, local=Orpheus)"
    )

    # Output configuration
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )

    parser.add_argument(
        "--report", "-r",
        type=str,
        nargs="*",
        default=None,
        help="Generate report in format(s): json, md, html, txt"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--silent", "-s",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        args.samples = 5

    verbose = not args.silent

    # Print header
    if verbose:
        print("\n" + "=" * 60)
        print(" LangCoach Evaluation Framework")
        print(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 60)

    # Run evaluation
    if args.compare:
        from runners.comparison_runner import ComparisonRunner

        runner = ComparisonRunner(output_dir=args.output)

        if args.module == "llm":
            providers = args.providers or ["deepseek", "ollama"]
            results = {"llm_comparison": runner.compare_llm_providers(
                providers, args.samples, verbose
            )}
        elif args.module == "tts":
            results = {"tts_comparison": runner.compare_tts_modes(
                ["fast", "local"], args.samples, verbose
            )}
        elif args.module == "e2e":
            providers = args.providers or ["deepseek", "ollama"]
            configs = [
                {"name": f"{p}-fast", "llm_provider": p, "tts_fast_mode": True}
                for p in providers
            ]
            results = {"e2e_comparison": runner.compare_e2e_configurations(
                configs, args.samples, verbose
            )}
        else:
            results = runner.run_standard_comparison(args.samples, verbose)

    else:
        from runners.evaluation_runner import EvaluationRunner

        runner = EvaluationRunner(output_dir=args.output)

        if args.module == "llm":
            result = runner.run_llm_evaluation(args.samples, args.provider, verbose)
            results = {"llm": result}
        elif args.module == "tts":
            result = runner.run_tts_evaluation(
                args.samples, args.tts_mode == "fast", verbose
            )
            results = {"tts": result}
        elif args.module == "stt":
            result = runner.run_stt_evaluation(args.samples, verbose)
            results = {"stt": result}
        elif args.module == "e2e":
            result = runner.run_e2e_evaluation(
                args.samples, args.provider, args.tts_mode == "fast", verbose
            )
            results = {"e2e": result}
        elif args.module == "quality":
            # Run quality evaluation
            from runners.quality_runner import QualityEvaluationRunner

            quality_runner = QualityEvaluationRunner(
                output_dir=args.output,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
            )
            result = quality_runner.run_daily_evaluation(
                date=args.eval_date,
                n_samples=args.samples or 50,
                dqi_threshold=args.dqi_threshold,
                verbose=verbose
            )
            results = {"quality": result}
        else:
            results = runner.run_full_evaluation(args.samples, args.provider, verbose)

    # Generate reports
    if args.report is not None:
        from reports.report_generator import ReportGenerator

        formats = args.report if args.report else ["json", "md", "html"]
        generator = ReportGenerator(output_dir=args.output or "evaluation/reports")

        report_name = f"{'comparison' if args.compare else 'evaluation'}_{args.module}"
        saved = generator.save_report(results, report_name, formats)

        if verbose:
            print("\n" + "-" * 60)
            print("Reports saved:")
            for fmt, path in saved.items():
                print(f"  {fmt}: {path}")

    # Save raw results
    if args.compare:
        filepath = runner.save_results(results, f"comparison_{args.module}")
    else:
        filepath = runner.save_results(results, args.module)

    if verbose:
        print(f"\nRaw results saved to: {filepath}")
        print("\n" + "=" * 60)
        print(" Evaluation Complete")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
