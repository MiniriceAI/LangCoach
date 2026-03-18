"""
Quality Evaluation Runner

Implements the "Daily Evals" pipeline from EVALUATION_PLAN.md:
1. Sample 50 conversations from previous day's logs
2. Run LLM-as-a-Judge evaluation
3. Calculate Daily Quality Index (DQI)
4. Generate alerts if DQI < 85%
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
eval_path = Path(__file__).parent.parent
src_path = eval_path.parent / "src"
sys.path.insert(0, str(eval_path))
sys.path.insert(0, str(src_path))

from benchmark.conversation_logs import (
    ConversationLogManager,
    ConversationLog,
    sample_daily_conversations,
)
from evaluators.quality_evaluator import (
    QualityEvaluator,
    QualityMetrics,
    calculate_dqi,
    check_dqi_threshold,
)
from evaluators.base import EvaluationResult


class QualityEvaluationRunner:
    """
    Runner for quality evaluation using LLM-as-a-Judge.

    Implements the Daily Evals pipeline for monitoring conversation quality.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize quality evaluation runner.

        Args:
            log_dir: Directory containing conversation logs
            output_dir: Directory for saving results
            judge_provider: LLM provider for judge
            judge_model: Model to use as judge
            config: Additional configuration
        """
        # Load environment configuration
        self._load_env_config()
        
        # Validate configuration
        self._validate_config(judge_provider, judge_model)
        
        self.log_manager = ConversationLogManager(log_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation/reports/quality")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.config = config or {}

        # Load environment configuration
        self._load_env_config()

    def _validate_config(self, judge_provider: Optional[str], judge_model: Optional[str]):
        """Validate configuration parameters."""
        valid_providers = {"openai", "anthropic", "ollama", "deepseek"}
        provider = judge_provider or os.getenv("JUDGE_PROVIDER", "openai")
        
        if provider not in valid_providers:
            raise ValueError(f"Invalid judge_provider: {provider}. Must be one of {valid_providers}")
        
        # Validate provider-model combinations
        if provider == "openai" and judge_model and not judge_model.startswith(("gpt-", "o1-")):
            raise ValueError(f"Invalid OpenAI model: {judge_model}")
        elif provider == "anthropic" and judge_model and not judge_model.startswith("claude-"):
            raise ValueError(f"Invalid Anthropic model: {judge_model}")

    def _load_env_config(self):
        """Load configuration from environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Loaded environment configuration from .env file")
        except ImportError:
            logger.warning("python-dotenv not installed. Using system environment variables only.")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")

    def run_daily_evaluation(
        self,
        date: Optional[str] = None,
        n_samples: int = 50,
        dqi_threshold: float = 85.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run daily quality evaluation.

        This implements the "Daily Evals" pipeline:
        1. Sample conversations from specified date (or yesterday)
        2. Extract turns for evaluation
        3. Run LLM-as-a-Judge on each turn
        4. Calculate DQI
        5. Check threshold and generate alerts

        Args:
            date: Date to evaluate (YYYY-MM-DD). If None, uses yesterday.
            n_samples: Number of conversations to sample
            dqi_threshold: Minimum acceptable DQI (default: 85%)
            verbose: Print progress

        Returns:
            Dictionary with evaluation results and DQI
        """
        # Determine evaluation date
        if date is None:
            eval_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            eval_date = date

        if verbose:
            logger.info("="*60)
            logger.info("Daily Quality Evaluation")
            logger.info(f"Date: {eval_date}")
            logger.info(f"Sample Size: {n_samples} conversations") 
            logger.info(f"DQI Threshold: {dqi_threshold}%")
            logger.info("="*60)

        # Step 1: Sample conversations
        if verbose:
            logger.info(f"Sampling conversations from {eval_date}...")

        try:
            conversations = self.log_manager.sample_conversations(
                eval_date,
                n=n_samples,
                random_seed=42  # For reproducibility
            )
        except Exception as e:
            logger.error(f"Error sampling conversations: {e}")
            if verbose:
                logger.error(f"Error sampling conversations: {e}")
            return {
                "date": eval_date,
                "error": f"Failed to sample conversations: {str(e)}",
                "dqi": 0.0,
                "alert": None,
            }

        if not conversations:
            if verbose:
                logger.warning(f"No conversations found for {eval_date}")
            return {
                "date": eval_date,
                "error": "No conversations found",
                "dqi": 0.0,
                "alert": None,
            }

        if verbose:
            logger.info(f"Sampled {len(conversations)} conversations")

        # Step 2: Extract turns
        if verbose:
            print("Extracting turns for evaluation...")

        turns = self.log_manager.extract_turns_for_evaluation(conversations)

        if verbose:
            print(f"Extracted {len(turns)} turns")

        # Step 3: Run LLM-as-a-Judge evaluation
        if verbose:
            print(f"\nRunning LLM-as-a-Judge evaluation...")
            print(f"Judge: {self.judge_provider or 'openai'} / {self.judge_model or 'gpt-4o'}")

        evaluator = QualityEvaluator(
            judge_provider=self.judge_provider,
            judge_model=self.judge_model,
            config=self.config
        )
        evaluator.initialize()

        def progress(current, total):
            if verbose:
                print(f"\rProgress: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)

        result = evaluator.evaluate(turns, progress)

        if verbose:
            print(f"\n\nEvaluation Complete:")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Evaluated Turns: {result.success_count}")

        # Step 4: Calculate DQI
        dqi = result.extra_metrics.get("dqi", 0.0)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Daily Quality Index (DQI): {dqi:.2f}%")
            print(f"{'='*60}")
            print(f"\nMetric Pass Rates:")
            for metric in ["q1_role_adherence", "q2_tone_consistency", "q3_turn_count_limit",
                          "q4_level_appropriateness", "q5_correction_behavior",
                          "q6_brevity_encouragement", "q7_language_quality", "q8_safety_filter"]:
                pass_rate = result.extra_metrics.get(f"{metric}_pass_rate", 0.0)
                print(f"  {metric}: {pass_rate:.1f}%")

        # Step 5: Check threshold and generate alert
        alert = check_dqi_threshold(dqi, dqi_threshold)

        if verbose:
            print(f"\n{'='*60}")
            if alert["alert"]:
                print(f"⚠️  ALERT: {alert['message']}")
                print(f"Severity: {alert['severity'].upper()}")
            else:
                print(f"✓ {alert['message']}")
            print(f"{'='*60}\n")

        evaluator.cleanup()

        # Compile results
        return {
            "date": eval_date,
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(conversations),
            "total_turns": len(turns),
            "evaluated_turns": result.success_count,
            "dqi": dqi,
            "dqi_threshold": dqi_threshold,
            "alert": alert,
            "evaluation_result": result.to_dict(),
            "metric_pass_rates": {
                metric: result.extra_metrics.get(f"{metric}_pass_rate", 0.0)
                for metric in ["q1_role_adherence", "q2_tone_consistency", "q3_turn_count_limit",
                              "q4_level_appropriateness", "q5_correction_behavior",
                              "q6_brevity_encouragement", "q7_language_quality", "q8_safety_filter"]
            },
        }

    def run_date_range_evaluation(
        self,
        start_date: str,
        end_date: str,
        n_samples: int = 50,
        dqi_threshold: float = 85.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run quality evaluation over a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            n_samples: Number of conversations to sample per day
            dqi_threshold: Minimum acceptable DQI
            verbose: Print progress

        Returns:
            Dictionary with results for each date
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        results = {}
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            result = self.run_daily_evaluation(
                date=date_str,
                n_samples=n_samples,
                dqi_threshold=dqi_threshold,
                verbose=verbose
            )
            results[date_str] = result
            current += timedelta(days=1)

        # Calculate aggregate statistics
        dqis = [r["dqi"] for r in results.values() if "dqi" in r and r["dqi"] > 0]
        if dqis:
            import statistics
            aggregate = {
                "start_date": start_date,
                "end_date": end_date,
                "total_days": len(results),
                "avg_dqi": statistics.mean(dqis),
                "min_dqi": min(dqis),
                "max_dqi": max(dqis),
                "days_below_threshold": sum(1 for dqi in dqis if dqi < dqi_threshold),
            }
        else:
            aggregate = {
                "start_date": start_date,
                "end_date": end_date,
                "total_days": len(results),
                "error": "No valid DQI data",
            }

        return {
            "aggregate": aggregate,
            "daily_results": results,
        }

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
        filename = f"{name}_{timestamp}.json" if name else f"quality_eval_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def generate_alert_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate alert report for engineering team.

        Args:
            results: Evaluation results
            output_path: Output file path

        Returns:
            Path to alert report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"alert_report_{timestamp}.md"

        alert = results.get("alert", {})
        dqi = results.get("dqi", 0.0)
        date = results.get("date", "unknown")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Quality Alert Report\n\n")
            f.write(f"**Date:** {date}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if alert.get("alert"):
                f.write(f"## ⚠️ ALERT: DQI Below Threshold\n\n")
                f.write(f"**Severity:** {alert['severity'].upper()}\n")
                f.write(f"**DQI:** {dqi:.2f}%\n")
                f.write(f"**Threshold:** {alert['threshold']}%\n\n")
                f.write(f"**Message:** {alert['message']}\n\n")

                f.write(f"## Recommended Actions\n\n")
                f.write(f"1. Review recent model weight changes\n")
                f.write(f"2. Check system prompt modifications\n")
                f.write(f"3. Analyze failing metrics (see below)\n")
                f.write(f"4. Review conversation samples with low scores\n\n")
            else:
                f.write(f"## ✓ Quality Check Passed\n\n")
                f.write(f"**DQI:** {dqi:.2f}%\n")
                f.write(f"**Threshold:** {alert['threshold']}%\n\n")

            f.write(f"## Metric Pass Rates\n\n")
            metric_pass_rates = results.get("metric_pass_rates", {})
            for metric, pass_rate in metric_pass_rates.items():
                status = "✓" if pass_rate >= 85.0 else "⚠️"
                f.write(f"- {status} **{metric}:** {pass_rate:.1f}%\n")

            f.write(f"\n## Evaluation Details\n\n")
            f.write(f"- **Sample Size:** {results.get('sample_size', 0)} conversations\n")
            f.write(f"- **Total Turns:** {results.get('total_turns', 0)}\n")
            f.write(f"- **Evaluated Turns:** {results.get('evaluated_turns', 0)}\n")

        return str(output_path)

    def collect_backend_logs(
        self,
        date: Optional[str] = None,
        backend_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect conversation logs from the running backend service.
        
        This integrates with the LangCoach backend API to fetch real conversation logs
        from the "audio input -> STT -> LLM Inference -> TTS -> audio output" workflow.
        
        Args:
            date: Date to collect logs for (YYYY-MM-DD). If None, uses yesterday.
            backend_url: Backend service URL. If None, reads from .env
            
        Returns:
            List of conversation log dictionaries
        """
        import requests
        from urllib.parse import urljoin
        
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
        if backend_url is None:
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            
        try:
            # Collect logs from backend API
            response = requests.get(
                urljoin(backend_url, f"/api/logs/conversations"),
                params={"date": date},
                timeout=30
            )
            response.raise_for_status()
            
            logs = response.json()
            logger.info(f"Collected {len(logs)} conversation logs from backend for {date}")
            
            # Convert to ConversationLog objects and save locally
            conversation_logs = []
            for log_data in logs:
                try:
                    conv_log = ConversationLog.from_dict(log_data)
                    self.log_manager.save_conversation(conv_log)
                    conversation_logs.append(conv_log)
                except Exception as e:
                    logger.warning(f"Failed to process log {log_data.get('session_id', 'unknown')}: {e}")
                    
            return conversation_logs
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to collect logs from backend: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error collecting backend logs: {e}")
            return []

    def _call_judge_with_retry(
        self, 
        judge_prompt: str, 
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> str:
        """
        Call judge LLM with exponential backoff retry logic.
        
        Args:
            judge_prompt: The prompt to send to the judge
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            Response content from judge LLM
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self._judge_llm.invoke([
                    {"role": "user", "content": judge_prompt}
                ])
                
                # Extract response content
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
                    
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Judge LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Judge LLM call failed after {max_retries + 1} attempts: {e}")
        
        raise last_exception or Exception("Judge LLM call failed")

    def evaluate_concurrent(
        self,
        turns: List[Any],
        max_workers: int = 5,
        progress_callback: Optional[callable] = None
    ) -> EvaluationResult:
        """
        Evaluate multiple turns concurrently for better performance.
        
        Args:
            turns: List of ConversationTurn objects or dicts
            max_workers: Maximum number of concurrent workers
            progress_callback: Optional callback(current, total)
            
        Returns:
            EvaluationResult with quality metrics
        """
        import concurrent.futures
        
        if verbose:
            logger.info(f"Running concurrent evaluation with {max_workers} workers...")
            
        evaluator = QualityEvaluator(
            judge_provider=self.judge_provider,
            judge_model=self.judge_model,
            config=self.config
        )
        evaluator.initialize()
        
        # Use ThreadPoolExecutor for I/O bound tasks (LLM API calls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_turn = {
                executor.submit(evaluator.evaluate_single, turn): turn 
                for turn in turns
            }
            
            results = []
            completed = 0
            total = len(turns)
            
            for future in concurrent.futures.as_completed(future_to_turn):
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation task failed: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "latency": 0.0,
                        "output": None
                    })
        
        evaluator.cleanup()
        
        # Process results same as sequential evaluation
        return self._process_evaluation_results(results)
    
    def _process_evaluation_results(self, results: List[Dict[str, Any]]) -> EvaluationResult:
        """Process evaluation results into EvaluationResult object."""
        # Implementation would be similar to the existing evaluate() method
        # ... (implementation details omitted for brevity)
        pass

    # ...existing code...
def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="LangCoach Quality Evaluation Runner")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to evaluate (YYYY-MM-DD). Default: yesterday"
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Evaluate date range (START END in YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=50,
        help="Number of conversations to sample (default: 50)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=85.0,
        help="DQI threshold for alerts (default: 85.0)"
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        help="Judge LLM provider (openai, anthropic, ollama)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model name"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory containing conversation logs"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--alert-report",
        action="store_true",
        help="Generate alert report"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--collect-logs",
        action="store_true",
        help="Collect logs from backend service before evaluation"
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=None,
        help="Backend service URL for log collection"
    )

    args = parser.parse_args()

    runner = QualityEvaluationRunner(
        log_dir=args.log_dir,
        output_dir=args.output,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )

    verbose = not args.quiet

    # Collect logs from backend if requested
    if args.collect_logs:
        if verbose:
            logger.info("Collecting logs from backend service...")
        runner.collect_backend_logs(
            date=args.date,
            backend_url=args.backend_url
        )

    # Run evaluation
    if args.date_range:
        results = runner.run_date_range_evaluation(
            start_date=args.date_range[0],
            end_date=args.date_range[1],
            n_samples=args.samples,
            dqi_threshold=args.threshold,
            verbose=verbose
        )
    else:
        results = runner.run_daily_evaluation(
            date=args.date,
            n_samples=args.samples,
            dqi_threshold=args.threshold,
            verbose=verbose
        )

    # Save results
    filepath = runner.save_results(results, "daily_quality")
    print(f"\nResults saved to: {filepath}")

    # Generate alert report if requested
    if args.alert_report and "alert" in results:
        alert_path = runner.generate_alert_report(results)
        print(f"Alert report saved to: {alert_path}")


if __name__ == "__main__":
    main()
