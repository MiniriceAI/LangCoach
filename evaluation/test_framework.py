#!/usr/bin/env python3
"""
Test script for the evaluation framework.

Tests both performance and quality evaluation components.
"""

import sys
from pathlib import Path

# Add paths
eval_path = Path(__file__).parent
project_root = eval_path.parent
src_path = project_root / "src"

sys.path.insert(0, str(eval_path))
sys.path.insert(0, str(src_path))


def test_benchmark_dataset():
    """Test benchmark dataset loading."""
    print("\n" + "="*60)
    print("Testing Benchmark Dataset")
    print("="*60)

    from benchmark.dataset import BenchmarkDataset

    dataset = BenchmarkDataset()
    samples = dataset.get_samples(5)

    print(f"✓ Loaded {len(samples)} samples")
    print(f"  Sample 1: {samples[0].user_input[:50]}...")
    print(f"  Scenario: {samples[0].scenario}")
    print(f"  Difficulty: {samples[0].difficulty}")

    return True


def test_conversation_logs():
    """Test conversation log management."""
    print("\n" + "="*60)
    print("Testing Conversation Log Management")
    print("="*60)

    from benchmark.conversation_logs import (
        ConversationLogManager,
        create_mock_conversation,
    )
    from datetime import datetime

    manager = ConversationLogManager()

    # Create and save mock conversation
    conv = create_mock_conversation(
        session_id=f"test_{datetime.now().timestamp()}",
        scenario="job_interview",
        n_turns=3
    )

    filepath = manager.save_conversation(conv)
    print(f"✓ Saved conversation to: {filepath}")

    # Get statistics
    stats = manager.get_statistics()
    print(f"✓ Total conversations: {stats['total_conversations']}")
    print(f"  Total turns: {stats['total_turns']}")

    return True


def test_quality_evaluator():
    """Test quality evaluator (without actual LLM call)."""
    print("\n" + "="*60)
    print("Testing Quality Evaluator")
    print("="*60)

    from evaluators.quality_evaluator import (
        QualityMetrics,
        ConversationTurn,
        calculate_dqi,
        check_dqi_threshold,
    )

    # Test QualityMetrics
    metrics = QualityMetrics(
        role_adherence=1,
        tone_consistency=1,
        turn_count_limit=1,
        level_appropriateness=0,
        correction_behavior=1,
        brevity_encouragement=1,
        language_quality=1,
        safety_filter=1,
        reasoning="Level too advanced for user"
    )

    print(f"✓ Quality Score: {metrics.quality_score:.2f}%")
    print(f"  Total Score: {metrics.total_score}/8")

    # Test DQI calculation
    quality_scores = [87.5, 100.0, 75.0, 87.5, 100.0]
    dqi = calculate_dqi(quality_scores)
    print(f"✓ DQI: {dqi:.2f}%")

    # Test threshold check
    alert = check_dqi_threshold(dqi, 85.0)
    print(f"✓ Alert Status: {alert['alert']}")
    print(f"  Message: {alert['message']}")

    return True


def test_report_generator():
    """Test report generator."""
    print("\n" + "="*60)
    print("Testing Report Generator")
    print("="*60)

    from reports.report_generator import ReportGenerator
    from evaluators.base import EvaluationResult, TimingMetrics

    generator = ReportGenerator()

    # Create mock result
    timing = TimingMetrics()
    timing.latencies = [0.5, 0.6, 0.7, 0.8, 0.9]

    result = EvaluationResult(
        evaluator="TestEvaluator",
        provider="test-provider",
        model="test-model",
        timing=timing,
        success_count=5,
        failure_count=0,
        extra_metrics={
            "dqi": 87.5,
            "q1_role_adherence_pass_rate": 90.0,
            "q2_tone_consistency_pass_rate": 85.0,
        }
    )

    # Generate console report
    console_report = generator.generate_console_report(
        {"test": result},
        "Test Report"
    )

    print("✓ Generated console report")
    print(f"  Length: {len(console_report)} characters")

    # Generate markdown report
    md_report = generator.generate_markdown_report(
        {"test": result},
        "Test Report"
    )

    print("✓ Generated markdown report")
    print(f"  Length: {len(md_report)} characters")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print(" LangCoach Evaluation Framework - Test Suite")
    print("="*60)

    tests = [
        ("Benchmark Dataset", test_benchmark_dataset),
        ("Conversation Logs", test_conversation_logs),
        ("Quality Evaluator", test_quality_evaluator),
        ("Report Generator", test_report_generator),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {name} - PASSED")
            else:
                failed += 1
                print(f"\n✗ {name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} - FAILED")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print(f" Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
