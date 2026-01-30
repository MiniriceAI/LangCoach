"""
Report Generator

Generates formatted evaluation reports in multiple formats:
- Console output
- JSON
- Markdown
- HTML
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from evaluators.base import EvaluationResult, TimingMetrics
from evaluators.e2e_evaluator import E2EResult


class ReportGenerator:
    """
    Generates evaluation reports in various formats.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_console_report(
        self,
        results: Dict[str, Any],
        title: str = "Evaluation Report"
    ) -> str:
        """
        Generate a formatted console report.

        Args:
            results: Evaluation results dictionary
            title: Report title

        Returns:
            Formatted string for console output
        """
        lines = []
        width = 70

        # Header
        lines.append("=" * width)
        lines.append(f" {title}")
        lines.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * width)

        for name, result in results.items():
            lines.append("")
            lines.append(f"## {name.upper()}")
            lines.append("-" * width)

            if isinstance(result, dict) and not hasattr(result, 'to_dict'):
                # Comparison results
                for sub_name, sub_result in result.items():
                    lines.extend(self._format_single_result(sub_name, sub_result))
            else:
                lines.extend(self._format_single_result(name, result))

        lines.append("")
        lines.append("=" * width)

        return "\n".join(lines)

    def _format_single_result(self, name: str, result: Any) -> List[str]:
        """Format a single evaluation result."""
        lines = []

        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        else:
            return [f"  {name}: {result}"]

        lines.append(f"\n  [{name}]")
        lines.append(f"    Provider: {data.get('provider', 'N/A')}")
        lines.append(f"    Model: {data.get('model', 'N/A')}")
        lines.append(f"    Success Rate: {data.get('success_rate', 0):.1f}%")
        lines.append(f"    Total Samples: {data.get('total_count', 0)}")

        # Quality metrics (DQI)
        extra = data.get('extra_metrics', {})
        if 'dqi' in extra:
            dqi = extra['dqi']
            status = "✓" if dqi >= 85.0 else "⚠️"
            lines.append(f"    Daily Quality Index (DQI): {status} {dqi:.2f}%")

            # Show metric pass rates
            lines.append(f"    Quality Metrics:")
            for metric in ['q1_role_adherence', 'q2_tone_consistency', 'q3_turn_count_limit',
                          'q4_level_appropriateness', 'q5_correction_behavior',
                          'q6_brevity_encouragement', 'q7_language_quality', 'q8_safety_filter']:
                if f"{metric}_pass_rate" in extra:
                    pass_rate = extra[f"{metric}_pass_rate"]
                    status = "✓" if pass_rate >= 85.0 else "⚠️"
                    lines.append(f"      {status} {metric}: {pass_rate:.1f}%")

        # Timing
        timing = data.get('timing', {})
        if timing:
            lines.append(f"    Timing:")
            lines.append(f"      Mean: {timing.get('mean_ms', 0):.0f}ms")
            lines.append(f"      Median: {timing.get('median_ms', 0):.0f}ms")
            lines.append(f"      P95: {timing.get('p95_ms', 0):.0f}ms")
            lines.append(f"      P99: {timing.get('p99_ms', 0):.0f}ms")
            lines.append(f"      Min: {timing.get('min_ms', 0):.0f}ms")
            lines.append(f"      Max: {timing.get('max_ms', 0):.0f}ms")

        # E2E Timing Breakdown
        if 'timing_breakdown' in data:
            breakdown = data['timing_breakdown']
            lines.append(f"    Component Breakdown:")
            for component in ['stt', 'llm', 'tts']:
                if component in breakdown:
                    comp_timing = breakdown[component]
                    lines.append(f"      {component.upper()}: {comp_timing.get('mean_ms', 0):.0f}ms (P95: {comp_timing.get('p95_ms', 0):.0f}ms)")

        # Extra metrics (excluding quality metrics already shown)
        if extra:
            other_metrics = {k: v for k, v in extra.items()
                           if not k.startswith('q') and k not in ['dqi', 'dqi_median', 'dqi_min', 'dqi_max']}
            if other_metrics:
                lines.append(f"    Extra Metrics:")
                for key, value in other_metrics.items():
                    if isinstance(value, float):
                        lines.append(f"      {key}: {value:.2f}")
                    else:
                        lines.append(f"      {key}: {value}")

        return lines

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        title: str = "Evaluation Report"
    ) -> str:
        """
        Generate a Markdown report.

        Args:
            results: Evaluation results dictionary
            title: Report title

        Returns:
            Markdown formatted string
        """
        lines = []

        # Header
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Module | Provider | Success Rate | Mean Latency | P95 Latency |")
        lines.append("|--------|----------|--------------|--------------|-------------|")

        for name, result in results.items():
            if isinstance(result, dict) and not hasattr(result, 'to_dict'):
                for sub_name, sub_result in result.items():
                    lines.append(self._format_summary_row(sub_name, sub_result))
            else:
                lines.append(self._format_summary_row(name, result))

        lines.append("")

        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")

        for name, result in results.items():
            lines.append(f"### {name.upper()}")
            lines.append("")

            if isinstance(result, dict) and not hasattr(result, 'to_dict'):
                for sub_name, sub_result in result.items():
                    lines.extend(self._format_detailed_markdown(sub_name, sub_result))
            else:
                lines.extend(self._format_detailed_markdown(name, result))

        return "\n".join(lines)

    def _format_summary_row(self, name: str, result: Any) -> str:
        """Format a summary table row."""
        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        else:
            return f"| {name} | N/A | N/A | N/A | N/A |"

        timing = data.get('timing', {})
        return (
            f"| {name} | {data.get('provider', 'N/A')} | "
            f"{data.get('success_rate', 0):.1f}% | "
            f"{timing.get('mean_ms', 0):.0f}ms | "
            f"{timing.get('p95_ms', 0):.0f}ms |"
        )

    def _format_detailed_markdown(self, name: str, result: Any) -> List[str]:
        """Format detailed Markdown section."""
        lines = []

        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        else:
            return [f"**{name}:** {result}", ""]

        lines.append(f"#### {name}")
        lines.append("")
        lines.append(f"- **Provider:** {data.get('provider', 'N/A')}")
        lines.append(f"- **Model:** {data.get('model', 'N/A')}")
        lines.append(f"- **Success Rate:** {data.get('success_rate', 0):.1f}%")
        lines.append(f"- **Total Samples:** {data.get('total_count', 0)}")
        lines.append("")

        # Quality metrics (DQI)
        extra = data.get('extra_metrics', {})
        if 'dqi' in extra:
            dqi = extra['dqi']
            status = "✅" if dqi >= 85.0 else "⚠️"
            lines.append(f"**Daily Quality Index (DQI):** {status} {dqi:.2f}%")
            lines.append("")

            # Quality metrics table
            lines.append("**Quality Metrics Pass Rates:**")
            lines.append("")
            lines.append("| Metric | Pass Rate | Status |")
            lines.append("|--------|-----------|--------|")
            for metric in ['q1_role_adherence', 'q2_tone_consistency', 'q3_turn_count_limit',
                          'q4_level_appropriateness', 'q5_correction_behavior',
                          'q6_brevity_encouragement', 'q7_language_quality', 'q8_safety_filter']:
                if f"{metric}_pass_rate" in extra:
                    pass_rate = extra[f"{metric}_pass_rate"]
                    status = "✅" if pass_rate >= 85.0 else "⚠️"
                    metric_name = metric.replace('_', ' ').title()
                    lines.append(f"| {metric_name} | {pass_rate:.1f}% | {status} |")
            lines.append("")

        # Timing table
        timing = data.get('timing', {})
        if timing:
            lines.append("**Timing Metrics:**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean | {timing.get('mean_ms', 0):.0f}ms |")
            lines.append(f"| Median | {timing.get('median_ms', 0):.0f}ms |")
            lines.append(f"| P50 | {timing.get('p50_ms', 0):.0f}ms |")
            lines.append(f"| P90 | {timing.get('p90_ms', 0):.0f}ms |")
            lines.append(f"| P95 | {timing.get('p95_ms', 0):.0f}ms |")
            lines.append(f"| P99 | {timing.get('p99_ms', 0):.0f}ms |")
            lines.append(f"| Min | {timing.get('min_ms', 0):.0f}ms |")
            lines.append(f"| Max | {timing.get('max_ms', 0):.0f}ms |")
            lines.append("")

        # E2E Breakdown
        if 'timing_breakdown' in data:
            breakdown = data['timing_breakdown']
            lines.append("**Component Breakdown:**")
            lines.append("")
            lines.append("| Component | Mean | P95 |")
            lines.append("|-----------|------|-----|")
            for component in ['stt', 'llm', 'tts', 'total']:
                if component in breakdown:
                    comp = breakdown[component]
                    lines.append(f"| {component.upper()} | {comp.get('mean_ms', 0):.0f}ms | {comp.get('p95_ms', 0):.0f}ms |")
            lines.append("")

        # Extra metrics (excluding quality metrics already shown)
        if extra:
            other_metrics = {k: v for k, v in extra.items()
                           if not k.startswith('q') and k not in ['dqi', 'dqi_median', 'dqi_min', 'dqi_max']}
            if other_metrics:
                lines.append("**Additional Metrics:**")
                lines.append("")
                for key, value in other_metrics.items():
                    if isinstance(value, float):
                        lines.append(f"- {key}: {value:.2f}")
                    else:
                        lines.append(f"- {key}: {value}")
                lines.append("")

        return lines

    def generate_html_report(
        self,
        results: Dict[str, Any],
        title: str = "Evaluation Report"
    ) -> str:
        """
        Generate an HTML report.

        Args:
            results: Evaluation results dictionary
            title: Report title

        Returns:
            HTML formatted string
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .label {{ color: #666; font-size: 14px; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .timestamp {{ color: #888; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        # Summary section
        html += "<h2>Summary</h2>\n<div class='card'>\n<table>\n"
        html += "<tr><th>Module</th><th>Provider</th><th>Success Rate</th><th>Mean Latency</th><th>P95 Latency</th></tr>\n"

        for name, result in results.items():
            if isinstance(result, dict) and not hasattr(result, 'to_dict'):
                for sub_name, sub_result in result.items():
                    html += self._format_html_row(sub_name, sub_result)
            else:
                html += self._format_html_row(name, result)

        html += "</table>\n</div>\n"

        # Detailed sections
        html += "<h2>Detailed Results</h2>\n"

        for name, result in results.items():
            html += f"<h3>{name.upper()}</h3>\n"

            if isinstance(result, dict) and not hasattr(result, 'to_dict'):
                for sub_name, sub_result in result.items():
                    html += self._format_html_detail(sub_name, sub_result)
            else:
                html += self._format_html_detail(name, result)

        html += "</body>\n</html>"
        return html

    def _format_html_row(self, name: str, result: Any) -> str:
        """Format an HTML table row."""
        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        else:
            return f"<tr><td>{name}</td><td colspan='4'>N/A</td></tr>\n"

        timing = data.get('timing', {})
        success_rate = data.get('success_rate', 0)
        success_class = 'success' if success_rate >= 99 else ('warning' if success_rate >= 95 else 'danger')

        return (
            f"<tr><td>{name}</td><td>{data.get('provider', 'N/A')}</td>"
            f"<td class='{success_class}'>{success_rate:.1f}%</td>"
            f"<td>{timing.get('mean_ms', 0):.0f}ms</td>"
            f"<td>{timing.get('p95_ms', 0):.0f}ms</td></tr>\n"
        )

    def _format_html_detail(self, name: str, result: Any) -> str:
        """Format detailed HTML section."""
        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        else:
            return f"<div class='card'><p>{name}: {result}</p></div>\n"

        timing = data.get('timing', {})

        html = f"<div class='card'>\n<h4>{name}</h4>\n"
        html += "<div class='grid'>\n"
        html += f"<div><span class='label'>Provider</span><br><span class='metric'>{data.get('provider', 'N/A')}</span></div>\n"
        html += f"<div><span class='label'>Success Rate</span><br><span class='metric'>{data.get('success_rate', 0):.1f}%</span></div>\n"
        html += f"<div><span class='label'>Mean Latency</span><br><span class='metric'>{timing.get('mean_ms', 0):.0f}ms</span></div>\n"
        html += f"<div><span class='label'>P95 Latency</span><br><span class='metric'>{timing.get('p95_ms', 0):.0f}ms</span></div>\n"
        html += "</div>\n"

        # Timing breakdown for E2E
        if 'timing_breakdown' in data:
            breakdown = data['timing_breakdown']
            html += "<h5>Component Breakdown</h5>\n<table>\n"
            html += "<tr><th>Component</th><th>Mean</th><th>P95</th></tr>\n"
            for component in ['stt', 'llm', 'tts', 'total']:
                if component in breakdown:
                    comp = breakdown[component]
                    html += f"<tr><td>{component.upper()}</td><td>{comp.get('mean_ms', 0):.0f}ms</td><td>{comp.get('p95_ms', 0):.0f}ms</td></tr>\n"
            html += "</table>\n"

        html += "</div>\n"
        return html

    def save_report(
        self,
        results: Dict[str, Any],
        name: str = "evaluation",
        formats: List[str] = None
    ) -> Dict[str, str]:
        """
        Save reports in multiple formats.

        Args:
            results: Evaluation results
            name: Base name for report files
            formats: List of formats ("json", "md", "html", "txt")

        Returns:
            Dictionary mapping format to file path
        """
        formats = formats or ["json", "md", "html"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = {}

        for fmt in formats:
            filename = f"{name}_{timestamp}.{fmt}"
            filepath = self.output_dir / filename

            if fmt == "json":
                # Convert to serializable format
                serializable = {}
                for key, value in results.items():
                    if isinstance(value, dict) and not hasattr(value, 'to_dict'):
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

            elif fmt == "md":
                content = self.generate_markdown_report(results, name)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

            elif fmt == "html":
                content = self.generate_html_report(results, name)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

            elif fmt == "txt":
                content = self.generate_console_report(results, name)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

            saved[fmt] = str(filepath)

        return saved
