#!/usr/bin/env python3
"""
Generate comparison report for GPT and Claude models across RewardBench, RM-Bench, PPE, and JudgeBench.

Usage:
    python generate_comparison_report.py --result_dir ./result
    python generate_comparison_report.py --result_dir ./result --output report.md
    python generate_comparison_report.py --result_dir ./result --format json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comparison report for model evaluations"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.md",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "csv"],
        default="markdown",
        help="Output format"
    )
    return parser.parse_args()


def load_results(result_dir: str) -> Dict[str, Any]:
    """Load all evaluation results from the result directory."""
    result_path = Path(result_dir)
    all_results = {}

    if not result_path.exists():
        print(f"Result directory not found: {result_dir}")
        return all_results

    # Look for combined_results.json in each model subdirectory
    for model_dir in result_path.iterdir():
        if model_dir.is_dir():
            combined_file = model_dir / "combined_results.json"
            if combined_file.exists():
                with open(combined_file) as f:
                    all_results[model_dir.name] = json.load(f)

    return all_results


def extract_rewardbench_scores(result: Dict) -> Dict[str, float]:
    """Extract RewardBench scores from result."""
    scores = {}
    benchmark = result.get("benchmarks", {}).get("rewardbench", {})

    if "error" in benchmark:
        return {"error": benchmark["error"]}

    # Extract section scores
    for key, value in benchmark.items():
        if key in ["Chat", "Chat Hard", "Safety", "Reasoning", "avg_Result_each_section", "absoluate_Result"]:
            scores[key] = value

    return scores


def extract_rm_bench_scores(result: Dict) -> Dict[str, float]:
    """Extract RM-Bench scores from result."""
    scores = {}
    benchmark = result.get("benchmarks", {}).get("rm-bench", {})

    if "error" in benchmark:
        return {"error": benchmark["error"]}

    # Extract main scores
    for key, value in benchmark.items():
        if isinstance(value, (int, float)):
            scores[key] = value

    return scores


def extract_ppe_scores(result: Dict) -> Dict[str, float]:
    """Extract PPE scores from result."""
    scores = {}
    benchmark = result.get("benchmarks", {}).get("ppe", {})

    if "error" in benchmark:
        return {"error": benchmark["error"]}

    # PPE results are typically viewed via view_result.py
    # For now, just return status
    scores["status"] = benchmark.get("status", "unknown")

    return scores


def extract_judgebench_scores(result: Dict) -> Dict[str, float]:
    """Extract JudgeBench scores from result."""
    scores = {}
    benchmark = result.get("benchmarks", {}).get("judgebench", {})

    if "error" in benchmark:
        return {"error": benchmark["error"]}

    # JudgeBench results are in output files
    scores["status"] = benchmark.get("status", "unknown")

    return scores


def generate_markdown_report(all_results: Dict[str, Any]) -> str:
    """Generate markdown comparison report."""
    lines = []

    lines.append("# Model Evaluation Comparison Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report compares the performance of GPT and Claude models across four major reward model benchmarks:")
    lines.append("- **RewardBench**: Evaluates reward model capabilities and safety")
    lines.append("- **RM-Bench**: Reward model benchmark with challenging cases")
    lines.append("- **PPE**: Preference Proxy Evaluations with human preference data")
    lines.append("- **JudgeBench**: Benchmark for evaluating LLM-based judges")
    lines.append("")

    # Model list
    lines.append("## Models Evaluated")
    lines.append("")
    for model_name in sorted(all_results.keys()):
        model_data = all_results[model_name]
        model = model_data.get("model", model_name)
        lines.append(f"- {model}")
    lines.append("")

    # RewardBench Results
    lines.append("## RewardBench Results")
    lines.append("")
    lines.append("| Model | Chat | Chat Hard | Safety | Reasoning | Average |")
    lines.append("|-------|------|-----------|--------|-----------|---------|")

    for model_name in sorted(all_results.keys()):
        model_data = all_results[model_name]
        model = model_data.get("model", model_name)
        scores = extract_rewardbench_scores(model_data)

        if "error" in scores:
            lines.append(f"| {model} | Error | Error | Error | Error | Error |")
        else:
            chat = scores.get("Chat", "N/A")
            chat_hard = scores.get("Chat Hard", "N/A")
            safety = scores.get("Safety", "N/A")
            reasoning = scores.get("Reasoning", "N/A")
            avg = scores.get("avg_Result_each_section", "N/A")

            # Format as percentages if numeric
            def fmt(val):
                if isinstance(val, (int, float)):
                    return f"{val*100:.2f}%"
                return str(val)

            lines.append(f"| {model} | {fmt(chat)} | {fmt(chat_hard)} | {fmt(safety)} | {fmt(reasoning)} | {fmt(avg)} |")

    lines.append("")

    # RM-Bench Results
    lines.append("## RM-Bench Results")
    lines.append("")
    lines.append("| Model | Score |")
    lines.append("|-------|-------|")

    for model_name in sorted(all_results.keys()):
        model_data = all_results[model_name]
        model = model_data.get("model", model_name)
        scores = extract_rm_bench_scores(model_data)

        if "error" in scores:
            lines.append(f"| {model} | Error |")
        else:
            # Get the main score
            main_score = scores.get("absoluate_Result", "N/A")
            if isinstance(main_score, (int, float)):
                main_score = f"{main_score*100:.2f}%"
            lines.append(f"| {model} | {main_score} |")

    lines.append("")

    # PPE Results
    lines.append("## PPE Results")
    lines.append("")
    lines.append("| Model | Status |")
    lines.append("|-------|--------|")

    for model_name in sorted(all_results.keys()):
        model_data = all_results[model_name]
        model = model_data.get("model", model_name)
        scores = extract_ppe_scores(model_data)
        status = scores.get("status", "unknown")
        lines.append(f"| {model} | {status} |")

    lines.append("")

    # JudgeBench Results
    lines.append("## JudgeBench Results")
    lines.append("")
    lines.append("| Model | Status |")
    lines.append("|-------|--------|")

    for model_name in sorted(all_results.keys()):
        model_data = all_results[model_name]
        model = model_data.get("model", model_name)
        scores = extract_judgebench_scores(model_data)
        status = scores.get("status", "unknown")
        lines.append(f"| {model} | {status} |")

    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")

    # Find best model per benchmark
    best_models = defaultdict(list)

    for model_name, model_data in all_results.items():
        model = model_data.get("model", model_name)

        # RewardBench
        rb_scores = extract_rewardbench_scores(model_data)
        if "error" not in rb_scores:
            avg = rb_scores.get("avg_Result_each_section", 0)
            if isinstance(avg, (int, float)):
                best_models["RewardBench"].append((model, avg))

        # RM-Bench
        rmb_scores = extract_rm_bench_scores(model_data)
        if "error" not in rmb_scores:
            score = rmb_scores.get("absoluate_Result", 0)
            if isinstance(score, (int, float)):
                best_models["RM-Bench"].append((model, score))

    # Sort and display best models
    for benchmark, models in best_models.items():
        models.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"**{benchmark}:**")
        for i, (model, score) in enumerate(models[:3], 1):
            lines.append(f"  {i}. {model}: {score*100:.2f}%")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated automatically from evaluation results.*")

    return "\n".join(lines)


def generate_json_report(all_results: Dict[str, Any]) -> str:
    """Generate JSON comparison report."""
    report = {
        "models": {},
        "summary": {}
    }

    for model_name, model_data in all_results.items():
        model = model_data.get("model", model_name)
        report["models"][model] = {
            "rewardbench": extract_rewardbench_scores(model_data),
            "rm_bench": extract_rm_bench_scores(model_data),
            "ppe": extract_ppe_scores(model_data),
            "judgebench": extract_judgebench_scores(model_data),
        }

    return json.dumps(report, indent=2)


def generate_csv_report(all_results: Dict[str, Any]) -> str:
    """Generate CSV comparison report."""
    lines = []

    # Header
    lines.append("Model,RewardBench_Avg,RewardBench_Chat,RewardBench_ChatHard,RewardBench_Safety,RewardBench_Reasoning,RM-Bench_Score")

    for model_name in sorted(all_results.keys()):
        model_data = all_results[model_name]
        model = model_data.get("model", model_name)

        rb_scores = extract_rewardbench_scores(model_data)
        rmb_scores = extract_rm_bench_scores(model_data)

        def get_val(scores, key, default=""):
            val = scores.get(key, default)
            if isinstance(val, (int, float)):
                return f"{val:.4f}"
            return str(val)

        row = [
            model,
            get_val(rb_scores, "avg_Result_each_section"),
            get_val(rb_scores, "Chat"),
            get_val(rb_scores, "Chat Hard"),
            get_val(rb_scores, "Safety"),
            get_val(rb_scores, "Reasoning"),
            get_val(rmb_scores, "absoluate_Result"),
        ]

        lines.append(",".join(row))

    return "\n".join(lines)


def main():
    args = get_args()

    print(f"Loading results from: {args.result_dir}")
    all_results = load_results(args.result_dir)

    if not all_results:
        print("No results found!")
        return

    print(f"Found results for {len(all_results)} models")

    # Generate report
    if args.format == "markdown":
        report = generate_markdown_report(all_results)
    elif args.format == "json":
        report = generate_json_report(all_results)
    elif args.format == "csv":
        report = generate_csv_report(all_results)
    else:
        print(f"Unknown format: {args.format}")
        return

    # Save report
    with open(args.output, "w") as f:
        f.write(report)

    print(f"Report saved to: {args.output}")

    # Also print to console if markdown
    if args.format == "markdown":
        print("\n" + "="*60)
        print(report)
        print("="*60)


if __name__ == "__main__":
    main()
