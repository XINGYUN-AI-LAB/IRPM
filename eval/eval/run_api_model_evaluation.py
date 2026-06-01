#!/usr/bin/env python3
"""
Unified evaluation script for GPT and Claude 4 models on RewardBench, RM-Bench, PPE, and JudgeBench.

This script provides a unified interface to evaluate OpenAI GPT models and Anthropic Claude models
on four major reward model benchmarks.

Usage:
    python run_api_model_evaluation.py --model gpt-4o --benchmarks all
    python run_api_model_evaluation.py --model claude-3-5-sonnet-20240620 --benchmarks rewardbench,judgebench

Supported models:
    - GPT: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.
    - Claude: claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-5-sonnet-latest, etc.

Supported benchmarks:
    - rewardbench: RewardBench benchmark
    - rm-bench: RM-Bench benchmark
    - ppe: Preference Proxy Evaluations
    - judgebench: JudgeBench benchmark
    - all: Run all benchmarks
"""

import argparse
import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from tqdm import tqdm

# Add project root to path
CUR_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CUR_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# API Model Lists (from rewardbench/generative.py)
OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0314",
    "gpt-4-0613", "gpt-4-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06", "gpt-4o", "gpt-4o-mini",
    "o1-preview-2024-09-12", "o1-mini-2024-09-12",
)

ANTHROPIC_MODEL_LIST = (
    "claude-1", "claude-2", "claude-2.0", "claude-2.1",
    "claude-instant-1", "claude-instant-1.2",
    "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest", "claude-3-opus-latest",
    # Claude 4 series
    "claude-sonnet-4-20250514", "claude-opus-4-20250514",
    "claude-4-sonnet", "claude-4-opus",
)


GPT5_PROXY_MODEL_LIST = (
    "gpt-5",
)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate GPT or Claude models on RewardBench, RM-Bench, PPE, and JudgeBench"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20240620)"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Comma-separated list of benchmarks: rewardbench,rm-bench,ppe,judgebench, or 'all'"
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default=None,
        help="Name to save results under (defaults to model name)"
    )
    parser.add_argument(
        "--meta_result_save_dir",
        type=str,
        default=str(CUR_DIR / "result"),
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=20,
        help="Number of threads for parallel API calls"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens for generation"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only 10 examples)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip benchmarks that already have results"
    )
    return parser.parse_args()


def get_api_type(model: str) -> str:
    """Determine API type from model name."""
    if model in OPENAI_MODEL_LIST:
        return "openai"
    elif model in ANTHROPIC_MODEL_LIST:
        return "anthropic"
    elif model in GPT5_PROXY_MODEL_LIST:
        return "gpt5_proxy"
    else:
        # Try to infer from model name
        if "gpt" in model.lower() or "o1-" in model.lower():
            return "openai"
        elif "claude" in model.lower():
            return "anthropic"
        else:
            raise ValueError(f"Unknown model: {model}. Cannot determine API type.")


def check_api_key(api_type: str):
    """Check if required API key is set."""
    if api_type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
    elif api_type == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    elif api_type == "gpt5_proxy":
        # GPT-5 uses a hardcoded proxy API key, no need to check environment
        pass


def run_rewardbench(model: str, model_save_name: str, args) -> Dict[str, Any]:
    """Run RewardBench evaluation."""
    print(f"\n{'='*60}")
    print(f"Running RewardBench evaluation for {model}")
    print(f"{'='*60}\n")

    rewardbench_dir = CUR_DIR / "reward-bench"
    result_dir = Path(args.meta_result_save_dir) / model_save_name / "reward_bench"

    # Check if already exists
    if args.skip_existing and (result_dir / "score_result" / "main_score.json").exists():
        print(f"Results already exist at {result_dir}, skipping...")
        with open(result_dir / "score_result" / "main_score.json") as f:
            return json.load(f)

    cmd = [
        "python", "scripts/run_generative.py",
        "--model", model,
        "--model_save_name", model_save_name,
        "--meta_result_save_dir", args.meta_result_save_dir,
        "--num_threads", str(args.num_threads),
        "--max_tokens", str(args.max_tokens),
    ]

    if args.debug:
        cmd.append("--debug")

    try:
        result = subprocess.run(
            cmd,
            cwd=rewardbench_dir,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)

        # Load results
        result_file = result_dir / "score_result" / "main_score.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        else:
            print(f"Warning: Result file not found at {result_file}")
            return {"error": "Result file not found"}
    except subprocess.CalledProcessError as e:
        print(f"Error running RewardBench: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return {"error": str(e)}


def run_rm_bench(model: str, model_save_name: str, args) -> Dict[str, Any]:
    """Run RM-Bench evaluation."""
    print(f"\n{'='*60}")
    print(f"Running RM-Bench evaluation for {model}")
    print(f"{'='*60}\n")

    rm_bench_dir = CUR_DIR / "RM-Bench"
    result_dir = Path(args.meta_result_save_dir) / model_save_name / "rm_bench"

    # Check if already exists
    if args.skip_existing and (result_dir / "score_result" / "main_score.json").exists():
        print(f"Results already exist at {result_dir}, skipping...")
        with open(result_dir / "score_result" / "main_score.json") as f:
            return json.load(f)

    # RM-Bench has 3 dataset parts
    datasets = [
        "data/total_dataset_1.json",
        "data/total_dataset_2.json",
        "data/total_dataset_3.json"
    ]

    all_results = []
    for dataset in tqdm(datasets, desc="RM-Bench datasets", leave=False):
        cmd = [
            "python", "scripts/run_generative.py",
            "--trust_remote_code",
            "--model_save_name", model_save_name,
            "--model", model,
            "--datapath", dataset,
            "--max_tokens", str(args.max_tokens),
            "--META_RESULT_SAVE_DIR", args.meta_result_save_dir,
        ]

        if args.debug:
            cmd.append("--debug")

        try:
            result = subprocess.run(
                cmd,
                cwd=rm_bench_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            all_results.append({"dataset": dataset, "status": "success"})
        except subprocess.CalledProcessError as e:
            print(f"Error running RM-Bench on {dataset}: {e}")
            print(f"stderr: {e.stderr}")
            all_results.append({"dataset": dataset, "status": "error", "error": str(e)})

    # Process final results
    try:
        cmd = [
            "python", "scripts/process_final_result.py",
            "--model_save_name", model_save_name,
            "--model", model,
            "--meta_result_save_dir", args.meta_result_save_dir
        ]
        subprocess.run(cmd, cwd=rm_bench_dir, check=True)

        # Load results
        result_file = result_dir / "score_result" / "main_score.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
    except Exception as e:
        print(f"Error processing RM-Bench results: {e}")

    return {"results": all_results}


def run_ppe(model: str, model_save_name: str, args) -> Dict[str, Any]:
    """Run PPE (Preference Proxy Evaluations) evaluation."""
    print(f"\n{'='*60}")
    print(f"Running PPE evaluation for {model}")
    print(f"{'='*60}\n")

    ppe_dir = CUR_DIR / "ppe"
    api_type = get_api_type(model)

    # Check if already exists
    output_path = ppe_dir / "data"
    result_file = output_path / f"human_preference_v1/arena-hard-pointwise-pointwise--{model}.jsonl"
    if args.skip_existing and result_file.exists():
        print(f"Results already exist at {result_file}, skipping...")
        # Load and compute scores
        return {"status": "skipped", "file": str(result_file)}

    # Run human preference benchmark
    cmd = [
        "python", "-m", "llm_judge.evaluate",
        "--judge", "arena-hard",
        "--judge-mode", "pointwise",
        "--model", model,
        "--api-type", api_type,
        "--benchmark-names", "human_preference_v1",
        "--output-path", str(output_path),
        "--temp", "0.0",
        "--max-token-length", str(args.max_tokens),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=ppe_dir,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running PPE: {e}")
        print(f"stderr: {e.stderr}")
        return {"error": str(e)}

    # View results
    try:
        cmd = [
            "python", "view_result.py",
            str(result_file),
            "--detailed"
        ]
        view_result = subprocess.run(
            cmd,
            cwd=ppe_dir,
            capture_output=True,
            text=True,
            check=True
        )
        print(view_result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error viewing PPE results: {e}")

    return {"status": "completed", "file": str(result_file)}


def run_judgebench(model: str, model_save_name: str, args) -> Dict[str, Any]:
    """Run JudgeBench evaluation."""
    print(f"\n{'='*60}")
    print(f"Running JudgeBench evaluation for {model}")
    print(f"{'='*60}\n")

    judgebench_dir = CUR_DIR / "judgebench"
    api_type = get_api_type(model)

    # Determine judge name based on model
    judge_name = "arena_hard"  # Default judge

    # Check if already exists
    output_file = judgebench_dir / "outputs" / f"dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name={judge_name},judge_model={model.replace('/', '_')}.jsonl"
    if args.skip_existing and output_file.exists():
        print(f"Results already exist at {output_file}, skipping...")
        return {"status": "skipped", "file": str(output_file)}

    # Run on GPT-4o pairs
    pairs_file = judgebench_dir / "data" / "dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"

    cmd = [
        "python", "run_judge.py",
        "--judge_name", judge_name,
        "--judge_model", model,
        "--pairs", str(pairs_file),
        "--concurrency_limit", str(args.num_threads),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=judgebench_dir,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running JudgeBench: {e}")
        print(f"stderr: {e.stderr}")
        return {"error": str(e)}

    # Also run on Claude-3.5-Sonnet pairs if available
    claude_pairs = judgebench_dir / "data" / "dataset=judgebench,response_model=claude-3-5-sonnet-20240620.jsonl"
    if claude_pairs.exists():
        cmd = [
            "python", "run_judge.py",
            "--judge_name", judge_name,
            "--judge_model", model,
            "--pairs", str(claude_pairs),
            "--concurrency_limit", str(args.num_threads),
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=judgebench_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running JudgeBench on Claude pairs: {e}")

    return {"status": "completed", "file": str(output_file)}


def main():
    args = get_args()

    # Validate model
    model = args.model
    api_type = get_api_type(model)
    print(f"Detected API type: {api_type}")

    # Check API key
    check_api_key(api_type)

    # Set model save name
    model_save_name = args.model_save_name or model.replace("/", "_")

    # Parse benchmarks
    if args.benchmarks == "all":
        benchmarks = ["rewardbench", "rm-bench", "ppe", "judgebench"]
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    print(f"\n{'='*60}")
    print(f"Evaluating {model} on benchmarks: {', '.join(benchmarks)}")
    print(f"Results will be saved to: {args.meta_result_save_dir}")
    print(f"{'='*60}\n")

    # Run evaluations
    all_results = {
        "model": model,
        "model_save_name": model_save_name,
        "benchmarks": {}
    }

    # Progress bar for benchmarks
    pbar = tqdm(benchmarks, desc="Overall Progress", unit="benchmark")
    for benchmark in pbar:
        pbar.set_postfix_str(f"Running: {benchmark}")
        start_time = time.time()

        if benchmark == "rewardbench":
            result = run_rewardbench(model, model_save_name, args)
        elif benchmark == "rm-bench":
            result = run_rm_bench(model, model_save_name, args)
        elif benchmark == "ppe":
            result = run_ppe(model, model_save_name, args)
        elif benchmark == "judgebench":
            result = run_judgebench(model, model_save_name, args)
        else:
            print(f"Unknown benchmark: {benchmark}")
            continue

        elapsed_time = time.time() - start_time
        result["elapsed_time"] = f"{elapsed_time:.2f}s"

        all_results["benchmarks"][benchmark] = result
        
        # Update progress bar with result status
        status = "✓" if "error" not in result else "✗"
        pbar.set_postfix_str(f"{benchmark}: {status} ({elapsed_time:.1f}s)")

    # Save combined results
    result_save_path = Path(args.meta_result_save_dir) / model_save_name
    result_save_path.mkdir(parents=True, exist_ok=True)

    combined_result_file = result_save_path / "combined_results.json"
    with open(combined_result_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to {combined_result_file}")
    print(f"{'='*60}\n")

    # Print summary
    print("\nSummary:")
    print(f"  Model: {model}")
    print(f"  Benchmarks evaluated: {', '.join(benchmarks)}")
    for benchmark, result in all_results["benchmarks"].items():
        status = "✓" if "error" not in result else "✗"
        time_taken = result.get("elapsed_time", "N/A")
        print(f"  {status} {benchmark}: {time_taken}")

    return all_results


if __name__ == "__main__":
    main()
