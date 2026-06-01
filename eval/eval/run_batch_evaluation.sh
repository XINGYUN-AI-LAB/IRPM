#!/bin/bash
# Batch evaluation script for GPT and Claude models on RewardBench, RM-Bench, PPE, and JudgeBench

# Default values
BENCHMARKS="all"
META_RESULT_SAVE_DIR="./result"
NUM_THREADS=10
MAX_TOKENS=8192
DEBUG=""
SKIP_EXISTING=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --benchmarks) BENCHMARKS="$2"; shift ;;
        --meta_result_save_dir) META_RESULT_SAVE_DIR="$2"; shift ;;
        --num_threads) NUM_THREADS="$2"; shift ;;
        --max_tokens) MAX_TOKENS="$2"; shift ;;
        --debug) DEBUG="--debug" ;;
        --skip_existing) SKIP_EXISTING="--skip_existing" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Create result directory
mkdir -p "$META_RESULT_SAVE_DIR"

# Define models to evaluate
# GPT models
GPT_MODELS=(
    "gpt-4o"
    "gpt-4o-mini"
)

# Claude models
CLAUDE_MODELS=(
    "claude-3-5-sonnet-20240620"
    "claude-3-opus-20240229"
)

echo "========================================"
echo "Batch Evaluation Script"
echo "========================================"
echo "Benchmarks: $BENCHMARKS"
echo "Results directory: $META_RESULT_SAVE_DIR"
echo ""

# Function to run evaluation
run_eval() {
    local model=$1
    echo "----------------------------------------"
    echo "Evaluating: $model"
    echo "----------------------------------------"

    python run_api_model_evaluation.py \
        --model "$model" \
        --benchmarks "$BENCHMARKS" \
        --meta_result_save_dir "$META_RESULT_SAVE_DIR" \
        --num_threads "$NUM_THREADS" \
        --max_tokens "$MAX_TOKENS" \
        $DEBUG \
        $SKIP_EXISTING

    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated $model"
    else
        echo "✗ Failed to evaluate $model"
    fi
    echo ""
}

# Evaluate GPT models
echo "========================================"
echo "Evaluating GPT Models"
echo "========================================"
for model in "${GPT_MODELS[@]}"; do
    run_eval "$model"
done

# Evaluate Claude models
echo "========================================"
echo "Evaluating Claude Models"
echo "========================================"
for model in "${CLAUDE_MODELS[@]}"; do
    run_eval "$model"
done

echo "========================================"
echo "Batch Evaluation Complete!"
echo "========================================"
echo "Results saved to: $META_RESULT_SAVE_DIR"
echo ""
echo "To generate a comparison report, run:"
echo "  python generate_comparison_report.py --result_dir $META_RESULT_SAVE_DIR"
