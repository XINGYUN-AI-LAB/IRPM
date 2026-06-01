# Evaluations for Reward Models

This directory provides a **one-command evaluation pipeline** for Reward Models, adapted from [RewardBench](https://github.com/allenai/reward-bench), [RM-Bench](https://github.com/THU-KEG/RM-Bench), and [RMB](https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark). We sincerely thank the authors of these benchmarks for their foundational contributions.

Compared to standard score-only evaluation pipelines, our framework emphasizes **transparency and interpretability** by logging **per-sample evaluation outputs**, which are particularly useful for analyzing generative reward models. Example outputs are available in the [`result/`](result/) directory.

The original codebases are mostly reused with minimal changes, except for RMB, where we fully re-implemented the pipeline to support our unified framework. Further details are available in the [`RMB-Reward-Model-Benchmark/`](RMB-Reward-Model-Benchmark/) directory. We also extend RM-Bench to enable evaluation of **generative reward models**, which was not originally supported.


---

### Requirements

This codebase has been tested with the following main dependencies (no need to `pip install -e .`):

```bash
torch=2.6.0cu12.4+
transformers=4.51.0
vllm=0.8.3
fastchat
accelerate==1.8.1
datasets
openai
google-generativeai
anthropic
together
```

> ⚠️ **Note:** `transformers` and `vllm` may have compatibility issues. Please ensure the versions align with your backend setup.

---

### Quick Start

```bash
# 1. Enter the eval directory
cd eval

# 2. Prepare the environment (install reward-bench2 dependencies)
sh eval/run_env.sh

# 3. Run batch evaluation
sh eval/batch_eval.sh
```

`run_env.sh` installs the `reward-bench2` package in editable mode. `batch_eval.sh` loops over specified models and runs the full evaluation pipeline (RewardBench, RM-Bench, PPE, JudgeBench) via `eval_one_command2.sh`.

You can customize `batch_eval.sh` to change the evaluated models, evaluation mode (`pointwise`), number of GPUs, and visible devices.

---

### Running the evaluation (manual)

Our evaluation supports one-command running. An example is provided below:

```bash
bash eval_one_command.sh --model gaotang/RM-R1-Qwen2.5-Instruct-32B --model_save_name RM-R1-Qwen2.5-Instruct-32B --device 0,1,2,3 --vllm_gpu_util 0.90 --num_gpus 4
```

Or use `eval_one_command2.sh` with positional arguments:

```bash
sh eval/eval_one_command2.sh <bench> <model_name> <mode> <num_gpus> <devices> <voting>
# Example:
sh eval/eval_one_command2.sh "all" "Qwen3-8B" "pointwise" 8 "0,1,2,3,4,5,6,7" 1
```

| Argument | Description |
|----------|-------------|
| `bench` | Benchmark to run: `rb` (RewardBench), `rm` (RM-Bench), `ppe` (PPE), `jb` (JudgeBench), or `all` |
| `model_name` | Model directory name (relative to current working directory) |
| `mode` | Evaluation mode, e.g. `pointwise` |
| `num_gpus` | Number of GPUs to use |
| `devices` | CUDA visible devices, e.g. `0,1,2,3,4,5,6,7` |
| `voting` | Voting rounds (1 for single pass) |
---

### Community Contribution

We believe in the power of open evaluation and that the comprehensive benchmarks used in this project can support a wide range of research in reward modeling. Feel free to submit pull requests to:

- 🚀 **Add your models** to our codebase
- 📊 **Introduce new benchmarks** compatible with our one-command, multi-benchmark evaluation pipeline

Let’s work together to build a transparent, extensible, and community-driven evaluation suite for reward models.