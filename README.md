# IRPM: Intergroup Relative Preference Modeling for Pointwise Generative Reward Models

[**Paper**](https://arxiv.org/abs/2601.00677) | [**Models (Coming Soon)**]()

---

## Overview

**IRPM** proposes Intergroup Relative Preference Modeling, a training framework for pointwise generative reward models. By leveraging relative preference signals across groups, IRPM improves the accuracy and calibration of reward models without requiring pairwise comparisons at inference time.

This repository contains:
- Training code built on the [veRL](https://github.com/volcengine/verl) framework
- Data preprocessing scripts for HelpSteer3 and other preference datasets
- Custom reward functions implementing the IRPM algorithm
- Evaluation harness for public reward model benchmarks (RewardBench, RM-Bench, RMB, PPE, JudgeBench)

---

## Project Structure

```
IRPM/
├── config_dir/              # Training configuration files (YAML)
├── custom_reward/           # IRPM reward function implementations
├── eval/                    # Evaluation code and benchmarks
├── examples/                # Data preprocessing scripts and prompt templates
├── expshells/               # Training launch scripts for different models
├── recipe/                  # Additional training strategies (GRPO, PPO, SPIN, etc.)
├── verl/                    # Core veRL framework (trainer, workers, models)
├── requirements.txt         # Python dependencies
└── setup.py                 # Package installation
```

---


## Training

### Step 1: Prepare Data

Download the [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) dataset and preprocess it into veRL-compatible format:

```bash
cd examples/data_preprocess

# Generate training data from HelpSteer3 preference data
python gen_train_data.py /path/to/helpsteer3_data

# Convert to parquet format for veRL
python trans_pub.py --dataset ./data/steer3filter_prompt4_x2_score --output_dir ./verl_steer3filter_prompt4_x2_score
```

This produces `train.parquet` and `test.parquet` in the output directory.

### Step 2: Download Model

Download a base model (e.g., Qwen3-32B, Qwen3-14B, Qwen3-8B, or LLaMA-3.1-8B-Instruct) from HuggingFace:

```bash
# Example: download Qwen3-32B
huggingface-cli download Qwen/Qwen3-32B --local-dir ./Qwen3-32B
```

### Step 3: Launch Training

Choose a training script based on your model:

```bash
# Qwen3-32B
bash expshells/irpm_qwen3_32b.sh

# Qwen3-14B
bash expshells/irpm_qwen3_14b.sh

# Qwen3-8B
bash expshells/irpm_qwen3_8b.sh

# LLaMA-3.1-8B-Instruct
bash expshells/irpm_llama_critique.sh
```

Before launching, edit the script to set:
- `model_name`: path to your downloaded model
- `train_data`: path to the preprocessed data directory
- `SWANLAB_API_KEY`: your SwanLab API key (optional, for logging)

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.train_batch_size` | 96 | Global batch size |
| `actor_rollout_ref.actor.optim.lr` | 5e-6 | Learning rate |
| `actor_rollout_ref.rollout.n` | 4 | Number of rollout samples per prompt |
| `data.max_prompt_length` | 3072 | Max prompt token length |
| `data.max_response_length` | 2048 | Max response token length |
| `trainer.total_epochs` | 2 | Total training epochs |

---

## Custom Reward Function

The IRPM reward logic is implemented in `custom_reward/group_batch_pub_steer4.py`. The key function is `compute_score_batched_score_mean_adaptive_pre`, which:

1. Parses model outputs for `<critique>...</critique><score>...</score>` format
2. Computes intergroup relative preference rewards
3. Applies format penalties for malformed outputs

To use a custom reward function, specify it in your training config:

```yaml
custom_reward_function:
  name: compute_score_batched_score_mean
  path: custom_reward/group_batch_pub_steer4.py
```

---

## Evaluation

The `eval/` directory contains a unified evaluation harness supporting multiple public benchmarks:

- **RewardBench** - General reward model benchmark
- **RM-Bench** - Reward model benchmark from THU-KEG
- **PPE** - Preference Proxy Evaluations
- **JudgeBench** - LLM-as-Judge benchmark

### Quick Start

```bash
# 1. Enter the eval directory
cd eval

# 2. Prepare the environment
sh eval/run_env.sh

# 3. Run batch evaluation
sh eval/batch_eval.sh
```

You can customize `eval/batch_eval.sh` to change the evaluated models, evaluation mode, number of GPUs, and visible devices. For more details, see [`eval/eval/README.md`](eval/eval/README.md).

---

## Citation

```bibtex
@misc{song2026irpmintergrouprelativepreference,
      title={IRPM: Intergroup Relative Preference Modeling for Pointwise Generative Reward Models}, 
      author={Haonan Song and Qingchen Xie and Huan Zhu and Feng Xiao and Luxi Xing and Liu Kang and Fuzhen Li and Zhiyong Zheng and Feng Jiang and Ziheng Li and Kun Yan and Qingyi Si and Yanghua Xiao and Hongcheng Guo and Fan Yang},
      year={2026},
      eprint={2601.00677},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.00677}, 
}
```

---

## Acknowledgements

This project is built upon the following open-source projects:
- [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
- [RM-R1](https://github.com/RM-R1-UIUC/RM-R1) - Reward Modeling as Reasoning
- [RewardBench](https://github.com/allenai/reward-bench) - Evaluation benchmark for reward models
- [RM-Bench](https://github.com/THU-KEG/RM-Bench) - Reward model benchmark
- [PPE](https://github.com/lmarena/PPE) - Preference Proxy Evaluations
- [JudgeBench](https://github.com/ScalerLab/JudgeBench) - LLM-as-Judge benchmark

