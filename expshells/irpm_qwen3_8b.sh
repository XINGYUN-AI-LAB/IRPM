#!/bin/bash

pip install -r requirements.txt --no-deps
pip install -e . --no-build-isolation
pip install swanlab==0.7.2


# --
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --
export HF_HUB_DOWNLOAD_TIMEOUT=1000
export HF_ENDPOINT=https://hf-mirror.com
JOB_COMPLETION_INDEX=0
CLUSTER_SIZE=1
model_name="Qwen3-8B"


train_data="verl_steer3filter_prompt4_x2_score"


export SWANLAB_PROJECT=irpm
# You need to fill in your own API key.
export SWANLAB_API_KEY=""

if [ "$JOB_COMPLETION_INDEX" = 0 ] ; then
  ray start --head --include-dashboard true --dashboard-host 0.0.0.0
else
  ray start --address "$MASTER_ADDR:6379"
  python misc/ray_join.py "$MASTER_ADDR:6379"
fi

#if [ "$JOB_COMPLETION_INDEX" = 0 ] ; then
#MODEL_PATH=
EXP_NAME=steer3_filter-prompt4-b96-lr5e6-score-060-qwen3-8b-think-ada1
DATASET=${train_data}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATASET}/train.parquet \
    data.val_files=${DATASET}/test.parquet \
    custom_reward_function.name=compute_score_batched_score_mean_adaptive_pre \
    custom_reward_function.path=custom_reward/group_batch_pub_steer4.py\
    data.train_batch_size=96 \
    data.shuffle=false \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.checkpoint.save_contents="['model']" \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.model.path=${model_name} \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=50 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=96 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=batch2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=false \
    trainer.critic_warmup=0.1 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$SWANLAB_PROJECT \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${CLUSTER_SIZE} \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.default_local_dir=$EXP_NAME \
    trainer.total_epochs=2 $@ 2>&1 | tee run6.log

ray stop --force
# fi

sleep 30

