#!/bin/bash
bench=$1
model_name=$2
mode=$3
num_gpus=$4
device=$5
voting=$6
# Default values
model="$PWD/$model_name"
model_save_name="$model_name"
#device="0,1,2,3,4,5,6,7"
vllm_gpu_util=0.95
max_tokens=10000
prompt="prompt4"

# Print the arguments
echo "Model: $model"
echo "Model Save Name: $model_save_name"
echo "Device: $device"
echo "VLLM GPU Util: $vllm_gpu_util"
echo "Num GPUs: $num_gpus"
echo "Max Tokens: $max_tokens"
echo "Mode: $mode"

# Record the absolute path of the current directory (RM-R1)
CUR_DIR="$(pwd)"
META_RESULT_SAVE_DIR="${CUR_DIR}/eval/result"


## Reward Bench
if [ "$bench" = "rb" -o "$bench" = "all" ]; then
  cd ${CUR_DIR}/eval/reward-bench
  echo $PWD

  CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py \
      --model $model \
      --vllm_gpu_util $vllm_gpu_util \
      --trust_remote_code \
      --model_save_name $model_save_name \
      --meta_result_save_dir $META_RESULT_SAVE_DIR \
      --num_gpus=$num_gpus \
      --max_tokens=$max_tokens \
      --mode=$mode \
      --voting=$voting \
      --prompt $prompt
fi

## RM-Bench
if [ "$bench" = "rm" -o "$bench" = "all" ]; then
  cd ${CUR_DIR}/eval/RM-Bench
  echo $PWD

  CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py \
      --trust_remote_code \
      --model_save_name $model_save_name \
      --model $model \
      --datapath data/total_dataset_1.json \
      --vllm_gpu_util $vllm_gpu_util \
      --num_gpus=$num_gpus \
      --max_tokens=$max_tokens \
      --META_RESULT_SAVE_DIR $META_RESULT_SAVE_DIR \
      --mode $mode \
      --voting $voting \
      --prompt $prompt

  CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py \
      --trust_remote_code \
      --model_save_name $model_save_name \
      --model $model \
      --datapath data/total_dataset_2.json \
      --vllm_gpu_util $vllm_gpu_util \
      --num_gpus=$num_gpus \
      --max_tokens=$max_tokens \
      --META_RESULT_SAVE_DIR $META_RESULT_SAVE_DIR \
      --mode $mode \
      --voting $voting \
      --prompt $prompt

  CUDA_VISIBLE_DEVICES=$device python scripts/run_generative.py \
      --trust_remote_code \
      --model_save_name $model_save_name \
      --model $model \
      --datapath data/total_dataset_3.json \
      --vllm_gpu_util $vllm_gpu_util \
      --num_gpus=$num_gpus \
      --max_tokens=$max_tokens \
      --META_RESULT_SAVE_DIR $META_RESULT_SAVE_DIR \
      --mode $mode \
      --voting $voting \
      --prompt $prompt

  python scripts/process_final_result.py --mode $mode --model_save_name $model_save_name --model $model --meta_result_save_dir $META_RESULT_SAVE_DIR
fi


### RMB
#if [ "$bench" = "rmb" -o "$bench" = "all" ]; then
#  cd ${CUR_DIR}/eval/RMB-Reward-Model-Benchmark
#  echo $PWD
#
#  CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_generative.py \
#     --model $model \
#     --num_gpus=$num_gpus \
#     --trust_remote_code \
#     --model_save_name $model_save_name \
#     --vllm_gpu_util $vllm_gpu_util \
#     --meta_result_save_dir $META_RESULT_SAVE_DIR \
#     --dataset_dir RMB_dataset/Pairwise_set/Harmlessness \
#     --max_tokens=$max_tokens \
#     --mode $mode
#
#  CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_generative.py \
#     --model $model \
#     --num_gpus=$num_gpus \
#     --trust_remote_code \
#     --model_save_name $model_save_name \
#     --vllm_gpu_util $vllm_gpu_util \
#     --meta_result_save_dir $META_RESULT_SAVE_DIR \
#     --dataset_dir RMB_dataset/Pairwise_set/Helpfulness \
#     --max_tokens=$max_tokens \
#     --mode $mode
#
#  CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_generative_bestofn.py \
#     --model $model \
#     --num_gpus=$num_gpus \
#     --trust_remote_code \
#     --model_save_name $model_save_name \
#     --vllm_gpu_util $vllm_gpu_util \
#     --meta_result_save_dir $META_RESULT_SAVE_DIR \
#     --dataset RMB_dataset/BoN_set/Helpfulness \
#     --max_tokens=$max_tokens \
#     --mode $mode
#
#  CUDA_VISIBLE_DEVICES=$device python eval/scripts/run_generative_bestofn.py \
#     --model $model \
#     --num_gpus=$num_gpus \
#     --trust_remote_code \
#     --model_save_name $model_save_name \
#     --vllm_gpu_util $vllm_gpu_util \
#     --meta_result_save_dir $META_RESULT_SAVE_DIR \
#     --dataset RMB_dataset/BoN_set/Harmlessness \
#     --max_tokens=$max_tokens \
#     --mode $mode
#
#  python eval/scripts/process_final_result.py --model_save_name $model_save_name --meta_result_save_dir $META_RESULT_SAVE_DIR
#fi

## PPE
if [ "$bench" = "ppe" -o "$bench" = "all" ]; then
  cd ${CUR_DIR}/eval/ppe
  echo $PWD

#  CUDA_VISIBLE_DEVICES=$device python -m llm_judge.evaluate4 \
#    --judge arena-hard \
#    --judge-mode ${mode} \
#    --model ${model} \
#    --api-type local \
#    --benchmark-names human_preference_v1 \
#    --meta_result_save_dir $META_RESULT_SAVE_DIR \
#    --model_save_name $model_save_name \
#    --temp 0.0 \
#    --max-token-length $max_tokens \
#    --voting $voting

    CUDA_VISIBLE_DEVICES=$device python -m llm_judge.evaluate2 \
    --judge arena-hard \
    --mode ${mode} \
    --model ${model} \
    --benchmark-names human_preference_v1 \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --model_save_name $model_save_name \
    --max_tokens $max_tokens \
    --num_gpus $num_gpus \
    --voting $voting \
    --prompt $prompt

#  python view_result.py "${META_RESULT_SAVE_DIR}/${model_save_name}/human_preference_v1/${mode}.jsonl" --detailed

#  CUDA_VISIBLE_DEVICES=$device python -m llm_judge.evaluate4 \
#    --judge arena-hard \
#    --judge-mode ${mode} \
#    --model ${model} \
#    --api-type local \
#    --benchmark-names mmlu_pro_best_of_k math_best_of_k gpqa_best_of_k ifeval_best_of_k mbpp_plus_best_of_k \
#    --meta_result_save_dir $META_RESULT_SAVE_DIR \
#    --model_save_name $model_save_name \
#    --temp 0.0   \
#    --max-token-length $max_tokens \
#    --voting $voting

    CUDA_VISIBLE_DEVICES=$device python -m llm_judge.evaluate2 \
    --judge arena-hard \
    --mode ${mode} \
    --model ${model} \
    --benchmark-names mmlu_pro_best_of_k math_best_of_k gpqa_best_of_k ifeval_best_of_k mbpp_plus_best_of_k \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --model_save_name $model_save_name \
    --max_tokens $max_tokens \
    --num_gpus $num_gpus \
    --voting $voting \
    --prompt $prompt

#  python view_correctness_results.py "${META_RESULT_SAVE_DIR}/${model_save_name}" --mode "${mode}" --summary
fi

## judgebench
if [ "$bench" = "jb" -o "$bench" = "all" ]; then
  cd ${CUR_DIR}/eval/judgebench
  echo $PWD

  CUDA_VISIBLE_DEVICES=$device python run_judge2.py \
    --mode ${mode} \
    --model ${model} \
    --meta_result_save_dir $META_RESULT_SAVE_DIR \
    --model_save_name $model_save_name \
    --max_tokens 8192 \
    --vllm_gpu_util $vllm_gpu_util \
    --model_save_name $model_save_name \
    --num_gpus $num_gpus \
    --voting $voting \
    --prompt $prompt

fi