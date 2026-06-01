#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

for model in "Qwen3-8B"; do
  echo "#############eval $model ###############"
  sh ./eval/eval_one_command2.sh "all" $model "pointwise" 8 "0,1,2,3,4,5,6,7" 1
done

