
for param in "global_step_100" "global_step_200" "global_step_300" "global_step_400" "global_step_500" "global_step_600"; do
  echo "${param}"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $1 \
    --target_dir $2
    # 检查上一个命令是否成功
    if [ $? -eq 0 ]; then
        echo "成功执行: $param"
    else
        echo "执行失败: $param"
    fi
done