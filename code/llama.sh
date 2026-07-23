#!/bin/bash

# 定义基础路径
MODEL_PATH="../../../llama-2-7b-chat-hf"
DATA_BASE_PATH="../datasets/Abstract/open-source LLM"
OUTPUT_BASE_PATH="../../output_llama"

# 定义 fold 列表
FOLDS=(0 1 2 3 4) # 假设有 5 个 folds

# 遍历每个 fold 并执行命令
for fold in "${FOLDS[@]}"; do
    echo "Starting experiment for fold $fold..."

    python llama3.py \
        --model_path "$MODEL_PATH" \
        --train_path "$DATA_BASE_PATH/fold_$fold/train" \
        --val_path "$DATA_BASE_PATH/fold_$fold/val" \
        --test_path "$DATA_BASE_PATH/fold_$fold/val" \
        --output_dir "$OUTPUT_BASE_PATH/fold_${fold}_t5_lora_output" \
        --log_dir "$OUTPUT_BASE_PATH/fold_${fold}_t5_tensorboard_logs" \
        --epochs 2 \
        --learning_rate 2e-4 \
        --batch_size 1 \
        --lora_r 16 \
        --lora_alpha 64 \
        1>"$OUTPUT_BASE_PATH/fold_${fold}_t5_tensorboard_logs/log" \
        2>"$OUTPUT_BASE_PATH/fold_${fold}_t5_tensorboard_logs/error"

    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "Fold $fold completed successfully."
    else
        echo "Fold $fold failed with exit status $?. Check the error log for details."
    fi
done

echo "All experiments completed."