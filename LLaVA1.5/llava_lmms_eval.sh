#!/bin/bash

# 0. 加载环境变量
set -a
[ -f .env ] && source .env
set +a

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="liuhaotian/llava-v1.5-7b"
TASKS="gqa,mmbench_en_dev,mme,pope"
BASE_OUTPUT_PATH="logs/"

prune=True

CURRENT_OUTPUT="${BASE_OUTPUT_PATH}/"
mkdir -p "$CURRENT_OUTPUT"
PORT=$((10000 + RANDOM % 55000))

accelerate launch --num_processes=$NUM_GPUS \
    --main_process_port $PORT \
    -m lmms_eval \
    --model llava \
    --model_args "pretrained=${MODEL_PATH},prune=${prune}\
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "$CURRENT_OUTPUT" \
    --log_samples   
echo "All tasks finished."