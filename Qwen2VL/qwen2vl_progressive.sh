#!/bin/bash

# 0. 加载环境变量
set -a
[ -f .env ] && source .env
set +a

# 1. GPU 配置
NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 2. 基础配置
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
TASKS="gqa,mmbench_en_dev,mmbench_cn_dev,mme,pope,scienceqa,textvqa_val"


BASE_OUTPUT_PATH="./logs/qwen2vl_progressive_new"

# 3. 定义参数矩阵
prune=True
cutoffs=(0.2)
temps=(0.1)
score_types=("clse_attn")
l_lists=("[0,9,18]") 
k_lists=("[1,10,19]") 

# 剪枝率比例配置
pruning_ratios=(
    "[0.57,0.36,0.098]" 
    "[0.38,0.24,0.066]" 
    "[0.19,0.12,0.034]" 
    "[0.334,0.334,0.334]" 
    "[0.223,0.223,0.223]" 
    "[0.112,0.112,0.112]"
)
pruning_ratios=(
    "[0.57,0.36,0.098]" 
   "[0.38,0.24,0.066]" 
    "[0.19,0.12,0.034]" 
)


# 4. 开始多重遍历
for pruning_ratio in "${pruning_ratios[@]}"; do
    for cutoff in "${cutoffs[@]}"; do
        for temp in "${temps[@]}"; do
            for score_type in "${score_types[@]}"; do
                for l_list in "${l_lists[@]}"; do
                    for k_list in "${k_lists[@]}"; do

                        echo "=========================================================="
                        echo "RUNNING: Ratio=$pruning_ratio, C=$cutoff, T=$temp"
                        echo "ST=$score_type, L_LIST=$l_list, K_LIST=$k_list"
                        echo "=========================================================="

                        # 动态生成子文件夹
                        CURRENT_OUTPUT="${BASE_OUTPUT_PATH}/r${pruning_ratio}_c${cutoff}_t${temp}_st_${score_type}_l${l_list}_k${k_list}"
                        mkdir -p "$CURRENT_OUTPUT"

                        # 随机端口防止分布式冲突
                        PORT=$((20000 + RANDOM % 10000))

                        # 5. 运行评估
                        accelerate launch --num_processes=$NUM_GPUS \
                            --main_process_port $PORT \
                            -m lmms_eval \
                            --model qwen2_vl \
                            --model_args "pretrained=${MODEL_PATH},prune=${prune},pruning_ratio=${pruning_ratio},cutoff=${cutoff},temp=${temp},score_type=${score_type},l_list=${l_list},k_list=${k_list}" \
                            --tasks $TASKS \
                            --batch_size 1 \
                            --output_path "$CURRENT_OUTPUT"
                            # --log_samples

                        # 任务结束后清理并稍作停顿
                        echo "Task finished. Cooling down..."
                        sleep 5

                    done
                done
            done
        done
    done
done

echo "All Qwen2-VL tasks finished."