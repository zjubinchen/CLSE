#!/bin/bash

# 0. 加载环境变量
set -a
[ -f .env ] && source .env
set +a

# 1. GPU 配置
NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 2. 基础配置
MODEL_PATH="liuhaotian/llava-v1.5-7b"
# 如果需要多个任务，取消下面注释
# TASKS="gqa,mme,pope,scienceqa,textvqa,vizwiz_vqa,ocrbench"
TASKS="gqa,mmbench_en_dev,mme,pope,scienceqa,vizwiz_vqa,textvqa_val,ocrbench"
BASE_OUTPUT_PATH="logs/gest_progessive_llava15_test"

# 3. 定义参数矩阵
prune=True
cutoffs=(0.1)
temps=(0.1)
l_lists=("[2,10,20]")
k_lists=("[3,11,21]")

# keep_tokens=("[576,576,576]" "[330,210,62]" "[220,140,41]" "[110,70,20]")

keep_tokens=("[330,210,102]" "[220,140,68]" "[110,70,34]") # 如需对比实验，取消下面注释
score_types=("attn" "clse_attn")


# 4. 开始多重遍历
for kp in "${keep_tokens[@]}"; do
    for cutoff in "${cutoffs[@]}"; do
        for temp in "${temps[@]}"; do
            for score_type in "${score_types[@]}"; do
                for l_list in "${l_lists[@]}"; do
                    for k_list in "${k_lists[@]}"; do

                        # 逻辑过滤：如果 kp 是全量 token 且 score_type 不是默认的，可以跳过以节省时间
                        if [[ "$kp" == "[576,576,576]" && "$score_type" != "attn" ]]; then
                            echo ">>>> Skipping baseline with non-default score_type: kp=$kp <<<<"
                            continue
                        fi

                        if [[ "$prune" == "True" ]]; then
                            echo "=========================================================="
                            echo "RUNNING: kp=$kp, C=$cutoff, T=$temp, ST=$score_type"
                            echo "L_LIST=$l_list, K_LIST=$k_list"
                            echo "=========================================================="
                        fi 
                        # 动态生成子文件夹（注意：路径名若包含方括号在某些系统下需小心，这里加了引号）
                        CURRENT_OUTPUT="${BASE_OUTPUT_PATH}/kp_${kp}_c${cutoff}_t${temp}_st_${score_type}_l${l_list}_k${k_list}"
                        mkdir -p "$CURRENT_OUTPUT"
                        

                        # 随机端口防止分布式冲突
                        PORT=$((10000 + RANDOM % 55000))
                        # 5. 运行评估
                        # 注意：model_args 中的参数需要与你的模型加载代码中的 kwargs 一一对应
                        accelerate launch --num_processes=$NUM_GPUS \
                            --main_process_port $PORT \
                            -m lmms_eval \
                            --model llava \
                            --model_args "pretrained=${MODEL_PATH},prune=${prune},kp=${kp},cutoff=${cutoff},temp=${temp},score_type=${score_type},l_list=${l_list},k_list=${k_list}" \
                            --tasks $TASKS \
                            --batch_size 1 \
                            --output_path "$CURRENT_OUTPUT" \
                            # --log_samples
                        
                        # 每次运行完清理显存并稍作停顿
                        echo "Task finished. Cooling down..."
                        sleep 5
                        
                    done
                done
            done
        done
    done
done

echo "All tasks finished."