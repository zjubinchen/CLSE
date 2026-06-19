#!/bin/bash

# --- 1. 超参数搜索空间 ---
cutoffs=(0.1)
temps=(0.1)
keep_tokens=("[330,210,62]" "[220,140,41]" "[110,70,20]")
score_types=("attn" "clse_attn")
score_types=("clse")
# --- 2. 硬件配置 ---
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# --- 3. 基础路径配置 ---
SPLIT="mmbench_dev_20230712"
MODEL_PATH="liuhaotian/llava-v1.5-7b"
QUESTION_FILE="./playground/data/eval/mmbench/$SPLIT.tsv"
BASE_ANS_DIR="./playground/data/eval/mmbench/answers/$SPLIT"
UPLOAD_DIR="./playground/data/eval/mmbench/answers_upload/$SPLIT"

mkdir -p "$BASE_ANS_DIR"
mkdir -p "$UPLOAD_DIR"

# --- 4. 开始参数循环 ---
for kp in "${keep_tokens[@]}"; do
    for cutoff in "${cutoffs[@]}"; do
        for temp in "${temps[@]}"; do
            for score_type in "${score_types[@]}"; do

                CLEAN_KP=$(echo $kp | tr -d '[]' | tr ',' '-')
                # 这里的 EXP_NAME 决定了最终生成文件的后缀
                EXP_NAME="KP${CLEAN_KP}_C${cutoff}_T${temp}_ST${score_type}"
                
                echo "🚀 Running MMBench: $EXP_NAME using $CHUNKS GPUs"

                # --- 5. 多卡并行推理 ---
                for IDX in $(seq 0 $((CHUNKS-1))); do
                    CHUNK_FILE="${BASE_ANS_DIR}/${EXP_NAME}_${CHUNKS}_${IDX}.jsonl"
                    
                    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_mmbench \
                        --model-path $MODEL_PATH \
                        --question-file $QUESTION_FILE \
                        --answers-file "$CHUNK_FILE" \
                        --num-chunks $CHUNKS \
                        --chunk-idx $IDX \
                        --single-pred-prompt \
                        --temperature 0 \
                        --conv-mode vicuna_v1 \
                        --kp "$kp" \
                        --cutoff $cutoff \
                        --temp $temp \
                        --score_type "$score_type" &
                done

                wait

                # --- 6. 合并结果 ---
                MERGED_FILE="${BASE_ANS_DIR}/${EXP_NAME}.jsonl"
                > "$MERGED_FILE"
                for IDX in $(seq 0 $((CHUNKS-1))); do
                    CHUNK_FILE="${BASE_ANS_DIR}/${EXP_NAME}_${CHUNKS}_${IDX}.jsonl"
                    if [ -f "$CHUNK_FILE" ]; then
                        cat "$CHUNK_FILE" >> "$MERGED_FILE"
                        rm "$CHUNK_FILE"
                    fi
                done

                # --- 7. 转换为提交格式 ---
                # 注意：convert 脚本通常会处理整个 result-dir，
                # 我们通过指定 experiment 参数来区分不同的超参数结果
                python scripts/convert_mmbench_for_submission.py \
                    --annotation-file "$QUESTION_FILE" \
                    --result-dir "$BASE_ANS_DIR" \
                    --upload-dir "$UPLOAD_DIR" \
                    --experiment "$EXP_NAME"

                echo -e "✅ Finished: $EXP_NAME\n"

            done
        done
    done
done

echo "All MMBench tasks finished."