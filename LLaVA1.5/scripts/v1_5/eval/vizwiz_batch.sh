#!/bin/bash

# --- 1. 超参数搜索空间 (根据需要调整) ---
cutoffs=(0.1)
temps=(0.1)
keep_tokens=("[330,210,62]" "[220,140,41]" "[110,70,20]")
keep_tokens=("[576,576,576]")
score_types=("clse_attn")

# --- 2. 硬件配置（自动检测 CUDA_VISIBLE_DEVICES） ---
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}" # 如果环境变量没设，默认用 0,1
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# --- 3. 基础路径配置 ---
MODEL_PATH="liuhaotian/llava-v1.5-7b"
QUESTION_FILE="./playground/data/eval/vizwiz/llava_val.jsonl"
IMAGE_FOLDER="./playground/data/eval/vizwiz/val"
BASE_ANS_DIR="./playground/data/eval/vizwiz/answers"
UPLOAD_DIR="./playground/data/eval/vizwiz/answers_upload"

mkdir -p "$BASE_ANS_DIR"
mkdir -p "$UPLOAD_DIR"

# --- 4. 开始参数循环 ---
for kp in "${keep_tokens[@]}"; do
    for cutoff in "${cutoffs[@]}"; do
        for temp in "${temps[@]}"; do
            for score_type in "${score_types[@]}"; do

                # 清理变量名用于文件名
                CLEAN_KP=$(echo $kp | tr -d '[]' | tr ',' '-')
                CONFIG_NAME="KP${CLEAN_KP}_C${cutoff}_T${temp}_ST${score_type}"
                MERGED_FILE="${BASE_ANS_DIR}/${CONFIG_NAME}.jsonl"
                RESULT_UPLOAD="${UPLOAD_DIR}/${CONFIG_NAME}.json"

                echo "-------------------------------------------------------"
                echo "RUNNING: $CONFIG_NAME"
                echo "Using $CHUNKS GPUs for parallel inference..."
                echo "-------------------------------------------------------"

                # --- 5. 启动多卡并行推理 ---
                for IDX in $(seq 0 $((CHUNKS-1))); do
                    CHUNK_FILE="${BASE_ANS_DIR}/${CONFIG_NAME}_${CHUNKS}_${IDX}.jsonl"
                    
                    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
                        --model-path $MODEL_PATH \
                        --question-file $QUESTION_FILE \
                        --image-folder $IMAGE_FOLDER \
                        --answers-file "$CHUNK_FILE" \
                        --num-chunks $CHUNKS \
                        --chunk-idx $IDX \
                        --temperature 0 \
                        --conv-mode vicuna_v1 \
                        --kp "$kp" \
                        --cutoff $cutoff \
                        --temp $temp \
                        --score_type "$score_type" &
                done

                # 等待所有 GPU 任务完成
                wait

                # --- 6. 合并分块结果 ---
                echo "Merging chunks into $MERGED_FILE"
                > "$MERGED_FILE"
                for IDX in $(seq 0 $((CHUNKS-1))); do
                    CHUNK_FILE="${BASE_ANS_DIR}/${CONFIG_NAME}_${CHUNKS}_${IDX}.jsonl"
                    if [ -f "$CHUNK_FILE" ]; then
                        cat "$CHUNK_FILE" >> "$MERGED_FILE"
                        rm "$CHUNK_FILE"
                    fi
                done

                # --- 7. 转换为 VizWiz 提交格式 ---
                echo "Converting results for submission..."
                python scripts/convert_vizwiz_for_submission.py \
                    --annotation-file "$QUESTION_FILE" \
                    --result-file "$MERGED_FILE" \
                    --result-upload-file "$RESULT_UPLOAD"

                echo -e "Done: $RESULT_UPLOAD\n"
                
            done
        done
    done
done

echo "All hyperparameter searches finished."