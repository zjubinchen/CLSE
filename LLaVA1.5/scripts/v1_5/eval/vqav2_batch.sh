#!/bin/bash

# --- 1. 超参数搜索空间 (在此修改参数) ---
cutoffs=(0.1)
temps=(0.1)
keep_tokens=("[576,576,576]" "[330,210,62]" "[220,140,41]" "[110,70,20]")

score_types=("clse_attn")

# --- 2. 硬件配置 ---
gpu_list="${CUDA_VISIBLE_DEVICES:-0,5}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# --- 3. 基础路径与模型配置 ---
MODEL_PATH="liuhaotian/llava-v1.5-7b"
CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"
QUESTION_FILE="./playground/data/eval/vqav2/$SPLIT.jsonl"
IMAGE_FOLDER="./playground/data/eval/vqav2/test2015"
BASE_ANS_DIR="./playground/data/eval/vqav2/answers/$SPLIT/$CKPT"

mkdir -p "$BASE_ANS_DIR"

# --- 4. 开始超参数循环 ---
for kp in "${keep_tokens[@]}"; do
    for cutoff in "${cutoffs[@]}"; do
        for temp in "${temps[@]}"; do
            for score_type in "${score_types[@]}"; do

                # 清理变量名用于文件名
                CLEAN_KP=$(echo $kp | tr -d '[]' | tr ',' '-')
                # 定义该配置下的合并文件名
                MERGED_FILE="${BASE_ANS_DIR}/merged_KP${CLEAN_KP}_C${cutoff}_T${temp}_ST${score_type}.jsonl"
                
                echo "========================================================"
                echo "RUNNING: KP=$kp, Cutoff=$cutoff, Temp=$temp, Score=$score_type"
                echo "Parallel on $CHUNKS GPUs..."
                echo "========================================================"

                # # --- 5. 启动多卡并行推理 ---
                # for IDX in $(seq 0 $((CHUNKS-1))); do
                #     CHUNK_FILE="${BASE_ANS_DIR}/CHUNK_KP${CLEAN_KP}_C${cutoff}_T${temp}_ST${score_type}_${CHUNKS}_${IDX}.jsonl"
                    
                #     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
                #         --model-path $MODEL_PATH \
                #         --question-file $QUESTION_FILE \
                #         --image-folder $IMAGE_FOLDER \
                #         --answers-file "$CHUNK_FILE" \
                #         --num-chunks $CHUNKS \
                #         --chunk-idx $IDX \
                #         --temperature 0 \
                #         --conv-mode vicuna_v1 \
                #         --kp "$kp" \
                #         --cutoff $cutoff \
                #         --temp $temp \
                #         --score_type "$score_type" &
                # done

                # # 等待当前组所有 GPU 任务完成
                # wait

                # --- 6. 合并分块结果 ---
                # echo "Merging chunks into $MERGED_FILE"
                # > "$MERGED_FILE"
                # for IDX in $(seq 0 $((CHUNKS-1))); do
                #     CHUNK_FILE="${BASE_ANS_DIR}/CHUNK_KP${CLEAN_KP}_C${cutoff}_T${temp}_ST${score_type}_${CHUNKS}_${IDX}.jsonl"
                #     if [ -f "$CHUNK_FILE" ]; then
                #         cat "$CHUNK_FILE" >> "$MERGED_FILE"
                #         rm "$CHUNK_FILE"
                #     fi
                # done

                # --- 7. 转换为 VQAv2 提交格式 ---
                # 注意：这里需要根据转换脚本的要求，可能需要临时把 MERGED_FILE 软链接为 merge.jsonl 
                # 或者直接修改转换脚本的输入路径。这里假设转换脚本读取的是 merge.jsonl
                cp "$MERGED_FILE" "${BASE_ANS_DIR}/merge.jsonl"
                
                echo "Converting to VQAv2 submission format..."
                python scripts/convert_vqav2_for_submission.py \
                    --split $SPLIT \
                    --ckpt $CKPT
                
                # 为防止文件混淆，转换完可以给结果改名（可选）
                # mv "./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/test.json" "./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/submit_KP${CLEAN_KP}.json"

                echo -e "Config finished.\n"
                
            done
        done
    done
done

echo "All grid search tasks finished."