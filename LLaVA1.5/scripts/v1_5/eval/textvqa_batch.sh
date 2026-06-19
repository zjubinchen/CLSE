#!/bin/bash

# --- 1. 超参数搜索空间 (GEST 相关参数) ---
cutoffs=(0.1)
temps=(0.1)
keep_tokens=("[330,210,62]" "[220,140,41]" "[110,70,20]")
score_types=("clse_attn")

# --- 2. 硬件配置（多卡并行） ---
# 自动读取 CUDA_VISIBLE_DEVICES 里的 GPU 列表，默认使用 0,5
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,5}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# --- 3. 基础路径配置 (保持不变) ---
MODEL_PATH="liuhaotian/llava-v1.5-13b"
QUESTION_FILE="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_FOLDER="./playground/data/eval/textvqa/train_images"
ANNOTATION_FILE="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"
BASE_ANS_DIR="./playground/data/eval/textvqa/answers"

mkdir -p "$BASE_ANS_DIR"

# --- 4. 开始超参数循环 ---
for kp in "${keep_tokens[@]}"; do
    for cutoff in "${cutoffs[@]}"; do
        for temp in "${temps[@]}"; do
            for score_type in "${score_types[@]}"; do
                
                # 逻辑过滤：如果是满 token 且不是常规 attn，则跳过
                # if [[ "$kp" == "[576,576,576]" && "$score_type" != "attn" ]]; then
                #     echo ">>>> Skipping: kp=$kp with score_type=$score_type <<<<"
                #     continue
                # fi

                # 清理变量名用于文件名 (例如 [120,65,16] -> 120-65-16)
                CLEAN_KP=$(echo $kp | tr -d '[]' | tr ',' '-')
                # 定义最终合并后的文件名
                MERGED_FILE="${BASE_ANS_DIR}/merged_KP${CLEAN_KP}_C${cutoff}_T${temp}_ST${score_type}.jsonl"
                
                echo "========================================================"
                echo "RUNNING: KP=$kp, Cutoff=$cutoff, Temp=$temp, Score=$score_type"
                echo "Using $CHUNKS GPUs for parallel inference..."
                echo "========================================================"
                
                # --- 5. 启动多卡并行分块推理 ---
                for IDX in $(seq 0 $((CHUNKS-1))); do
                    # 每个分块的临时文件名
                    CHUNK_FILE="${BASE_ANS_DIR}/KP${CLEAN_KP}_C${cutoff}_T${temp}_ST${score_type}_${CHUNKS}_${IDX}.jsonl"
                    
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

                # 等待当前超参数组合下的所有进程完成
                wait

                # --- 6. 合并分块结果 ---
                echo "Merging chunks into $MERGED_FILE"
                > "$MERGED_FILE"
                for IDX in $(seq 0 $((CHUNKS-1))); do
                    CHUNK_FILE="${BASE_ANS_DIR}/KP${CLEAN_KP}_C${cutoff}_T${temp}_ST${score_type}_${CHUNKS}_${IDX}.jsonl"
                    if [ -f "$CHUNK_FILE" ]; then
                        cat "$CHUNK_FILE" >> "$MERGED_FILE"
                        rm "$CHUNK_FILE"
                    fi
                done

                # --- 7. 运行 TextVQA 专用评估 ---
                echo "Starting TextVQA evaluation..."
                python -m llava.eval.eval_textvqa \
                    --annotation-file "$ANNOTATION_FILE" \
                    --result-file "$MERGED_FILE"
                
                echo -e "Done with this config.\n"
                sleep 2
                
            done
        done
    done
done

echo "All tasks finished."