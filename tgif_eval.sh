#!/bin/bash

# --- 基础配置 ---
CKPT_NAME="Video-LLaVA-7B"
model_path="${CKPT_NAME}"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/mp4"
gt_file_question="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/test_a.json"

# 定义参数矩阵
only_eval=False

# --- GPU 资源配置 ---
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


current_output_dir="./results/TGIF_Zero_Shot_QA/${CKPT_NAME}"
merge_file=${current_output_dir}/merge.jsonl

if [ "$only_eval" != "True" ]; then
    # --- 1. 分块推理阶段 ---
    mkdir -p "${current_output_dir}"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollava/eval/video/run_inference_video_qa.py \
            --model_path ${model_path} \
            --cache_dir ${cache_dir} \
            --video_dir ${video_dir} \
            --gt_file_question ${gt_file_question} \
            --gt_file_answers ${gt_file_answers} \
            --output_dir ${current_output_dir} \
            --output_name ${CHUNKS}_${IDX} \
            --num_chunks $CHUNKS \
            --chunk_idx $IDX \
    done
    wait # 等待所有 GPU 完成

    # --- 2. 合并结果阶段 ---
    > "$merge_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${current_output_dir}/${CHUNKS}_${IDX}.json >> "$merge_file"
    done
fi

# --- 3. GPT 评估阶段 ---
eval_output_dir="${current_output_dir}/gpt-4o-mini"
# eval_output_dir="${current_output_dir}/gemini-2.5-flash-nothinking"
eval_output_json="${current_output_dir}/results.json"
api_key=""
api_base=""
num_tasks=8

python3 videollava/eval/video/eval_video_qa.py \
    --pred_path ${merge_file} \
    --output_dir ${eval_output_dir} \
    --output_json ${eval_output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}

echo "All tasks finished."
