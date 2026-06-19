#!/bin/bash

CKPT_NAME="Video-LLaVA-7B"
model_path="${CKPT_NAME}"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/videos"
gt_file_question="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/MSVD_Zero_Shot_QA/test_a.json"

# 定义需要遍历的参数
methods=("gest")
seeds=(42)

gpu_list="${CUDA_VISIBLE_DEVICES:-3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for m in "${methods[@]}"; do
    for s in "${seeds[@]}"; do
        
        # 动态设置当前实验的输出目录
        current_output_dir="results/MSVD_Zero_Shot_QA/${CKPT_NAME}/${m}_${s}"
        mkdir -p "${current_output_dir}"

        echo "🚀 Running: Method=$m, Seed=$s"

        # --- 1. 分块推理阶段 ---
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
                --seed ${s} &  # 使用循环中的 seed
        done

        wait # 等待所有 chunk 完成

        # --- 2. 合并结果阶段 ---
        merge_file=${current_output_dir}/merge.jsonl
        > "$merge_file"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            cat ${current_output_dir}/${CHUNKS}_${IDX}.json >> "$merge_file"
        done

        # --- 3. GPT 评估阶段 ---
        # 注意：这里根据你的路径逻辑更新了 eval 的输入输出
        eval_output_dir="${current_output_dir}/gpt3.5"
        eval_output_json="${current_output_dir}/results.json"
        
        api_key="YOUR_API_KEY"
        api_base="https://api.openai.com/v1"
        num_tasks=8

        python3 videollava/eval/video/eval_video_qa.py \
            --pred_path ${merge_file} \
            --output_dir ${eval_output_dir} \
            --output_json ${eval_output_json} \
            --api_key ${api_key} \
            --api_base ${api_base} \
            --num_tasks ${num_tasks}
            
        echo "✅ Finished: Method=$m, Seed=$s"
        echo "---------------------------------------"
    done
done