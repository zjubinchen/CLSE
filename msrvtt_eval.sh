#!/bin/bash

CKPT_NAME="Video-LLaVA-7B"
model_path="${CKPT_NAME}"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/videos/all"
gt_file_question="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_a.json"

# 定义参数矩阵
prune=True
cutoffs=(0.1)
temps=(0.1)
# l_lists=("[0,10,20]")
# k_lists=("[1,11,21]")
# keep_tokens=("[330,210,67]")
l_lists=("[2]")
k_lists=("[3]")
keep_tokens=("[194]")

merge_tokens_list=(0)
score_types=("clse_attn")
use_cluster_merges=(False)

only_eval=False

gpu_list="${CUDA_VISIBLE_DEVICES:-2,3,4}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for kp in "${keep_tokens[@]}"; do
    for cutoff in "${cutoffs[@]}"; do
        for temp in "${temps[@]}"; do
            for score_type in "${score_types[@]}"; do
                for l_list in "${l_lists[@]}"; do
                    for k_list in "${k_lists[@]}"; do
                        for use_cluster_merge in "${use_cluster_merges[@]}"; do
                        for merge_tokens in "${merge_tokens_list[@]}"; do

                        current_output_dir="./results_3d/kp_${kp}_mt_${merge_tokens}_c${cutoff}_t${temp}_st_${score_type}_l${l_list}_k${k_list}_cm${use_cluster_merge}/MSRVTT_Zero_Shot_QA/${CKPT_NAME}"

                        if [ "$only_eval" != "True" ]; then
                            # --- 1. 分块推理阶段 ---
                            mkdir -p "${current_output_dir}"
                            echo "Running: kp=$kp, MT=$merge_tokens, C=$cutoff, T=$temp, ST=$score_type, L=$l_list, K=$k_list, CM=$use_cluster_merge"
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
                                    --score_type ${score_type} \
                                    --cutoff ${cutoff} \
                                    --temp ${temp} \
                                    --l_list "${l_list}" \
                                    --k_list "${k_list}" \
                                    --keep_tokens "${kp}" \
                                    --use_cluster_merge ${use_cluster_merge} \
                                    --merge_tokens ${merge_tokens} &
                            done
                            wait # 等待所有 chunk 完成
                        fi

                        # --- 2. 合并结果阶段 ---
                        merge_file=${current_output_dir}/merge.jsonl

                        if [ "$only_eval" != "True" ]; then
                            > "$merge_file"
                            for IDX in $(seq 0 $((CHUNKS-1))); do
                                cat ${current_output_dir}/${CHUNKS}_${IDX}.json >> "$merge_file"
                            done
                        fi

                        # --- 3. GPT 评估阶段 ---
                        eval_output_dir="${current_output_dir}/gpt-4o-mini"
                        eval_output_dir="${current_output_dir}/gemini-2.5-flash-nothinking"
                        eval_output_json="${current_output_dir}/results.json"
       
                        api_key="sk-eFeesiKgCR2nDb6jPQbTDqCEIpB1KhxAXZ1i6Y8DMZNLiRsq"
                        api_base="https://api.chatanywhere.tech"
                        num_tasks=8

                        python3 videollava/eval/video/eval_video_qa.py \
                            --pred_path ${merge_file} \
                            --output_dir ${eval_output_dir} \
                            --output_json ${eval_output_json} \
                            --api_key ${api_key} \
                            --api_base ${api_base} \
                            --num_tasks ${num_tasks}

                        echo "Finished: kp=$kp, MT=$merge_tokens, C=$cutoff, T=$temp, ST=$score_type, L=$l_list, K=$k_list, CM=$use_cluster_merge"
                        echo "---------------------------------------"
                        sleep 5

                        done
                        done
                    done
                done
            done
        done
    done
done

echo "All tasks finished."
