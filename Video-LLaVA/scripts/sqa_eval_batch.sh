#!/bin/bash

CHUNKS=8
for IDX in {0..7}; do
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_science \
        --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
        --question-file llava_test_QCM-LEA.json \
        --image-folder images/test \
        --answers-file ./test_llava-13b-chunk$CHUNKS_$IDX.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode llava_v1 &
done
