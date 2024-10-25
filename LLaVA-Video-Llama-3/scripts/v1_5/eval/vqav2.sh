#!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=8 #${#GPULIST[@]}

CKPT="llava-v1.5-13b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

# IDX=0
# CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
#         --model-path /mnt/raid5/weizhi/checkpoints/llava_llama3_8b \
#         --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#         --image-folder ./playground/data/eval/vqav2/test2015 \
#         --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode llama_3

for IDX in $(seq 0 7); do
    sleep 20
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_loader \
        --model-path /mnt/raid5/weizhi/checkpoints/llava_llama3_8b \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llama_3 &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

