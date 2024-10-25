#!/bin/bash
LLAVA_EVAL="/mnt/bn/datacompv6/weizhi_multimodal/llava_eval"
python -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/datacompv6/weizhi_multimodal/llava_checkpoints/llava-v1.5-13b \
    --question-file $LLAVA_EVAL/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /mnt/bn/datacompv6/weizhi_multimodal/data/mscoco_2014/val2014\
    --answers-file $LLAVA_EVAL/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir $LLAVA_EVAL/playground/data/eval/pope/coco \
    --question-file $LLAVA_EVAL/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $LLAVA_EVAL/playground/data/eval/pope/answers/original.jsonl