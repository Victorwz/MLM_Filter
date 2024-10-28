#!/bin/bash
# LLAVA_EVAL="/mnt/bn/datacompv6/weizhi_multimodal/llava_eval"
LLAVA_EVAL=./
python -m llava.eval.model_vqa_loader \
    --model-path /mnt/raid5/weizhi/checkpoints/llava_llama3_8b \
    --question-file $LLAVA_EVAL/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $LLAVA_EVAL/playground/data/eval/textvqa/train_images \
    --answers-file $LLAVA_EVAL/playground/data/eval/textvqa/answers/llava-v1.5-llama3-8b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file $LLAVA_EVAL/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $LLAVA_EVAL/playground/data/eval/textvqa/answers/llava-v1.5-llama3-8b.jsonl
