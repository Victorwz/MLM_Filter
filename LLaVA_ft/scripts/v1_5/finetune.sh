#!/bin/bash
pip install -e ./
pip install -e ".[train]"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install flash-attn --no-build-isolation

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./data/mlm_filter_instruct_50k_gpt4v_cc12m_4k.json \
    --image_folder ./data/images \
    --vision_tower clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/mlm-filter-llava-13b-gpt4v \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none