#!/bin/bash
# pip install -e ./
# pip install -e ".[train]"
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# pip install flash-attn --no-build-isolation

deepspeed --num_gpus=8 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/raid5/weizhi/models/Meta-Llama-3.1-8B \
    --version plain \
    --data_path /mnt/raid5/weizhi/llava/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/raid5/weizhi/llava/data/llava/llava_pretrain/images/ \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type aapool_mlp \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_num_image_tokens 144 \
    --bf16 True \
    --output_dir /mnt/raid5/weizhi/checkpoints/lavia-llama-3.1-pretrain-siglip-g-384-aapool-144 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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