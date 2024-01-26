#!/bin/bash
export MODEL_PATH="./models/sdxl/sdxlbase_v1"
export DATASET_PATH="./datasets/ip2p/ip2p_data"

CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision=fp16 train_ip2p_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_PATH \
    --pretrained_vae_model_name_or_path="./models/vae/sdxl-vae-fp16-fix" \
    --dataset_name=$DATASET_PATH \
    --output_dir="./models/ip2p/ip2p_sdxl" \
    --resolution=768 --random_flip \
    --train_batch_size=8 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=40000 \
    --resume_from_checkpoint="latest" \
    --checkpointing_steps=1000 --checkpoints_total_limit=10 \
    --learning_rate=5e-06 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --original_image_column="source_img" \
    --edited_image_column="target_img" \
    --edit_prompt_column="instruction"
