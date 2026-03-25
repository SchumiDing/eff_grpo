#!/bin/bash

export WANDB_MODE=disabled

mkdir -p data/outputs/efficient_auto

# 使用 4 卡训练 (--nproc_per_node=4)
# 调整 gradient_accumulation_steps 为 24 以保持总 Batch Size 为 96 (4 * 1 * 24)
torchrun --nproc_per_node=8 --master_port 19003 \
    fastvideo/train_grpo_qwenimage_eff_mta_auto.py \
    --seed 42 \
    --pretrained_model_name_or_path data/qwenimage \
    --vae_model_path data/qwenimage \
    --cache_dir data/.cache \
    --data_json_path data/qwenimage/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 12 \
    --max_train_steps 300 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 60 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_auto \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 20 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 12627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --use_hpsv2 \
    --num_generations 12 \
    --num_infer 8 \
    --num_guess 4 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --selective_checkpointing 1 \
    --use_cpu_offload