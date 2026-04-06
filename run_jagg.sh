#!/bin/bash
# JAGG (Jacobian-Aggregated Gradient) backward acceleration training
# Based on run_standard.sh with sum_sample grouping enabled.
#
# Key difference: --sum_sample=5 groups 5 consecutive denoising timesteps,
# reducing backward passes from 5 to 2 per group (2.5x backward speedup).
# sampling_steps changed from 20 to 21 so that (21-1)=20 is divisible by 5.

cd /home/dingruiyi/eff_grpo
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export RANK=0
export WORLD_SIZE=8
mkdir -p /dry-data/grpo-jagg

torchrun --nproc_per_node=8 --master_port 19003 \
    fastvideo/train_grpo_qwenimage_jagg.py \
    --seed 42 \
    --pretrained_model_name_or_path data/qwenimage \
    --vae_model_path data/qwenimage \
    --cache_dir data/.cache \
    --data_json_path data/qwenimage/rl_embeddings_prompts/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 8 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 2000 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 25 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir /dry-data/grpo_jagg \
    --rollout_image_dir /dry-data/grpo_jagg/rollout_scratch \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 21 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 12627 \
    --max_grad_norm 10.0 \
    --weight_decay 0.0001 \
    --use_hpsv2 \
    --num_generations 8 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 0.1 \
    --adv_clip_max 5.0 \
    --selective_checkpointing 0.0 \
    --sum_sample 5
