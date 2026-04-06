#!/bin/bash
# RAB2-B030 训练：early_frac=0.4 + v_so=2*v_prev-1*v_pp + residual_bias=0.30 + residual_growth=1.35
# 使用 train_grpo_qwenimage_eff_mta_rab2.py，并通过 --rab2_preset b030 选择 rollout 配方。
# 路径、资源和 batch 配置对齐 run_ab2_ef04_strong.sh，便于直接横向比较。

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export RANK=0
export WORLD_SIZE=8

OUT_DIR=/dry-data/grpo_eff_len
mkdir -p "$OUT_DIR"

torchrun --nproc_per_node=8 --master_port 19005 \
    fastvideo/train_grpo_qwenimage_eff_mta_rab2.py \
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
    --gradient_accumulation_steps 12 \
    --max_train_steps 2000 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 25 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir "$OUT_DIR" \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 40 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 12627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --use_hpsv2 \
    --num_generations 8 \
    --num_infer 4 \
    --num_guess 4 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.3 \
    --init_same_noise \
    --clip_range 0.1 \
    --adv_clip_max 5.0 \
    --selective_checkpointing 0 \
    --rab2_preset b030
