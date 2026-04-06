#!/bin/bash
# DrawBench 上复现 comparison_stats 这组方法，统一改为 6 infer + 6 guess。

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Rollout decode scratch (optional): keep separate from training's ./images when jobs share the repo cwd.
export DANCEGRPO_ROLLOUT_IMAGE_DIR="${DANCEGRPO_ROLLOUT_IMAGE_DIR:-data/outputs/rollout_scratch_drawbench_6i6g}"

export LD_LIBRARY_PATH=/mnt/shared-storage-user/mineru4s/dingruiyi/share/cuda-12.8/lib64:$LD_LIBRARY_PATH
export RANK=0
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 fastvideo/test_qwen_rollout_comparison.py \
    --model_path data/qwenimage \
    --embeddings_path data/qwenimage/rl_embeddings_drawbench \
    --batch_size 10 \
    --num_generations 12 \
    --num_guess 6 \
    --num_guess_min 6 \
    --sampling_steps 20 \
    --seed 42 \
    --init_same_noise \
    --methods original original_14 ab2_abl_ef04_strong rab2 rab2_b030 rab2_explorer \
    --rollout_image_dir "$DANCEGRPO_ROLLOUT_IMAGE_DIR" \
    --output_path ./test_results_comparison_rl_embeddings_drawbench_6i6g
