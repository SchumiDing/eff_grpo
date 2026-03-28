# Rollout decode scratch (optional): keep separate from training's ./images when jobs share the repo cwd.
# export DANCEGRPO_ROLLOUT_IMAGE_DIR=/path/to/your/rollout_scratch_drawbench

export LD_LIBRARY_PATH=/mnt/shared-storage-user/mineru4s/dingruiyi/share/cuda-12.8/lib64:$LD_LIBRARY_PATH
export RANK=0
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 fastvideo/test_qwen_rollout_comparison.py \
    --model_path data/qwenimage \
    --embeddings_path data/qwenimage/rl_embeddings_drawbench\
    --batch_size 10 \
    --num_generations 12 \
    --num_guess 4 \
    --rollout_image_dir data/outputs/rollout_scratch_drawbench \