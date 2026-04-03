# Quick re-eval after ResAB/Line/Heun v2 tuning (new output dir so PNG cache does not skip).
# Remote: conda activate dgrpo, cd DanceGRPO, bash test_hpd_drawbench_mta_v2.sh

export PYTHONPATH=/mnt/shared-storage-user/mineru4s/dingruiyi/DanceGRPO/fastvideo:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/mnt/shared-storage-user/mineru4s/dingruiyi/share/cuda-12.8/lib64:$LD_LIBRARY_PATH
export RANK=0
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 fastvideo/test_qwen_rollout_comparison.py \
    --model_path data/qwenimage \
    --embeddings_path data/qwenimage/rl_embeddings_drawbench \
    --batch_size 10 \
    --num_generations 12 \
    --num_guess 4 \
    --rollout_image_dir data/outputs/rollout_scratch_drawbench \
    --output_path ./test_results_comparison_rl_embeddings_drawbench_mta_v2 \
    --methods original ab2 ab2_2 resab line heun
