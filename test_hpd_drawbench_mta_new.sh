# Drawbench HPS comparison: baseline + MTA variants including v_tgt-based resab/line/heun.
# Run on 4x GPU after: conda activate dgrpo && cd /mnt/shared-storage-user/mineru4s/dingruiyi/DanceGRPO

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
    --output_path ./test_results_comparison_rl_embeddings_drawbench_mta_new \
    --methods original original_14 ab2 ab2_2 alpha resab line heun d2 auto
