cd /mnt/shared-storage-user/mineru4s/dingruiyi/DanceGRPO
export PYTHONPATH=/mnt/shared-storage-user/mineru4s/dingruiyi/DanceGRPO
# 若 HPSv2 与 DanceGRPO 并列，可不设；脚本会尝试 ../HPSv2
export HPSV2_ROOT=/mnt/shared-storage-user/mineru4s/dingruiyi/HPSv2

# HPSv2：仅从本地 hps_ckpt 加载，不访问 Hugging Face Hub
export HF_HUB_OFFLINE=1

export LD_LIBRARY_PATH=/mnt/shared-storage-user/mineru4s/dingruiyi/share/cuda-12.8/lib64:$LD_LIBRARY_PATH
export RANK=0
CKPT_PTH=/mnt/shared-storage-user/mineru4s/dingruiyi/DanceGRPO/checkpoint-176-0

python scripts/infer_qwen_dit_hpsv2_single_gpu.py \
  --dit_checkpoint $CKPT_PTH \
  --base_model data/qwenimage \
  --embeddings_path data/qwenimage/rl_embeddings_drawbench \
  --hps_checkpoint hps_ckpt/HPS_v2.1_compressed.pt \
  --hps_open_clip hps_ckpt/open_clip_pytorch_model.bin \
  --output_dir data/outputs/eval_drawbench_grpo_eff_auto_mta_2_ckpt176 \
  --batch_size 100 \
  --sampling_steps 50 \
  --device cuda:0