cd /home/dingruiyi/eff_grpo
export PYTHONPATH=/home/dingruiyi/eff_grpo
# 若 HPSv2 与 DanceGRPO 并列，可不设；脚本会尝试 ../HPSv2
export HPSV2_ROOT=/home/dingruiyi/HPSv2

# HPSv2：仅从本地 hps_ckpt 加载，不访问 Hugging Face Hub
export HF_HUB_OFFLINE=1

export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
CKPT_PTH=$1
echo "Evaluating $CKPT_PTH"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/infer_qwen_dit_hpsv2_single_gpu.py \
  --dit_checkpoint $CKPT_PTH \
  --base_model data/qwenimage \
  --embeddings_path data/qwenimage/rl_embeddings_drawbench \
  --hps_checkpoint hps_ckpt/HPS_v2.1_compressed.pt \
  --hps_open_clip hps_ckpt/open_clip_pytorch_model.bin \
  --output_dir data/outputs/eval_drawbench_grpo_standard_ckpt201 \
  --batch_size 25 \
  --sampling_steps 50
