#!/usr/bin/env bash
# 前台跑完整 Drawbench 对比：torchrun 阻塞直到结束，再检查 comparison_stats.txt。
# 墙上界（多卡+200 prompts×12+HPS）：`scripts/drawbench_time_estimate.sh` 用 original≈28min、其余≈16min、+40min buffer；
# 若你观测仍是 24/12，可把该脚本里的分钟数改小，或单独用 `wait_drawbench_stats.sh` 轮询到出结果。
#
# 远程: conda activate dgrpo && cd DanceGRPO && bash test_hpd_drawbench_mta_new_blocking.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

METHODS="${METHODS:-original original_14 ab2 ab2_2 alpha resab line heun d2 auto}"
OUT="${OUT:-./test_results_comparison_rl_embeddings_drawbench_mta_new}"

EST_SEC="$(bash "$ROOT/scripts/drawbench_time_estimate.sh" "$METHODS")"
echo "=== Estimated max wait ~ $(( (EST_SEC + 59) / 60 )) min (${EST_SEC}s, incl. buffer) ==="
echo "=== METHODS=$METHODS ==="
echo "=== OUT=$OUT ==="

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
    --output_path "$OUT" \
    --methods $METHODS

STAT="$OUT/comparison_stats.txt"
if [ ! -f "$STAT" ]; then
  echo "ERROR: torchrun finished but missing $STAT"
  exit 1
fi
echo "OK: wrote $STAT"
cat "$STAT"
