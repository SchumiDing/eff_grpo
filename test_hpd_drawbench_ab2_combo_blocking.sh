#!/usr/bin/env bash
# AB2 融合：early_frac=0.4 + v_so=2*v_prev-1*v_pp（ab2_abl_ef04_strong）
# 对照：original、标准 ab2、单独 ef04、单独 strong 外推
# 远程: conda activate dgrpo && cd DanceGRPO && bash test_hpd_drawbench_ab2_combo_blocking.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

METHODS="${METHODS:-original ab2 ab2_abl_ef04 ab2_abl_strong ab2_abl_ef04_strong}"
OUT="${OUT:-./test_results_comparison_rl_embeddings_drawbench_ab2_combo}"

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
    --rollout_image_dir data/outputs/rollout_scratch_drawbench_ab2_combo \
    --output_path "$OUT" \
    --methods $METHODS

STAT="$OUT/comparison_stats.txt"
if [ ! -f "$STAT" ]; then
  echo "ERROR: torchrun finished but missing $STAT"
  exit 1
fi
echo "OK: wrote $STAT"
cat "$STAT"
