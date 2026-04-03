#!/usr/bin/env bash
# 轮询直到 comparison_stats.txt 出现（或超时）。轮询间隔默认 10min，不超过「单方法 12min」量级。
# 用法:
#   wait_drawbench_stats.sh <output_path> <methods_space_separated> [poll_interval_sec]
#
# 例:
#   bash scripts/wait_drawbench_stats.sh ./test_results_comparison_rl_embeddings_drawbench_mta_new \
#     "original original_14 ab2 ab2_2 alpha resab line heun d2 auto"

set -euo pipefail
OUT="${1:?output_path}"
METHODS="${2:?methods}"
INTERVAL="${3:-600}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAX_SEC="$(bash "$ROOT/scripts/drawbench_time_estimate.sh" "$METHODS")"
STAT="$OUT/comparison_stats.txt"

mkdir -p "$OUT"
echo "[wait_drawbench_stats] OUT=$OUT"
echo "[wait_drawbench_stats] MAX_WAIT_SEC=$MAX_SEC (~$((MAX_SEC/60)) min), POLL_INTERVAL_SEC=$INTERVAL"

elapsed=0
while [ "$elapsed" -lt "$MAX_SEC" ]; do
  if [ -f "$STAT" ]; then
    echo "[wait_drawbench_stats] DONE: $STAT"
    head -40 "$STAT"
    exit 0
  fi
  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
  echo "[wait_drawbench_stats] elapsed ${elapsed}s / ${MAX_SEC}s — still waiting..."
done

echo "[wait_drawbench_stats] TIMEOUT after ${MAX_SEC}s; $STAT missing"
exit 1
