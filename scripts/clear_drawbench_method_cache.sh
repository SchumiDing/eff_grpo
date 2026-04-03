#!/usr/bin/env bash
# 删除某 rollout 对比输出目录下指定方法的 PNG + .judge.json，便于只重跑该方法。
# 用法: bash scripts/clear_drawbench_method_cache.sh <output_path> <method_dir_name>
# 例:   bash scripts/clear_drawbench_method_cache.sh ./test_results_comparison_rl_embeddings_drawbench_mta_reinfer ab2_2

set -euo pipefail
OUT="${1:?output_path}"
M="${2:?method e.g. ab2_2}"
TARGET="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")/$M"
if [ ! -d "$TARGET" ]; then
  echo "Nothing to delete (missing): $TARGET"
  exit 0
fi
echo "Removing: $TARGET"
rm -rf "$TARGET"
echo "Done. Re-run with same --output_path and a --methods list that includes $M (other methods will skip if their dirs still exist)."
