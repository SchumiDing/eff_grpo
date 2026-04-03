#!/usr/bin/env bash
# 经验上界（多卡+HPS+IO）：exact "original" ~28min，其余各 ~16min；再加 buffer（默认 40min）。
# 用户口头 24/12 偏乐观，此处乘系数避免轮询过早超时。
# 用法: drawbench_time_estimate.sh "original original_14 ab2" [buffer_min]
# 输出一行：建议最大等待秒数

methods="${1:?methods string}"
buffer="${2:-40}"
sum=0
for m in $methods; do
  if [ "$m" = "original" ]; then
    sum=$((sum + 28))
  else
    sum=$((sum + 16))
  fi
done
sum=$((sum + buffer))
echo $((sum * 60))
