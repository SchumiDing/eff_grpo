MTA-AB2（`train_grpo_qwenimage_eff_mta_ab2.py`）— 公式版摘要

每步对同一 prompt 的一组 rollout：子集做网络前向得 \(v^{\mathrm{infer}}\)，其余为 guess；组内仅对当步 infer 样本求平均并广播为 \(v^{\mathrm{group}}\)（实现里即 `v_mean` 槽位）。

记本步开始时该 rollout 已保存的上两步速度为 \(v^{(i-1)}, v^{(i-2)}\)（flatten 后做余弦）。

二阶外推（AB2，等步长启发）：

\[
v^{\mathrm{so}} =
\begin{cases}
\frac{3}{2}\,v^{(i-1)} - \frac{1}{2}\,v^{(i-2)}, & i \ge 2, \\[4pt]
v^{(i-1)}, & i < 2.
\end{cases}
\]

余弦映射到融合系数 \(\alpha \in [0,1]\)（对 \(v^{\mathrm{so}}\) 与 \(v^{\mathrm{group}}\) 整向量展平后算）：

\[
\alpha = \frac{1 + \cos\!\bigl(v^{\mathrm{so}},\, v^{\mathrm{group}}\bigr)}{2}.
\]

前半段去噪（\(i < \frac{1}{2}\,T\)，\(T\) 为总步数）guess 速度：

\[
v^{\mathrm{hat}} = \alpha\, v^{\mathrm{so}} + (1-\alpha)\, v^{\mathrm{group}}.
\]

后半段（\(i \ge \frac{1}{2}\,T\)）不再用组均值，仅：

\[
v^{\mathrm{hat}} = v^{\mathrm{so}}.
\]

infer 路径 \(v^{\mathrm{hat}} = v^{\mathrm{infer}}\)。用 \(v^{\mathrm{hat}}\) 进入 `flux_step` 更新 \(z\) 并写回速度历史供下一步使用。
