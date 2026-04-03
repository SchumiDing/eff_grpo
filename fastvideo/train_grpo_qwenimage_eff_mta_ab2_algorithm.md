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

---

## 相关变体（均固定每步 `num_guess` / infer 数；guess 无额外网络）

记 \(v^{\mathrm{tgt}} = (x_L^{\mathrm{mean}} - z) / (0-\sigma)\)，\(v^{\mathrm{so}}\) 同上。ResAB / Line / Heun 前半段 \(i < T/2\)，后半段仅 \(v^{\mathrm{so}}\)（与 AB2 一致）。

### MTA-AB2-2（`train_grpo_qwenimage_eff_mta_ab2_2.py`）

与 AB2 相同 \(\alpha = \bigl(1+\cos(v^{\mathrm{so}},v^{\mathrm{group}})\bigr)/2\)，但：

- 前半段阈值与 AB2 一致 \(i < T/2\)；
- 混合 \(a = \alpha^{p}\)，\(p=0.99\)，\(v^{\mathrm{hat}} = a\, v^{\mathrm{so}} + (1-a)\, v^{\mathrm{group}}\)（\(p<1\) 时在 \(\alpha\in(0,1)\) 内略增 \(v^{\mathrm{so}}\) 权重；此前试过 \(p>1\) 在 Drawbench 上略差于 AB2）。

### MTA-ResAB（`train_grpo_qwenimage_eff_mta_resab.py`）

前半段：\(v^{\mathrm{hat}} = v^{\mathrm{tgt}} + \eta\, (v^{\mathrm{so}} - v^{(i-1)})\)，\(\eta\) 为 \((v^{\mathrm{so}}-v^{(i-1)})\) 与 \((v^{\mathrm{tgt}}-v^{(i-1)})\) 的余弦标量；后半段 \(v^{\mathrm{hat}}=v^{\mathrm{so}}\)。

### MTA-Line（`train_grpo_qwenimage_eff_mta_line.py`）

前半段（\(i\ge2\)）：\(v^{\mathrm{hat}} = v^{\mathrm{tgt}} + w\, (v^{(i-1)}-v^{(i-2)})\)，\(w\) 由余弦定义；\(i<2\) 时 \(v^{\mathrm{hat}}=v^{\mathrm{tgt}}\)；后半段同 AB2。

### MTA-Heun（`train_grpo_qwenimage_eff_mta_heun.py`）

前半段：\(v^{\mathrm{hat}} = \frac{1}{2}(v^{\mathrm{so}}+v^{\mathrm{tgt}})\)；后半段 \(v^{\mathrm{hat}}=v^{\mathrm{so}}\)。

