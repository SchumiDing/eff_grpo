import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input1 = "/Users/schumi/eff_grpo/ab2.jsonl"
input2 = "/Users/schumi/eff_grpo/standard.jsonl"
length = 40
smooth_window = 5

sns.set_theme()

d1 = pd.read_json(input1, lines=True)
d2 = pd.read_json(input2, lines=True)

reward1 = d1["avg_reward"].tolist()[:length]
reward2 = d2["avg_reward"].tolist()[:length]

smooth1 = pd.Series(reward1).rolling(window=smooth_window, min_periods=1).mean()
smooth2 = pd.Series(reward2).rolling(window=smooth_window, min_periods=1).mean()

x1 = range(len(reward1))
x2 = range(len(reward2))

plt.figure(figsize=(10, 6))

plt.plot(x1, reward1, color="tab:blue", alpha=0.25, linewidth=2, label="Ours: 8+4 (raw)")
plt.plot(x1, smooth1, color="tab:blue", linewidth=2.5, label="Ours: 8+4 (smooth)")
plt.plot(x2, reward2, color="tab:orange", alpha=0.25, linewidth=2, label="Baseline: 8 (raw)")
plt.plot(x2, smooth2, color="tab:orange", linewidth=2.5, label="Baseline: 8 (smooth)")

plt.title("Reward Curve")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.legend()
plt.tight_layout()
plt.savefig("reward.png")
