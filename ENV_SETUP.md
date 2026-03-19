# 环境配置指南 (Environment Setup Guide)

本项目的环境配置基于 `env_setup.sh`，旨在为 DanceGRPO 训练提供高性能的运行环境。建议使用 Python 3.10+ 和 CUDA 12.1+。

## 1. 基础环境安装

您可以直接运行提供的脚本进行一键安装：

```bash
bash env_setup.sh
```

### 脚本核心内容解析：

1.  **PyTorch 安装**：
    安装 Torch 2.5.0 及对应的 CUDA 12.1 版本。
    ```bash
    pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **高性能算子 (Flash Attention 2)**：
    安装 FA2 以加速 Transformer 计算。
    ```bash
    pip install packaging ninja
    pip install flash-attn==2.7.0.post2 --no-build-isolation
    ```

3.  **核心库与 Lint 工具**：
    安装 `diffusers`, `transformers`, `accelerate` 以及代码规范工具。
    ```bash
    pip install -r requirements-lint.txt
    pip install -e .
    ```

4.  **其他依赖**：
    包括 `ml-collections`, `pydantic`, `huggingface_hub` 等必要工具库。

## 2. 额外强化学习依赖

针对 GRPO 训练和 HPSv2 评估，您需要额外安装以下库：

```bash
# HPSv2 评估模型
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2 && pip install -e . && cd ..

# 强化学习相关
pip install wandb tqdm datasets
```

## 3. 常见问题 (FAQ)

### 3.1 关于 `flash-attn` 安装失败
如果 `flash-attn` 安装缓慢或失败，请确保：
- 系统已安装 `gcc` 和 `g++` 11+。
- `nvcc -V` 输出的 CUDA 版本与 PyTorch 的 CUDA 版本一致。

### 3.2 显存优化
本项目默认启用 `gradient_checkpointing`。如果显存仍然不足（尤其是在非 A800/H800 显卡上），建议：
- 减小 `train_batch_size`。
- 增加 `gradient_accumulation_steps` 以保持相同的有效 Batch Size。
- 使用 `train_grpo_qwenimage_eff.py` 中的加速 Rollout 策略。

## 4. 环境验证

安装完成后，运行以下命令验证核心库版本：

```python
import torch
import flash_attn
import diffusers

print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Flash Attention: {flash_attn.__version__}")
```
