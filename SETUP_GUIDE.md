# DanceGRPO 训练准备与加速实验指南

本指南详细介绍了如何准备 DanceGRPO 的训练环境、模型和数据集，并提供了运行标准版与高效版（Efficient Rollout）对比实验的步骤。

## 1. 环境准备

首先克隆代码库并安装核心依赖：

```bash
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip install -e .
cd ..
pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0 accelerate wandb tqdm
```

## 2. 模型下载与布局

训练需要 **策略模型 (Qwen2.5-VL/Qwen Image)** 和 **评估模型 (HPSv2)**。

### 2.1 Qwen Image (策略模型)
从 HuggingFace 下载 [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) 或对应的 Transformer 模型，并放置在 `data/qwenimage`：

```bash
mkdir -p data/qwenimage
# 使用 huggingface-cli 或 git lfs 下载
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir data/qwenimage
```

### 2.2 HPSv2 (评估模型)
HPSv2 的权重通常会自动下载，但建议手动放置以避免节点网络问题：
- 将压缩包 `HPS_v2.1_compressed.pt` 放置在项目根目录或指定位置（脚本默认为 `./hps_ckpt/`）。

## 3. 数据集构造与准备

### 3.1 使用 HPDv2 构造 Prompt 文件
我们使用 [ymhao/HPDv2](https://huggingface.co/datasets/ymhao/HPDv2) 训练集中的 Prompt。该数据集包含约 80 万个人类偏好选择，非常适合 RL 训练。

运行以下脚本从 HuggingFace 自动下载并提取前 1000 条唯一的 Prompt：

```bash
python scripts/dataset_preparation/prepare_hpdv2.py
```

执行后，Prompt 将保存在 `assets/prompts.txt` 中。

### 3.2 预处理 Embeddings
为了加速训练并节省显存，需要预先提取文本的 Embedding：

```bash
# 修改脚本中的模型路径和输出路径
bash scripts/preprocess/preprocess_qwen_image_rl_embeddings.sh
```

处理完成后，`data/rl_embeddings/` 目录下会生成：
- `prompt_embed/`: 文本向量文件夹。
- `prompt_attention_mask/`: Attention Mask 文件夹。
- `videos2caption.json`: 数据集索引文件。

## 4. 运行对比实验

我们对比总 Rollout 数量为 8 的两种模式：
- **Standard**: 8 个样本全部由模型推理生成。
- **Efficient**: 6 个样本由模型推理，2 个样本通过“均值引导”猜测生成。

### 4.1 运行标准版
```bash
chmod +x run_standard.sh
./run_standard.sh
```

### 4.2 运行高效版
```bash
chmod +x run_efficient.sh
./run_efficient.sh
```

## 5. 性能监控

实验结果将自动同步至 WandB：
- **Project**: `grpo_comparison`
- **Runs**: `standard_rollout_8` vs `efficient_rollout_6_2`

通过对比 `reward` 曲线判断收敛性是否一致，通过 `step_time` 判断 Efficient 版本带来的推理加速比（预计推理时间减少约 25%）。
