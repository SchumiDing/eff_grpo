"""
对比测试脚本: 同时运行多种 rollout 方法并对比结果
1. 原始方法: 不使用guess,每步都完整推理 (baseline)
2. 平均方向方法: 使用平均方向计算guess
3. 噪声方法: 直接添加随机噪声guess
4. 动量方法: 使用平均方向+动量计算guess
5. 直接复用方法: 直接复用上一轮速度
6. 自适应动量方法: 随时间步自适应调整动量权重
7. MTA D2: 二阶轨迹预测 + 动态动量 + 防过冲加速度耦合 (train_grpo_qwenimage_eff_mta_d2)
8. MTA Alpha: 一阶 v_hat = alpha*v_prev + (1-alpha)*v_group，alpha 由速度一致性 (train_grpo_qwenimage_eff_mta_alpha)
9. MTA AB2: 二阶外推 1.5*v_prev - 0.5*v_prevprev 再与 v_group 融合 (train_grpo_qwenimage_eff_mta_ab2)
10. MTA AB2-2: AB2 + 更长前半融合 + alpha^p 略增 v_group (train_grpo_qwenimage_eff_mta_ab2_2)
11. MTA VarGuess: 组内速度方差自适应 guess 数量，预测仍为 auto 动量 (train_grpo_qwenimage_eff_mta_varguess)
12. MTA ResAB: v_tgt + cos 加权 AB2 加速度项 (train_grpo_qwenimage_eff_mta_resab)
13. MTA Line: v_tgt + cos 加权 (v_prev - v_prevprev) (train_grpo_qwenimage_eff_mta_line)
14. MTA Heun: 前半段 0.5*(v_so + v_tgt) (train_grpo_qwenimage_eff_mta_heun)
15. MTA RAB2: 对 rollout 残差 v - v_group 做强 AB2 外推，再回锚到当前 group mean (train_grpo_qwenimage_eff_mta_rab2)
16. MTA RAB2-Mid: 更保守的 RAB2，缩小残差偏置与幅度上限 (train_grpo_qwenimage_eff_mta_rab2_tuned)
17. MTA RAB2-Tight: 更强保守的 RAB2，进一步收缩 rollout 残差 (train_grpo_qwenimage_eff_mta_rab2_tuned)
18. MTA AB2-Trust: 强 AB2 + smoothed group anchor + trust region 裁剪 (train_grpo_qwenimage_eff_mta_ab2_trust)
"""

import torch
import torch.distributed as dist
import os
import shutil
import argparse
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor
import json
import random
import numpy as np
import time
import gc

# 导入 FSDP 相关组件
from fastvideo.utils.fsdp_util_qwenimage import fsdp_wrapper, FSDPConfig
from fastvideo.utils.rollout_image_dir import rollout_image_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# 导入各 rollout 方法
from train_grpo_qwenimage import sample_reference_model as sample_reference_model_original
from train_grpo_qwenimage_eff import sample_reference_model as sample_reference_model_mean, sd3_time_shift
from train_grpo_qwenimage_noise import sample_reference_model_noise
from train_grpo_qwenimage_eff_mta import sample_reference_model as sample_reference_model_momentum
from train_grpo_qwenimage_mta import sample_reference_model as sample_reference_model_reuse
from train_grpo_qwenimage_eff_mta_auto import sample_reference_model as sample_reference_model_auto
from train_grpo_qwenimage_eff_mta_d2 import sample_reference_model as sample_reference_model_d2
from train_grpo_qwenimage_eff_mta_alpha import sample_reference_model as sample_reference_model_alpha
from train_grpo_qwenimage_eff_mta_ab2 import sample_reference_model as sample_reference_model_ab2
from train_grpo_qwenimage_eff_mta_ab2_2 import sample_reference_model as sample_reference_model_ab2_2
from train_grpo_qwenimage_eff_mta_ab2_ablate import (
    sample_reference_model_ab2_abl_ef04,
    sample_reference_model_ab2_abl_ef06,
    sample_reference_model_ab2_abl_v1,
    sample_reference_model_ab2_abl_strong,
    sample_reference_model_ab2_abl_ef04_strong,
)
from train_grpo_qwenimage_eff_mta_varguess import sample_reference_model as sample_reference_model_varguess
from train_grpo_qwenimage_eff_mta_resab import sample_reference_model as sample_reference_model_resab
from train_grpo_qwenimage_eff_mta_line import sample_reference_model as sample_reference_model_line
from train_grpo_qwenimage_eff_mta_heun import sample_reference_model as sample_reference_model_heun
from train_grpo_qwenimage_eff_mta_rab2 import sample_reference_model_rab2
from train_grpo_qwenimage_eff_mta_rab2_tuned import (
    sample_reference_model_rab2_mid,
    sample_reference_model_rab2_tight,
    sample_reference_model_rab2_g130,
    sample_reference_model_rab2_b020,
    sample_reference_model_rab2_b030,
    sample_reference_model_rab2_b035,
    sample_reference_model_rab2_b040,
    sample_reference_model_rab2_b030_g140,
)
from train_grpo_qwenimage_eff_mta_ab2_trust import sample_reference_model_ab2_trust
from train_grpo_qwenimage_eff_oneach import run_sample_step as run_sample_step_oneach, flux_step, unpack_latents, pack_latents

def set_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/qwenimage", help="Path to qwenimage weights")
    parser.add_argument("--embeddings_path", type=str, default="data/qwenimage/rl_embeddings_anime", help="Path to rl embeddings")
    parser.add_argument("--num_generations", type=int, default=12, help="Number of samples per prompt")
    parser.add_argument(
        "--num_guess",
        type=int,
        default=4,
        help="Per prompt: max rollouts to mock-guess each step (within each prompt's num_generations group)",
    )
    parser.add_argument(
        "--num_guess_min",
        type=int,
        default=4,
        help="Per prompt: min guess count for method varguess (variance-adaptive scheduling).",
    )
    parser.add_argument(
        "--var_guess_tau",
        type=float,
        default=1e-4,
        help="Tau for varguess: maps group velocity variance to guess count.",
    )
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--output_path", type=str, default="./test_results_comparison")
    parser.add_argument(
        "--rollout_image_dir",
        type=str,
        default=None,
        help="Scratch dir for decoded rollout PNGs (before move to --output_path). Default ./images. "
        "Override with env DANCEGRPO_ROLLOUT_IMAGE_DIR to avoid clashing with parallel training.",
    )
    parser.add_argument("--init_same_noise", action="store_true", default=False, help="Use same noise for all samples in a group")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of embeddings to load")
    parser.add_argument("--end_idx", type=int, default=None, help="End index of embeddings to load (None means all)")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of prompts per batch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--methods", type=str, nargs='+', default=["original", "original_14", "ab2_abl_ef04_strong", "reuse", "auto", "d2", "alpha", "ab2", "ab2_2", "rab2", "ab2_trust", "resab", "line", "heun"], 
                        help="Methods to run: ... ab2, ab2_2, ab2_abl_ef04, ab2_abl_ef06, ab2_abl_v1, ab2_abl_strong, varguess, ...")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="SHARD_GRAD_OP", help="FSDP sharding strategy")
    args = parser.parse_args()

    rollout_dir = args.rollout_image_dir or os.environ.get("DANCEGRPO_ROLLOUT_IMAGE_DIR", "./images")
    rollout_dir = os.path.abspath(os.path.expanduser(os.path.normpath(rollout_dir)))
    os.makedirs(rollout_dir, exist_ok=True)
    os.environ["DANCEGRPO_ROLLOUT_IMAGE_DIR"] = rollout_dir

    eq_sampling_steps = int(args.sampling_steps * (args.num_generations-args.num_guess)/args.num_generations + 0.5 )

    # 初始化分布式环境
    if "WORLD_SIZE" in os.environ or "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(f"Distributed initialized: rank {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda")

    # 设置随机种子
    set_seed(args.seed + rank)
    print(f"Random seed set to: {args.seed + rank} for rank {rank}")
    if rank == 0:
        print(f"Rollout image scratch dir (DANCEGRPO_ROLLOUT_IMAGE_DIR): {os.environ.get('DANCEGRPO_ROLLOUT_IMAGE_DIR', './images')}")

    os.makedirs(args.output_path, exist_ok=True)
    
    if args.output_path == "./test_results_comparison":
        args.output_path = f"./test_results_comparison_{args.embeddings_path.split('/')[-1]}"
    
    # 检查哪些方法已经生成过，跳过已有的方法
    available_methods = [
        "original",
        "original_14",
        "mean_direction",
        "noise",
        "momentum",
        "reuse",
        "auto",
        "d2",
        "alpha",
        "ab2",
        "ab2_2",
        "ab2_abl_ef04",
        "ab2_abl_ef06",
        "ab2_abl_v1",
        "ab2_abl_strong",
        "ab2_abl_ef04_strong",
        "varguess",
        "resab",
        "line",
        "heun",
        "rab2",
        "rab2_mid",
        "rab2_tight",
        "rab2_g130",
        "rab2_b020",
        "rab2_b030",
        "rab2_b035",
        "rab2_b040",
        "rab2_b030_g140",
        "ab2_trust",
        "oneach_14",
    ]
    methods_to_run = []
    for method in args.methods:
        if method not in available_methods:
            if rank == 0:
                print(f"Warning: Unknown method '{method}', skipping...")
            continue
        method_dir = os.path.join(args.output_path, method)
        # 多卡模式下，跳过检查稍微复杂，这里我们强制运行或在每个 rank 下检查
        # 为简化，保持原样，但如果是多卡，最好不要在这个阶段直接退出
        methods_to_run.append(method)
        if rank == 0:
            os.makedirs(method_dir, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()
    
    if not methods_to_run:
        if rank == 0:
            print("All requested methods already have results. Exiting...")
        return
    
    if rank == 0:
        print(f"Methods to run: {methods_to_run}")

    # 1. 加载组件
    print("Loading models...")
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from diffusers.models.autoencoders import AutoencoderKLQwenImage
    
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model_path, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    if world_size > 1:
        fsdp_config = FSDPConfig(
            sharding_strategy=args.fsdp_sharding_strategy,
            backward_prefetch="BACKWARD_PRE",
            mixed_precision_dtype=torch.bfloat16
        )
        transformer = fsdp_wrapper(transformer, fsdp_config)
    else:
        transformer = transformer.to(device)
        
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.model_path, 
        subfolder="vae", 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # 加载HPSv2 reward model
    print("Loading HPSv2 reward model...")
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    
    def initialize_hpsv2():
        model_dict = {}
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            './hps_ckpt/open_clip_pytorch_model.bin',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val
        return model_dict
    
    model_dict = initialize_hpsv2()
    hpsv2_model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']
    cp = "./hps_ckpt/HPS_v2.1_compressed.pt"
    
    checkpoint = torch.load(cp, map_location=device)
    hpsv2_model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    hpsv2_model = hpsv2_model.to(device)
    hpsv2_model.eval()
    print("HPSv2 model loaded successfully!")
    
    # 2. 加载 RL embeddings
    print(f"Loading embeddings from {args.embeddings_path}...")
    json_path = os.path.join(args.embeddings_path, "videos2caption.json")
    prompt_embed_dir = os.path.join(args.embeddings_path, "prompt_embed")
    prompt_attention_mask_dir = os.path.join(args.embeddings_path, "prompt_attention_mask")
    
    with open(json_path, "r") as f:
        data_anno = json.load(f)
    
    end_idx = args.end_idx if args.end_idx is not None else len(data_anno)
    data_anno = data_anno[args.start_idx:end_idx]
    
    # 根据 rank 进行数据分片
    num_total = len(data_anno)
    per_rank = (num_total + world_size - 1) // world_size
    data_anno = data_anno[rank * per_rank : (rank + 1) * per_rank]
    
    print(f"Rank {rank}/{world_size}: Processing {len(data_anno)} prompts (global index {args.start_idx + rank * per_rank} to {min(args.start_idx + (rank + 1) * per_rank, end_idx)-1})")
    
    # 加载所有 embeddings
    all_prompt_embeds = []
    all_prompt_attention_masks = []
    all_captions = []
    all_original_lengths = []
    
    print("Loading embeddings...")
    for item in tqdm(data_anno, desc="Loading embeddings"):
        prompt_embed = torch.load(
            os.path.join(prompt_embed_dir, item["prompt_embed_path"]),
            map_location="cpu",
            weights_only=True,
        )
        prompt_attention_mask = torch.load(
            os.path.join(prompt_attention_mask_dir, item["prompt_attention_mask"]),
            map_location="cpu",
            weights_only=True,
        )
        all_prompt_embeds.append(prompt_embed)
        all_prompt_attention_masks.append(prompt_attention_mask)
        all_captions.append(item['caption'])
        all_original_lengths.append(item['original_length'])
    
    all_prompt_embeds = torch.stack(all_prompt_embeds, dim=0)
    all_prompt_attention_masks = torch.stack(all_prompt_attention_masks, dim=0)
    
    print(f"Loaded embeddings shape: {all_prompt_embeds.shape}")
    print(f"Total prompts: {len(all_captions)}")
    
    num_prompts = len(all_captions)
    
    # 创建 mock args
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    mock_args = MockArgs(
        num_guess=args.num_guess,
        num_guess_min=args.num_guess_min,
        var_guess_tau=args.var_guess_tau,
        num_generations=args.num_generations,
        eta=args.eta,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        init_same_noise=args.init_same_noise,
        w=args.width,
        h=args.height,
        t=1,
        use_hpsv2=True,  # 启用HPSv2计算reward
        use_hpsv3=False,
        use_pickscore=False,
    )
    
    # 获取分布式下的 rank, 上面已经定义过，移除这里的 rank = 0 阴影
    current_rank = rank

    def _png_to_judge_path(png_path: str) -> str:
        return png_path[:-4] + ".judge.json" if png_path.endswith(".png") else png_path + ".judge.json"

    def _read_judge_hps(judge_path: str):
        if not os.path.isfile(judge_path):
            return None
        try:
            with open(judge_path, "r") as f:
                data = json.load(f)
            if "hps_score" not in data:
                return None
            return float(data["hps_score"])
        except (json.JSONDecodeError, TypeError, ValueError, OSError):
            return None

    def _write_judge_hps(png_path: str, hps_score: float) -> None:
        jp = _png_to_judge_path(png_path)
        with open(jp, "w") as f:
            json.dump({"hps_score": hps_score}, f)

    def _find_gen_png(prompt_dir: str, gen_idx: int, has_mock_flags: bool):
        if not has_mock_flags:
            p = os.path.join(prompt_dir, f"gen_{gen_idx}.png")
            return p if os.path.isfile(p) else None
        if not os.path.isdir(prompt_dir):
            return None
        for f in os.listdir(prompt_dir):
            if f.startswith(f"gen_{gen_idx}_") and f.endswith(".png"):
                return os.path.join(prompt_dir, f)
        return None

    def _collect_batch_png_paths(
        method_key: str, has_mock_flags: bool, start_prompt_idx: int, start_local_idx: int, end_local_idx: int
    ):
        """与采样顺序一致的 PNG 路径列表；任一图片缺失则返回 None。"""
        paths = []
        for i in range(start_local_idx, end_local_idx):
            prompt_idx = start_prompt_idx + (i - start_local_idx)
            prompt_dir = os.path.join(args.output_path, method_key, f"prompt_{prompt_idx}")
            for gen_idx in range(args.num_generations):
                p = _find_gen_png(prompt_dir, gen_idx, has_mock_flags)
                if p is None:
                    return None
                paths.append(p)
        return paths

    def _all_judge_valid(png_paths: list) -> bool:
        for p in png_paths:
            if _read_judge_hps(_png_to_judge_path(p)) is None:
                return False
        return True

    def _load_batch_rewards_from_judge(png_paths: list) -> torch.Tensor:
        vals = [_read_judge_hps(_png_to_judge_path(p)) for p in png_paths]
        return torch.tensor(vals, dtype=torch.float32)

    def _save_batch_judge_sidecars(png_paths: list, rewards_1d: torch.Tensor) -> None:
        r = rewards_1d.detach().cpu().flatten()
        if len(png_paths) != r.numel():
            raise ValueError(
                f"judge sidecar count mismatch: {len(png_paths)} paths vs {r.numel()} scores"
            )
        for p, val in zip(png_paths, r):
            _write_judge_hps(p, float(val.item()))

    def _expected_png_paths_for_batch(
        method_key: str,
        has_mock_flags: bool,
        start_prompt_idx: int,
        start_local_idx: int,
        end_local_idx: int,
        mock_flag_sums=None,
    ):
        """move 完成后磁盘上的 PNG 路径（与 batch_rewards 顺序一致）。"""
        g = args.num_generations
        paths = []
        for i in range(start_local_idx, end_local_idx):
            prompt_idx = start_prompt_idx + (i - start_local_idx)
            prompt_dir = os.path.join(args.output_path, method_key, f"prompt_{prompt_idx}")
            for gen_idx in range(g):
                if not has_mock_flags:
                    paths.append(os.path.join(prompt_dir, f"gen_{gen_idx}.png"))
                else:
                    local_idx = (i - start_local_idx) * g + gen_idx
                    guess_count = int(mock_flag_sums[local_idx])
                    paths.append(os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png"))
        return paths

    def move_original_batch_to_output(start_prompt_idx: int, batch_num_prompts: int) -> None:
        """本 batch 采样完成后立刻写入 output_path/original，再进入下一 batch / 下一 method。"""
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}.png")
            prompt_dir = os.path.join(args.output_path, "original", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_mean_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "mean_direction", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_noise_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}_noise.png")
            prompt_dir = os.path.join(args.output_path, "noise", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_momentum_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "momentum", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)
    
    def move_reuse_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "reuse", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_auto_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "auto", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_d2_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "d2", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_alpha_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "alpha", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_ab2_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "ab2", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_ab2_2_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "ab2_2", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def _move_ab2_ablate_batch_to_output(subdir: str):
        def _fn(start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums) -> None:
            g = args.num_generations
            n = batch_num_prompts * g
            for local_idx in range(n):
                prompt_idx = start_prompt_idx + local_idx // g
                gen_idx = local_idx % g
                guess_count = int(mock_flag_sums[local_idx])
                src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
                prompt_dir = os.path.join(args.output_path, subdir, f"prompt_{prompt_idx}")
                os.makedirs(prompt_dir, exist_ok=True)
                dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
                if os.path.isfile(src):
                    shutil.move(src, dst)

        return _fn

    move_ab2_abl_ef04_batch_to_output = _move_ab2_ablate_batch_to_output("ab2_abl_ef04")
    move_ab2_abl_ef06_batch_to_output = _move_ab2_ablate_batch_to_output("ab2_abl_ef06")
    move_ab2_abl_v1_batch_to_output = _move_ab2_ablate_batch_to_output("ab2_abl_v1")
    move_ab2_abl_strong_batch_to_output = _move_ab2_ablate_batch_to_output("ab2_abl_strong")
    move_ab2_abl_ef04_strong_batch_to_output = _move_ab2_ablate_batch_to_output("ab2_abl_ef04_strong")
    move_rab2_batch_to_output = _move_ab2_ablate_batch_to_output("rab2")
    move_rab2_mid_batch_to_output = _move_ab2_ablate_batch_to_output("rab2_mid")
    move_rab2_tight_batch_to_output = _move_ab2_ablate_batch_to_output("rab2_tight")
    move_rab2_g130_batch_to_output = _move_ab2_ablate_batch_to_output("rab2_g130")
    move_rab2_b020_batch_to_output = _move_ab2_ablate_batch_to_output("rab2_b020")
    move_rab2_b030_batch_to_output = _move_ab2_ablate_batch_to_output("rab2_b030")
    move_rab2_b035_batch_to_output = _move_ab2_ablate_batch_to_output("rab2_b035")
    move_rab2_b040_batch_to_output = _move_ab2_ablate_batch_to_output("rab2_b040")
    move_rab2_b030_g140_batch_to_output = _move_ab2_ablate_batch_to_output("rab2_b030_g140")
    move_ab2_trust_batch_to_output = _move_ab2_ablate_batch_to_output("ab2_trust")

    def move_varguess_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "varguess", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_resab_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "resab", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_line_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "line", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_heun_batch_to_output(
        start_prompt_idx: int, batch_num_prompts: int, mock_flag_sums
    ) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            guess_count = int(mock_flag_sums[local_idx])
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_guess{guess_count}.png")
            prompt_dir = os.path.join(args.output_path, "heun", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)
    
    def move_original_14_batch_to_output(start_prompt_idx: int, batch_num_prompts: int) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}.png")
            prompt_dir = os.path.join(args.output_path, "original_14", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)

    def move_oneach_14_batch_to_output(start_prompt_idx: int, batch_num_prompts: int) -> None:
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            src = rollout_image_file(f"qwenimage_{current_rank}_{local_idx}_oneach14.png")
            prompt_dir = os.path.join(args.output_path, "oneach_14", f"prompt_{prompt_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            dst = os.path.join(prompt_dir, f"gen_{gen_idx}.png")
            if os.path.isfile(src):
                shutil.move(src, dst)
    
    # 按batch处理所有prompts
    total_samples = num_prompts * args.num_generations
    print(f"\n{'='*80}")
    print(f"Total samples: {total_samples} ({num_prompts} prompts × {args.num_generations} generations)")
    print(f"Batch size: {args.batch_size} prompts per batch")
    print(f"{'='*80}\n")
    
    # 初始化结果容器 - 使用字典管理所有方法
    results = {
        "original": {"rewards": [], "latents": [], "log_probs": [], "txt_seq_lens": []},
        "original_14": {"rewards": [], "latents": [], "log_probs": [], "txt_seq_lens": []},
        "mean_direction": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "noise": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "momentum": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "reuse": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "auto": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "d2": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "alpha": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "ab2": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "ab2_2": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "ab2_abl_ef04": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "ab2_abl_ef06": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "ab2_abl_v1": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "ab2_abl_strong": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "ab2_abl_ef04_strong": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "varguess": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "resab": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "line": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "heun": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2_mid": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2_tight": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2_g130": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2_b020": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2_b030": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2_b035": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2_b040": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "rab2_b030_g140": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "ab2_trust": {"rewards": [], "latents": [], "log_probs": [], "mock_flags": [], "txt_seq_lens": []},
        "oneach_14": {"rewards": [], "latents": [], "log_probs": [], "txt_seq_lens": []},
    }
    
    # 计算batch数量
    num_batches = (len(data_anno) + args.batch_size - 1) // args.batch_size
    
    # 3. 按batch运行各种方法
    # 这里将 all_captions 替换为 data_anno
    # 我们需要在 rank 0 收集所有的 rewards
    
    for method in methods_to_run:
        method_name_map = {
            "original": ("METHOD 1: Original (No Guess - Baseline)", sample_reference_model_original, move_original_batch_to_output, False, args.sampling_steps),
            "original_14": ("METHOD 1.5: Original (14 Steps - Same Compute Baseline)", sample_reference_model_original, move_original_14_batch_to_output, False, eq_sampling_steps),
            "mean_direction": ("METHOD 2: Mean Direction", sample_reference_model_mean, move_mean_batch_to_output, True, args.sampling_steps),
            "noise": ("METHOD 3: Random Noise", sample_reference_model_noise, move_noise_batch_to_output, True, args.sampling_steps),
            "momentum": ("METHOD 4: Momentum", sample_reference_model_momentum, move_momentum_batch_to_output, True, args.sampling_steps),
            "reuse": ("METHOD 5: Reuse Velocity", sample_reference_model_reuse, move_reuse_batch_to_output, True, args.sampling_steps),
            "auto": ("METHOD 6: Auto Momentum", sample_reference_model_auto, move_auto_batch_to_output, True, args.sampling_steps),
            "d2": ("METHOD 7: MTA D2 (2nd-order + adaptive momentum)", sample_reference_model_d2, move_d2_batch_to_output, True, args.sampling_steps),
            "alpha": ("METHOD 8: MTA Alpha (consistency-mixed v_prev + v_group)", sample_reference_model_alpha, move_alpha_batch_to_output, True, args.sampling_steps),
            "ab2": ("METHOD 9: MTA AB2 (2nd-order extrapolation + v_group blend)", sample_reference_model_ab2, move_ab2_batch_to_output, True, args.sampling_steps),
            "ab2_2": ("METHOD 9b: MTA AB2-2 (early T/2, alpha^0.99 mix)", sample_reference_model_ab2_2, move_ab2_2_batch_to_output, True, args.sampling_steps),
            "ab2_abl_ef04": (
                "METHOD 9c: AB2 ablate early_frac=0.4 (c1=1.5,c2=-0.5)",
                sample_reference_model_ab2_abl_ef04,
                move_ab2_abl_ef04_batch_to_output,
                True,
                args.sampling_steps,
            ),
            "ab2_abl_ef06": (
                "METHOD 9d: AB2 ablate early_frac=0.6 (c1=1.5,c2=-0.5)",
                sample_reference_model_ab2_abl_ef06,
                move_ab2_abl_ef06_batch_to_output,
                True,
                args.sampling_steps,
            ),
            "ab2_abl_v1": (
                "METHOD 9e: AB2 ablate first-order v_so=v_prev (early=0.5)",
                sample_reference_model_ab2_abl_v1,
                move_ab2_abl_v1_batch_to_output,
                True,
                args.sampling_steps,
            ),
            "ab2_abl_strong": (
                "METHOD 9f: AB2 ablate strong extrap 2*v_prev-1*v_pp (early=0.5)",
                sample_reference_model_ab2_abl_strong,
                move_ab2_abl_strong_batch_to_output,
                True,
                args.sampling_steps,
            ),
            "ab2_abl_ef04_strong": (
                "METHOD 9g: AB2 combo early=0.4 + v_so=2*v_prev-1*v_pp",
                sample_reference_model_ab2_abl_ef04_strong,
                move_ab2_abl_ef04_strong_batch_to_output,
                True,
                args.sampling_steps,
            ),
            "varguess": ("METHOD 10: MTA VarGuess (variance-adaptive num_guess + auto velocity)", sample_reference_model_varguess, move_varguess_batch_to_output, True, args.sampling_steps),
            "resab": ("METHOD 11: MTA ResAB (v_tgt + eta*(v_so-v_prev), eta from cosine)", sample_reference_model_resab, move_resab_batch_to_output, True, args.sampling_steps),
            "line": ("METHOD 12: MTA Line (v_tgt + w*(v_prev-v_pp), w from cosine)", sample_reference_model_line, move_line_batch_to_output, True, args.sampling_steps),
            "heun": ("METHOD 13: MTA Heun (early: 0.5*(v_so+v_tgt))", sample_reference_model_heun, move_heun_batch_to_output, True, args.sampling_steps),
            "rab2": ("METHOD 14: MTA RAB2 (group-centered residual AB2)", sample_reference_model_rab2, move_rab2_batch_to_output, True, args.sampling_steps),
            "rab2_mid": ("METHOD 14b: MTA RAB2-Mid (residual bias=0.20, growth=1.25)", sample_reference_model_rab2_mid, move_rab2_mid_batch_to_output, True, args.sampling_steps),
            "rab2_tight": ("METHOD 14c: MTA RAB2-Tight (residual bias=0.15, growth=1.20)", sample_reference_model_rab2_tight, move_rab2_tight_batch_to_output, True, args.sampling_steps),
            "rab2_g130": ("METHOD 14d: MTA RAB2-G130 (bias=0.25, growth=1.30)", sample_reference_model_rab2_g130, move_rab2_g130_batch_to_output, True, args.sampling_steps),
            "rab2_b020": ("METHOD 14e: MTA RAB2-B020 (bias=0.20, growth=1.35)", sample_reference_model_rab2_b020, move_rab2_b020_batch_to_output, True, args.sampling_steps),
            "rab2_b030": ("METHOD 14f: MTA RAB2-B030 (bias=0.30, growth=1.35)", sample_reference_model_rab2_b030, move_rab2_b030_batch_to_output, True, args.sampling_steps),
            "rab2_b035": ("METHOD 14g: MTA RAB2-B035 (bias=0.35, growth=1.35)", sample_reference_model_rab2_b035, move_rab2_b035_batch_to_output, True, args.sampling_steps),
            "rab2_b040": ("METHOD 14h: MTA RAB2-B040 (bias=0.40, growth=1.35)", sample_reference_model_rab2_b040, move_rab2_b040_batch_to_output, True, args.sampling_steps),
            "rab2_b030_g140": ("METHOD 14i: MTA RAB2-B030-G140 (bias=0.30, growth=1.40)", sample_reference_model_rab2_b030_g140, move_rab2_b030_g140_batch_to_output, True, args.sampling_steps),
            "ab2_trust": ("METHOD 15: MTA AB2-Trust (EMA group anchor + confidence trust region)", sample_reference_model_ab2_trust, move_ab2_trust_batch_to_output, True, args.sampling_steps),
        }
        
        if method == "oneach_14":
            # 特殊处理 oneach_14，因为它不使用 sample_reference_model 系列函数
            if rank == 0:
                print(f"\n{'='*80}")
                print("METHOD 6.5: Oneach (14 Steps - Full Inference + Mean Correction)")
                print(f"{'='*80}\n")
            
            for batch_idx in range(num_batches):
                set_seed(42 + rank)
                mock_args.sampling_steps = eq_sampling_steps
                
                start_local_idx = batch_idx * args.batch_size
                end_local_idx = min((batch_idx + 1) * args.batch_size, len(data_anno))
                batch_num_prompts = end_local_idx - start_local_idx
                start_prompt_idx = args.start_idx + rank * per_rank + start_local_idx
                
                # 检查是否已生成
                need_generate = False
                for i in range(start_local_idx, end_local_idx):
                    for gen_idx in range(args.num_generations):
                        prompt_idx = start_prompt_idx + (i - start_local_idx)
                        dst = os.path.join(args.output_path, "oneach_14", f"prompt_{prompt_idx}", f"gen_{gen_idx}.png")
                        if not os.path.isfile(dst):
                            need_generate = True
                            break
                    if need_generate: break
                
                if not need_generate:
                    png_paths = _collect_batch_png_paths(
                        "oneach_14", False, start_prompt_idx, start_local_idx, end_local_idx
                    )
                    if png_paths is not None and _all_judge_valid(png_paths):
                        if rank == 0:
                            print(
                                f"[oneach_14] Batch {batch_idx + 1} images + judge cache found, reusing HPS scores..."
                            )
                        batch_rewards = _load_batch_rewards_from_judge(png_paths)
                        results["oneach_14"]["rewards"].append(batch_rewards)
                        results["oneach_14"]["log_probs"].append(
                            torch.zeros(batch_num_prompts * args.num_generations, 1).cpu()
                        )
                        results["oneach_14"]["txt_seq_lens"].append(
                            torch.zeros(batch_num_prompts * args.num_generations).cpu()
                        )
                        continue

                    if rank == 0:
                        print(
                            f"[oneach_14] Batch {batch_idx + 1} images exist; recomputing HPS (missing or stale .judge.json)..."
                        )

                    batch_rewards_list = []
                    png_paths_ordered = []
                    for i in range(start_local_idx, end_local_idx):
                        for gen_idx in range(args.num_generations):
                            prompt_idx = start_prompt_idx + (i - start_local_idx)
                            img_path = os.path.join(
                                args.output_path, "oneach_14", f"prompt_{prompt_idx}", f"gen_{gen_idx}.png"
                            )
                            from PIL import Image

                            img = Image.open(img_path).convert("RGB")
                            image_input = preprocess_val(img).unsqueeze(0).to(device)
                            text_input = tokenizer([all_captions[i]]).to(device)
                            with torch.no_grad():
                                with torch.amp.autocast("cuda"):
                                    outputs = hpsv2_model(image_input, text_input)
                                    hps_score = torch.diagonal(
                                        outputs["image_features"] @ outputs["text_features"].T
                                    )
                                    batch_rewards_list.append(hps_score)
                            png_paths_ordered.append(img_path)

                    batch_rewards = torch.cat(batch_rewards_list)
                    _save_batch_judge_sidecars(png_paths_ordered, batch_rewards)
                    results["oneach_14"]["rewards"].append(batch_rewards.cpu())
                    results["oneach_14"]["log_probs"].append(
                        torch.zeros(batch_num_prompts * args.num_generations, 1).cpu()
                    )
                    results["oneach_14"]["txt_seq_lens"].append(
                        torch.zeros(batch_num_prompts * args.num_generations).cpu()
                    )
                    continue

                if rank == 0:
                    print(f"[oneach_14] Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{start_prompt_idx + batch_num_prompts - 1})...")
                
                encoder_hidden_states_list = []
                prompt_attention_masks_list = []
                original_length_list = []
                captions_expanded = []
                
                for i in range(start_local_idx, end_local_idx):
                    for _ in range(args.num_generations):
                        encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                        prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                        original_length_list.append(all_original_lengths[i])
                        captions_expanded.append(all_captions[i])
                
                encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
                prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
                original_length = torch.tensor(original_length_list).to(device)
                B = encoder_hidden_states.shape[0]

                if args.init_same_noise:
                    input_latents = torch.randn((B//args.num_generations, 1, 16, args.height//8, args.width//8), device=device, dtype=torch.bfloat16).repeat_interleave(args.num_generations, dim=0)
                else:
                    input_latents = torch.randn((B, 1, 16, args.height//8, args.width//8), device=device, dtype=torch.bfloat16)
                
                input_latents_new = pack_latents(input_latents, B, 16, 2*(args.height//16), 2*(args.width//16))

                sigma_schedule_14 = torch.linspace(1, 0, eq_sampling_steps + 1)
                sigma_schedule_14 = sd3_time_shift(args.shift, sigma_schedule_14)
                
                img_shapes = [[(1, args.height // 8 // 2, args.width // 8 // 2)]]
                txt_seq_lens = [encoder_hidden_states.shape[1]] * B

                with torch.no_grad():
                    z, latents, all_latents_tensor, all_log_probs_tensor, all_mock_flags_tensor = run_sample_step_oneach(
                        mock_args,
                        input_latents_new,
                        tqdm(range(eq_sampling_steps), desc="Sampling", disable=(rank!=0)),
                        sigma_schedule_14,
                        transformer,
                        encoder_hidden_states,
                        prompt_attention_masks,
                        img_shapes,
                        txt_seq_lens,
                        grpo_sample=True,
                    )
                
                vae.enable_tiling()
                image_processor = VaeImageProcessor(16)
                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        latents_unpacked = unpack_latents(latents, args.height, args.width, 8)
                        latents_unpacked = latents_unpacked.to(vae.dtype)
                        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(device, torch.bfloat16)
                        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(device, torch.bfloat16)
                        latents_unpacked = latents_unpacked / latents_std + latents_mean
                        images = vae.decode(latents_unpacked, return_dict=False)[0][:, :, 0]
                        decoded_images = image_processor.postprocess(images)
                        for idx in range(B):
                            decoded_images[idx].save(rollout_image_file(f"qwenimage_{current_rank}_{idx}_oneach14.png"))

                all_rewards = []
                with torch.no_grad():
                    for idx in range(B):
                        image = preprocess_val(decoded_images[idx]).unsqueeze(0).to(device)
                        text = tokenizer([captions_expanded[idx]]).to(device)
                        with torch.amp.autocast('cuda'):
                            outputs = hpsv2_model(image, text)
                            hps_score = torch.diagonal(outputs["image_features"] @ outputs["text_features"].T)
                            all_rewards.append(hps_score)
                batch_rewards = torch.cat(all_rewards)

                move_oneach_14_batch_to_output(start_prompt_idx, batch_num_prompts)
                exp_png = _expected_png_paths_for_batch(
                    "oneach_14", False, start_prompt_idx, start_local_idx, end_local_idx, None
                )
                _save_batch_judge_sidecars(exp_png, batch_rewards)

                results["oneach_14"]["rewards"].append(batch_rewards.cpu())
                results["oneach_14"]["latents"].append(all_latents_tensor.cpu())
                results["oneach_14"]["log_probs"].append(all_log_probs_tensor.cpu())
                results["oneach_14"]["txt_seq_lens"].append(torch.tensor(txt_seq_lens))
                
                del encoder_hidden_states, prompt_attention_masks, original_length, input_latents, input_latents_new, latents, all_latents_tensor, all_log_probs_tensor, all_mock_flags_tensor, decoded_images, images, all_rewards, batch_rewards
                torch.cuda.empty_cache()
                gc.collect()

                if rank == 0:
                    print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
            continue

        display_name, sample_fn, move_fn, has_mock_flags, current_sampling_steps = method_name_map[method]
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(display_name)
            print(f"{'='*80}\n")
        
        for batch_idx in range(num_batches):
            # 每个batch前重置seed以确保可复现性
            set_seed(42 + rank) # 加上 rank 偏移
            mock_args.sampling_steps = current_sampling_steps
            
            start_local_idx = batch_idx * args.batch_size
            end_local_idx = min((batch_idx + 1) * args.batch_size, len(data_anno))
            batch_num_prompts = end_local_idx - start_local_idx
            
            # 全局索引，用于文件名
            start_prompt_idx = args.start_idx + rank * per_rank + start_local_idx
            
            if rank == 0:
                print(f"[{method}] Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{start_prompt_idx + batch_num_prompts - 1})...")
            
            # 准备当前batch的数据
            encoder_hidden_states_list = []
            prompt_attention_masks_list = []
            original_length_list = []
            captions_expanded = []
            
            for i in range(start_local_idx, end_local_idx):
                for _ in range(args.num_generations):
                    encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                    prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                    original_length_list.append(all_original_lengths[i])
                    captions_expanded.append(all_captions[i])
            
            encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
            prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
            original_length = torch.tensor(original_length_list).to(device)
            
            # 运行采样
            need_generate = False
            for i in range(start_local_idx, end_local_idx):
                for gen_idx in range(args.num_generations):
                    prompt_idx = start_prompt_idx + (i - start_local_idx)
                    # 确定预期的文件名
                    if not has_mock_flags:
                        if method == "oneach_14":
                            dst = os.path.join(args.output_path, method, f"prompt_{prompt_idx}", f"gen_{gen_idx}.png")
                        elif method == "original_14":
                            dst = os.path.join(args.output_path, "original_14", f"prompt_{prompt_idx}", f"gen_{gen_idx}.png")
                        else: # original
                            dst = os.path.join(args.output_path, "original", f"prompt_{prompt_idx}", f"gen_{gen_idx}.png")
                    else:
                        # 对于带有 guess 的方法，由于不知道 guess 步数，我们检查目录是否存在且包含 png
                        prompt_dir = os.path.join(args.output_path, method, f"prompt_{prompt_idx}")
                        dst = None
                        if os.path.isdir(prompt_dir):
                            files = os.listdir(prompt_dir)
                            for f in files:
                                if f.startswith(f"gen_{gen_idx}_") and f.endswith(".png"):
                                    dst = os.path.join(prompt_dir, f)
                                    break
                    
                    if dst is None or not os.path.isfile(dst):
                        need_generate = True
                        break
                if need_generate:
                    break

            if not need_generate:
                png_paths = _collect_batch_png_paths(
                    method, has_mock_flags, start_prompt_idx, start_local_idx, end_local_idx
                )
                if png_paths is not None and _all_judge_valid(png_paths):
                    if rank == 0:
                        print(
                            f"[{method}] Batch {batch_idx + 1}: images + judge cache found, reusing HPS scores..."
                        )
                    batch_rewards = _load_batch_rewards_from_judge(png_paths)
                    results[method]["rewards"].append(batch_rewards)
                else:
                    if rank == 0:
                        print(
                            f"[{method}] All images for batch {batch_idx + 1} exist; recomputing HPS "
                            f"(missing or stale .judge.json)..."
                        )

                    batch_rewards_list = []
                    png_paths_ordered = []
                    for i in range(start_local_idx, end_local_idx):
                        for gen_idx in range(args.num_generations):
                            prompt_idx = start_prompt_idx + (i - start_local_idx)
                            prompt_dir = os.path.join(args.output_path, method, f"prompt_{prompt_idx}")

                            img_path = None
                            if not has_mock_flags:
                                img_path = os.path.join(prompt_dir, f"gen_{gen_idx}.png")
                            else:
                                files = os.listdir(prompt_dir)
                                for f in files:
                                    if f.startswith(f"gen_{gen_idx}_") and f.endswith(".png"):
                                        img_path = os.path.join(prompt_dir, f)
                                        break

                            from PIL import Image

                            img = Image.open(img_path).convert("RGB")
                            image_input = preprocess_val(img).unsqueeze(0).to(device)
                            text_input = tokenizer([all_captions[i]]).to(device)
                            with torch.no_grad():
                                with torch.amp.autocast("cuda"):
                                    outputs = hpsv2_model(image_input, text_input)
                                    hps_score = torch.diagonal(
                                        outputs["image_features"] @ outputs["text_features"].T
                                    )
                                    batch_rewards_list.append(hps_score)
                            png_paths_ordered.append(img_path)

                    batch_rewards = torch.cat(batch_rewards_list)
                    _save_batch_judge_sidecars(png_paths_ordered, batch_rewards)
                    results[method]["rewards"].append(batch_rewards.cpu())

                # 如果是带 flags 的方法，我们还需要尝试从文件名恢复 guess 步数，以便后续统计
                if has_mock_flags:
                    mock_flags_list = []
                    for i in range(start_local_idx, end_local_idx):
                        for gen_idx in range(args.num_generations):
                            prompt_idx = start_prompt_idx + (i - start_local_idx)
                            prompt_dir = os.path.join(args.output_path, method, f"prompt_{prompt_idx}")
                            files = os.listdir(prompt_dir)
                            guess_val = 0
                            for f in files:
                                if f.startswith(f"gen_{gen_idx}_guess") and f.endswith(".png"):
                                    # 提取 guess 数字，例如 gen_0_guess9.png -> 9
                                    try:
                                        guess_val = int(f.split("guess")[-1].split(".")[0])
                                    except:
                                        guess_val = 0
                                    break

                            # Create dummy mock_flags with correct shape
                            dummy_flags = torch.zeros(current_sampling_steps, dtype=torch.float32)
                            if guess_val > 0:
                                dummy_flags[:min(guess_val, current_sampling_steps)] = 1.0
                            mock_flags_list.append(dummy_flags)  # Each is (current_sampling_steps,)
                    batch_mock_flags = torch.stack(mock_flags_list, dim=0)  # Shape: (B_total, current_sampling_steps)
                    results[method]["mock_flags"].append(batch_mock_flags.cpu())
                
                # 对于跳过的 batch，我们也需要添加 dummy log_probs 和 txt_seq_lens 以保持长度一致
                B_total = batch_num_prompts * args.num_generations
                results[method]["log_probs"].append(torch.zeros(B_total, current_sampling_steps).cpu())  # FIXED: was (B_total, 1)
                results[method]["txt_seq_lens"].append(torch.zeros(B_total).cpu())
                
                continue

            # 运行采样 (原有逻辑)
            if has_mock_flags:
                batch_rewards, batch_latents, batch_log_probs, _, batch_txt_seq_lens, batch_mock_flags = sample_fn(
                    mock_args,
                    device,
                    transformer,
                    vae,
                    encoder_hidden_states,
                    prompt_attention_masks,
                    original_length,
                    reward_model=hpsv2_model,
                    tokenizer=tokenizer,
                    caption=captions_expanded,
                    preprocess_val=preprocess_val,
                )
                batch_mock_flags_cpu = batch_mock_flags.sum(dim=1).cpu().numpy()
                move_fn(start_prompt_idx, batch_num_prompts, batch_mock_flags_cpu)
                results[method]["mock_flags"].append(batch_mock_flags.cpu()) # Offload to CPU
                exp_png = _expected_png_paths_for_batch(
                    method,
                    True,
                    start_prompt_idx,
                    start_local_idx,
                    end_local_idx,
                    batch_mock_flags_cpu,
                )
                _save_batch_judge_sidecars(exp_png, batch_rewards)

                # 清理显存
                del batch_latents, batch_mock_flags
            else:
                batch_rewards, batch_latents, batch_log_probs, _, batch_txt_seq_lens = sample_fn(
                    mock_args,
                    device,
                    transformer,
                    vae,
                    encoder_hidden_states,
                    prompt_attention_masks,
                    original_length,
                    reward_model=hpsv2_model,
                    tokenizer=tokenizer,
                    caption=captions_expanded,
                    preprocess_val=preprocess_val,
                )
                move_fn(start_prompt_idx, batch_num_prompts)
                exp_png = _expected_png_paths_for_batch(
                    method,
                    False,
                    start_prompt_idx,
                    start_local_idx,
                    end_local_idx,
                    None,
                )
                _save_batch_judge_sidecars(exp_png, batch_rewards)

                # 清理显存
                del batch_latents
            
            results[method]["rewards"].append(batch_rewards.cpu()) # Offload to CPU
            results[method]["log_probs"].append(batch_log_probs.cpu()) # Offload to CPU
            results[method]["txt_seq_lens"].append(batch_txt_seq_lens.cpu() if torch.is_tensor(batch_txt_seq_lens) else batch_txt_seq_lens)
            
            # 彻底清理当前 batch 占用的显存
            del encoder_hidden_states, prompt_attention_masks, original_length
            del batch_rewards, batch_log_probs, batch_txt_seq_lens
            torch.cuda.empty_cache()
            gc.collect()
            
            if rank == 0:
                print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
        
        # 合并当前方法的所有batch结果
        results[method]["rewards"] = torch.cat(results[method]["rewards"], dim=0)
        # results[method]["latents"] = torch.cat(results[method]["latents"], dim=0)
        results[method]["log_probs"] = torch.cat(results[method]["log_probs"], dim=0)
        # txt_seq_lens 可能是 list，如果是张量则 cat
        if len(results[method]["txt_seq_lens"]) > 0 and torch.is_tensor(results[method]["txt_seq_lens"][0]):
            results[method]["txt_seq_lens"] = torch.cat(results[method]["txt_seq_lens"], dim=0)
            
        if has_mock_flags:
            results[method]["mock_flags"] = torch.cat(results[method]["mock_flags"], dim=0)
        
        # 深度清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        if rank == 0:
            print(f"Method {method} all batches completed!")
            print(f"  Images -> {os.path.join(args.output_path, method)}\n")
    
    # 获取原始完整的 all_captions 以备后续报告使用 (仅 rank 0 需要)
    if rank == 0:
        with open(json_path, "r") as f:
            all_captions_full = [item['caption'] for item in json.load(f)[args.start_idx:end_idx]]
            num_prompts_full = len(all_captions_full)
    
    # 同步汇总所有 rank 的结果 (仅汇总 rewards 和统计信息)
    if world_size > 1:
        dist.barrier()
        for method in methods_to_run:
            has_flags = "mock_flags" in results[method] and len(results[method]["mock_flags"]) > 0
            
            # 汇总 rewards
            local_rewards = results[method]["rewards"].to(device)
            gather_list = [torch.zeros_like(local_rewards) for _ in range(world_size)] if rank == 0 else None
            dist.gather(local_rewards, gather_list, dst=0)
            if rank == 0:
                results[method]["rewards"] = torch.cat(gather_list, dim=0)
            
            # 汇总 mock_flags
            if has_flags:
                local_flags = results[method]["mock_flags"].to(device)
                gather_list_flags = [torch.zeros_like(local_flags) for _ in range(world_size)] if rank == 0 else None
                dist.gather(local_flags, gather_list_flags, dst=0)
                if rank == 0:
                    results[method]["mock_flags"] = torch.cat(gather_list_flags, dim=0)
        
        if rank != 0:
            # 非主进程在同步完后可以退出或等待
            print(f"Rank {rank} finished inference.")
            dist.barrier()
            return
    
    # 统计信息计算 (以下代码仅在 rank 0 执行)
    num_prompts = num_prompts_full # 使用全局总数
    all_captions = all_captions_full
    
    # 9. 计算和比较HPSv2 rewards
    # ... (原有统计逻辑，但需保证数据长度匹配)

    
    # 计算guess步数统计 (确保这些方法在结果中且有 mock_flags)
    num_guess_steps_mean = results["mean_direction"]["mock_flags"].sum(dim=1).cpu().numpy() if ("mean_direction" in methods_to_run and "mock_flags" in results["mean_direction"]) else None
    num_guess_steps_noise = results["noise"]["mock_flags"].sum(dim=1).cpu().numpy() if ("noise" in methods_to_run and "mock_flags" in results["noise"]) else None
    num_guess_steps_momentum = results["momentum"]["mock_flags"].sum(dim=1).cpu().numpy() if ("momentum" in methods_to_run and "mock_flags" in results["momentum"]) else None
    num_guess_steps_reuse = results["reuse"]["mock_flags"].sum(dim=1).cpu().numpy() if ("reuse" in methods_to_run and "mock_flags" in results["reuse"]) else None
    num_guess_steps_auto = results["auto"]["mock_flags"].sum(dim=1).cpu().numpy() if ("auto" in methods_to_run and "mock_flags" in results["auto"]) else None
    
    # 9. 计算和比较HPSv2 rewards
    if rank == 0:
        print(f"\n{'='*80}")
        print("Computing HPSv2 Reward Statistics...")
        print(f"{'='*80}\n")
        
        # 转换rewards到CPU numpy
        rewards_cpu = {}
        for method in methods_to_run:
            if len(results[method]["rewards"]) > 0:
                rewards_cpu[method] = results[method]["rewards"].cpu().numpy()
        
        # 按prompt分组统计
        rewards_by_prompt = {method: [] for method in methods_to_run}
        for p in range(num_prompts):
            start_idx = p * args.num_generations
            end_idx = (p + 1) * args.num_generations
            for method in methods_to_run:
                if method in rewards_cpu:
                    rewards_by_prompt[method].append(rewards_cpu[method][start_idx:end_idx])

        prompt_mean_stats = {
            method: np.array([vals.mean() for vals in rewards_by_prompt[method]])
            for method in methods_to_run
            if method in rewards_cpu
        }
        prompt_max_stats = {
            method: np.array([vals.max() for vals in rewards_by_prompt[method]])
            for method in methods_to_run
            if method in rewards_cpu
        }
        
        # 10. 生成详细对比报告
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Total prompts: {num_prompts}")
        print(f"Generations per prompt: {args.num_generations}")
        print(f"Total samples: {num_prompts * args.num_generations}")
        print(f"Random seed: 42 (fixed for each rollout)")
        
        method_names = {
            "original": "Original (No Guess - Baseline 20 steps)",
            "original_14": "Original (14 steps - Same Compute)",
            "mean_direction": "Mean Direction",
            "noise": "Random Noise",
            "momentum": "Momentum (Mean Direction + Momentum 0.5)",
            "reuse": "Reuse Velocity (Direct Reuse)",
            "auto": "Auto Momentum (Adaptive Weight)",
            "d2": "MTA D2 (2nd-order trajectory + adaptive momentum + accel coupling)",
            "alpha": "MTA Alpha (cosine alpha * v_prev + (1-alpha) * v_group)",
            "ab2": "MTA AB2 (1.5*v_prev - 0.5*v_prevprev fused with v_group)",
            "ab2_2": "MTA AB2-2 (early_frac=0.5, v_hat=a^p*v_so+(1-a)*v_group, p=0.99)",
            "ab2_abl_ef04": "AB2 ablate: early_frac=0.4, v_so=1.5*v_prev-0.5*v_pp",
            "ab2_abl_ef06": "AB2 ablate: early_frac=0.6, v_so=1.5*v_prev-0.5*v_pp",
            "ab2_abl_v1": "AB2 ablate: early_frac=0.5, v_so=v_prev (first-order)",
            "ab2_abl_strong": "AB2 ablate: early_frac=0.5, v_so=2*v_prev-1*v_pp",
            "ab2_abl_ef04_strong": "AB2 combo: early_frac=0.4, v_so=2*v_prev-1*v_pp",
            "varguess": "MTA VarGuess (adaptive guess count from group var; auto velocity)",
            "resab": "MTA ResAB (mean-field v_tgt + cosine-weighted AB2 accel)",
            "line": "MTA Line (v_tgt + cosine-weighted velocity difference)",
            "heun": "MTA Heun (early: average of AB2 and v_tgt)",
            "rab2": "MTA RAB2 (AB2 on rollout residual around current group mean)",
            "rab2_mid": "MTA RAB2-Mid (same residual AB2, bias=0.20, growth=1.25)",
            "rab2_tight": "MTA RAB2-Tight (same residual AB2, bias=0.15, growth=1.20)",
            "rab2_g130": "MTA RAB2-G130 (same residual AB2, bias=0.25, growth=1.30)",
            "rab2_b020": "MTA RAB2-B020 (same residual AB2, bias=0.20, growth=1.35)",
            "rab2_b030": "MTA RAB2-B030 (same residual AB2, bias=0.30, growth=1.35)",
            "rab2_b035": "MTA RAB2-B035 (same residual AB2, bias=0.35, growth=1.35)",
            "rab2_b040": "MTA RAB2-B040 (same residual AB2, bias=0.40, growth=1.35)",
            "rab2_b030_g140": "MTA RAB2-B030-G140 (same residual AB2, bias=0.30, growth=1.40)",
            "ab2_trust": "MTA AB2-Trust (strong AB2 + EMA group anchor + trust region)",
            "oneach_14": "Oneach (14 steps - Full Inference + Mean Correction)"
        }
        
        # 计算 guess 统计 (从结果中提取)
        num_guess_steps = {}
        for method in methods_to_run:
            if "mock_flags" in results[method] and results[method]["mock_flags"] is not None:
                num_guess_steps[method] = results[method]["mock_flags"].sum(dim=1).cpu().numpy()
        
        # 打印每个方法的统计信息
        for idx, method in enumerate(methods_to_run, 1):
            print(f"\n{'='*80}")
            print(f"Method {idx} ({method_names[method]}):")
            print(f"{'='*80}")
            
            if method == "original" or method == "original_14" or method == "oneach_14":
                print(f"  - Guess steps: 0 (no guess)")
            elif method in num_guess_steps:
                print(f"  - Average guess steps: {num_guess_steps[method].mean():.2f} ± {num_guess_steps[method].std():.2f}")
            
            if method in rewards_cpu:
                print(f"  - HPSv2 Reward (overall):")
                print(f"    * Mean: {rewards_cpu[method].mean():.4f}")
                print(f"    * Std:  {rewards_cpu[method].std():.4f}")
                print(f"    * Min:  {rewards_cpu[method].min():.4f}")
                print(f"    * Max:  {rewards_cpu[method].max():.4f}")
                print(f"  - Results saved to: {os.path.join(args.output_path, method)}")
        
        # 比较不同方法
        if len(methods_to_run) > 1:
            print(f"\n{'='*80}")
            print("Reward Comparison:")
            print(f"{'='*80}")
            
            # 如果有baseline（original），与其他方法比较
            if "original" in methods_to_run and "original" in rewards_cpu:
                baseline_mean = rewards_cpu["original"].mean()
                baseline_prompt_mean = prompt_mean_stats["original"]
                baseline_prompt_max = prompt_max_stats["original"]
                for method in methods_to_run:
                    if method != "original" and method in rewards_cpu:
                        diff = rewards_cpu[method].mean() - baseline_mean
                        improvement = (diff / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                        prompt_mean_drawdown = (baseline_prompt_mean - prompt_mean_stats[method]).mean()
                        prompt_max_worse_frac = (prompt_max_stats[method] < baseline_prompt_max).mean()
                        print(f"  - {method_names[method]} vs Baseline:")
                        print(f"    * Difference: {diff:+.4f}")
                        print(f"    * Improvement: {improvement:+.2f}%")
                        print(f"    * Prompt-mean drawdown: {prompt_mean_drawdown:+.4f}")
                        print(f"    * Prompt-max below baseline frac: {prompt_max_worse_frac:.2%}")
            
            # 两两比较其他方法
            print(f"\n  Pairwise Comparisons:")
            compared_pairs = set()
            for i, method1 in enumerate(methods_to_run):
                for method2 in methods_to_run[i+1:]:
                    if method1 in rewards_cpu and method2 in rewards_cpu:
                        pair = tuple(sorted([method1, method2]))
                        if pair not in compared_pairs:
                            compared_pairs.add(pair)
                            diff = rewards_cpu[method1].mean() - rewards_cpu[method2].mean()
                            print(f"  - {method_names[method1]} vs {method_names[method2]}:")
                            print(f"    * Difference: {diff:+.4f}")
                            if diff > 0:
                                print(f"    * Winner: {method_names[method1]} (better by {abs(diff):.4f})")
                            elif diff < 0:
                                print(f"    * Winner: {method_names[method2]} (better by {abs(diff):.4f})")
                            else:
                                print(f"    * Result: Tie")
        
        print(f"\n{'='*80}")
        print("Per-Prompt Reward Comparison:")
        print(f"{'='*80}")
        for p in range(num_prompts):
            print(f"  Prompt {p} ('{all_captions[p][:50]}...'):")
            for method in methods_to_run:
                if method in rewards_by_prompt and len(rewards_by_prompt[method]) > p:
                    mean_reward = rewards_by_prompt[method][p].mean()
                    if "original" in rewards_by_prompt and len(rewards_by_prompt["original"]) > p and method != "original":
                        diff = mean_reward - rewards_by_prompt["original"][p].mean()
                        print(f"    - {method_names[method]:20s}: {mean_reward:.4f} ({diff:+.4f})")
                    else:
                        print(f"    - {method_names[method]:20s}: {mean_reward:.4f}")
        
        # 保存统计结果到文件
        stats_path = os.path.join(args.output_path, "comparison_stats.txt")
        with open(stats_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPARISON STATISTICS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total prompts: {num_prompts}\n")
            f.write(f"Generations per prompt: {args.num_generations}\n")
            f.write(f"Total samples: {num_prompts * args.num_generations}\n")
            f.write(f"Random seed: 42 (fixed for each rollout)\n\n")
            
            for method in methods_to_run:
                f.write(f"{method_names[method]}:\n")
                if method in rewards_cpu:
                    f.write(f"  HPSv2 Reward Mean: {rewards_cpu[method].mean():.4f}\n")
                    f.write(f"  HPSv2 Reward Std:  {rewards_cpu[method].std():.4f}\n")
                    f.write(f"  Prompt Mean Avg:  {prompt_mean_stats[method].mean():.4f}\n")
                    f.write(f"  Prompt Max Avg:   {prompt_max_stats[method].mean():.4f}\n")
                    if "original" in rewards_cpu and method != "original":
                        diff = rewards_cpu[method].mean() - rewards_cpu["original"].mean()
                        improvement = (diff / abs(rewards_cpu["original"].mean())) * 100 if rewards_cpu["original"].mean() != 0 else 0
                        prompt_mean_drawdown = (prompt_mean_stats["original"] - prompt_mean_stats[method]).mean()
                        prompt_max_worse_frac = (prompt_max_stats[method] < prompt_max_stats["original"]).mean()
                        f.write(f"  vs Baseline: {diff:+.4f} ({improvement:+.2f}%)\n")
                        f.write(f"  Prompt-mean drawdown: {prompt_mean_drawdown:+.4f}\n")
                        f.write(f"  Prompt-max below baseline frac: {prompt_max_worse_frac:.2%}\n")
                f.write("\n")
        
        print(f"\n{'='*80}")
        print(f"Statistics saved to: {stats_path}")
        print(f"You can now visually compare the results in the following directories:")
        for method in methods_to_run:
            print(f"  - {os.path.join(args.output_path, method)}")
        print(f"{'='*80}\n")
    
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    test_comparison()
