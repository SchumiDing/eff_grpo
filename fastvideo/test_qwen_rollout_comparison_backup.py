"""
对比测试脚本: 同时运行六种方法并对比结果
1. 原始方法: 不使用guess,每步都完整推理 (baseline)
2. 平均方向方法: 使用平均方向计算guess
3. 噪声方法: 直接添加随机噪声guess
4. 动量方法: 使用平均方向+动量计算guess
5. 直接复用方法: 直接复用上一轮速度
6. 自适应动量方法: 随时间步自适应调整动量权重
"""

import torch
import os
import shutil
import argparse
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor
import json
import random
import numpy as np

# 导入六种方法
from train_grpo_qwenimage import sample_reference_model as sample_reference_model_original
from train_grpo_qwenimage_eff import sample_reference_model as sample_reference_model_mean, sd3_time_shift
from train_grpo_qwenimage_noise import sample_reference_model_noise
from train_grpo_qwenimage_eff_mta import sample_reference_model as sample_reference_model_momentum
from train_grpo_qwenimage_mta import sample_reference_model as sample_reference_model_reuse
from train_grpo_qwenimage_eff_mta_auto import sample_reference_model as sample_reference_model_auto
from train_grpo_qwenimage_eff_oneach import run_sample_step as run_sample_step_oneach, flux_step, unpack_latents

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
    parser.add_argument("--embeddings_path", type=str, default="data/qwenimage/rl_embeddings", help="Path to rl embeddings")
    parser.add_argument("--num_generations", type=int, default=12, help="Number of samples per prompt")
    parser.add_argument(
        "--num_guess",
        type=int,
        default=6,
        help="Per prompt: max rollouts to mock-guess each step (within each prompt's num_generations group)",
    )
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--output_path", type=str, default="./test_results_comparison")
    parser.add_argument("--init_same_noise", action="store_true", default=False, help="Use same noise for all samples in a group")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of embeddings to load")
    parser.add_argument("--end_idx", type=int, default=10, help="End index of embeddings to load (None means all)")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of prompts per batch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--methods", type=str, nargs='+', default=["original", "original_14", "momentum", "reuse", "auto"], 
                        help="Methods to run: original, mean_direction, noise, momentum, reuse, auto")
    args = parser.parse_args()
    eq_sampling_steps = int(args.sampling_steps * (args.num_generations-args.num_guess)/args.num_generations + 0.5 )
    print(f"Equivalent sampling steps: {eq_sampling_steps}")
    # 设置随机种子
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    device = torch.device("cuda")
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.output_path == "./test_results_comparison":
        args.output_path = f"./test_results_comparison_{args.embeddings_path.split('/')[-1]}"
    
    # 检查哪些方法已经生成过，跳过已有的方法
    available_methods = ["original", "original_14", "mean_direction", "noise", "momentum", "reuse", "auto", "oneach_14"]
    methods_to_run = []
    for method in args.methods:
        if method not in available_methods:
            print(f"Warning: Unknown method '{method}', skipping...")
            continue
        method_dir = os.path.join(args.output_path, method)
        if os.path.exists(method_dir) and os.listdir(method_dir):
            print(f"Method '{method}' already has results, skipping...")
        else:
            methods_to_run.append(method)
            os.makedirs(method_dir, exist_ok=True)
    
    if not methods_to_run:
        print("All requested methods already have results. Exiting...")
        return
    
    print(f"Methods to run: {methods_to_run}")

    # 1. 加载组件
    print("Loading models...")
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from diffusers.models.autoencoders import AutoencoderKLQwenImage
    
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model_path, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    ).to(device)
    
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
    print(f"Processing {len(data_anno)} prompts (index {args.start_idx} to {end_idx-1})")
    
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

    rank = 0

    def move_original_batch_to_output(start_prompt_idx: int, batch_num_prompts: int) -> None:
        """本 batch 采样完成后立刻写入 output_path/original，再进入下一 batch / 下一 method。"""
        g = args.num_generations
        n = batch_num_prompts * g
        for local_idx in range(n):
            prompt_idx = start_prompt_idx + local_idx // g
            gen_idx = local_idx % g
            src = f"./images/qwenimage_{rank}_{local_idx}.png"
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
            src = f"./images/qwenimage_{rank}_{local_idx}_guess{guess_count}.png"
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
            src = f"./images/qwenimage_{rank}_{local_idx}_guess{guess_count}_noise.png"
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
            src = f"./images/qwenimage_{rank}_{local_idx}_guess{guess_count}.png"
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
            src = f"./images/qwenimage_{rank}_{local_idx}_guess{guess_count}.png"
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
            src = f"./images/qwenimage_{rank}_{local_idx}_guess{guess_count}.png"
            prompt_dir = os.path.join(args.output_path, "auto", f"prompt_{prompt_idx}")
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
            src = f"./images/qwenimage_{rank}_{local_idx}.png"
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
            src = f"./images/qwenimage_{rank}_{local_idx}_oneach14.png"
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
        "oneach_14": {"rewards": [], "latents": [], "log_probs": [], "txt_seq_lens": []},
    }
    
    # 获取 eq_sampling_steps 所需的参数
    # 在 run_sample_step_oneach 中需要这些参数
    mock_args.sampling_steps = eq_sampling_steps 
    
    # 计算batch数量
    num_batches = (num_prompts + args.batch_size - 1) // args.batch_size
    
    # 3. 按batch运行各种方法
    if "original" in methods_to_run:
        print(f"\n{'='*80}")
        print("METHOD 1: Original (No Guess - Baseline)")
        print(f"{'='*80}\n")
        
        for batch_idx in range(num_batches):
            # 动量方法等使用原始 sampling_steps (20步)
            mock_args.sampling_steps = args.sampling_steps
            # 每个batch前重置seed=42以确保可复现性
            set_seed(42)
            
            start_prompt_idx = batch_idx * args.batch_size
            end_prompt_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
            batch_num_prompts = end_prompt_idx - start_prompt_idx
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{end_prompt_idx-1})...")
            
            # 准备当前batch的数据
            encoder_hidden_states_list = []
            prompt_attention_masks_list = []
            original_length_list = []
            captions_expanded = []
            
            for i in range(start_prompt_idx, end_prompt_idx):
                for _ in range(args.num_generations):
                    encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                    prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                    original_length_list.append(all_original_lengths[i])
                    captions_expanded.append(all_captions[i])
            
            encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
            prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
            original_length = torch.tensor(original_length_list).to(device)
            
            # 运行采样 - 原始方法
            # 原始方法使用 20 步
            mock_args.sampling_steps = args.sampling_steps
            batch_rewards, batch_latents, batch_log_probs, sigma_schedule, batch_txt_seq_lens = sample_reference_model_original(
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
            
            move_original_batch_to_output(start_prompt_idx, batch_num_prompts)
            
            results["original"]["rewards"].append(batch_rewards)
            results["original"]["latents"].append(batch_latents)
            results["original"]["log_probs"].append(batch_log_probs)
            results["original"]["txt_seq_lens"].append(batch_txt_seq_lens)
            
            print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
        
        # 合并所有batch的结果
        results["original"]["rewards"] = torch.cat(results["original"]["rewards"], dim=0)
        results["original"]["latents"] = torch.cat(results["original"]["latents"], dim=0)
        results["original"]["log_probs"] = torch.cat(results["original"]["log_probs"], dim=0)
        results["original"]["txt_seq_lens"] = torch.cat(results["original"]["txt_seq_lens"], dim=0)
        
        print("Method 1 all batches completed!")
        print(f"  Images -> {os.path.join(args.output_path, 'original')}\n")
    
    # 3.5. 按batch运行方法: Original (Steps=14)
    if "original_14" in methods_to_run:
        print(f"\n{'='*80}")
        print("METHOD 1.5: Original (14 Steps - Same Compute Baseline)")
        print(f"{'='*80}\n")
        
        # 创建 14 步的 mock args
        mock_args_14 = MockArgs(
            num_guess=args.num_guess,
            num_generations=args.num_generations,
            eta=args.eta,
            sampling_steps=eq_sampling_steps, # 设置为 14 步
            shift=args.shift,
            init_same_noise=args.init_same_noise,
            w=args.width,
            h=args.height,
            t=1,
            use_hpsv2=True,
            use_hpsv3=False,
            use_pickscore=False,
        )
        
        for batch_idx in range(num_batches):
            set_seed(42)
            
            start_prompt_idx = batch_idx * args.batch_size
            end_prompt_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
            batch_num_prompts = end_prompt_idx - start_prompt_idx
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{end_prompt_idx-1})...")
            
            encoder_hidden_states_list = []
            prompt_attention_masks_list = []
            original_length_list = []
            captions_expanded = []
            
            for i in range(start_prompt_idx, end_prompt_idx):
                for _ in range(args.num_generations):
                    encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                    prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                    original_length_list.append(all_original_lengths[i])
                    captions_expanded.append(all_captions[i])
            
            encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
            prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
            original_length = torch.tensor(original_length_list).to(device)
            
            batch_rewards, batch_latents, batch_log_probs, sigma_schedule, batch_txt_seq_lens = sample_reference_model_original(
                mock_args_14, # 使用 14 步的参数
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
            
            move_original_14_batch_to_output(start_prompt_idx, batch_num_prompts)
            
            results["original_14"]["rewards"].append(batch_rewards)
            results["original_14"]["latents"].append(batch_latents)
            results["original_14"]["log_probs"].append(batch_log_probs)
            results["original_14"]["txt_seq_lens"].append(batch_txt_seq_lens)
            
            print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
        
        results["original_14"]["rewards"] = torch.cat(results["original_14"]["rewards"], dim=0)
        results["original_14"]["latents"] = torch.cat(results["original_14"]["latents"], dim=0)
        results["original_14"]["log_probs"] = torch.cat(results["original_14"]["log_probs"], dim=0)
        results["original_14"]["txt_seq_lens"] = torch.cat(results["original_14"]["txt_seq_lens"], dim=0)
        
        print("Method Original_14 all batches completed!")
        print(f"  Images -> {os.path.join(args.output_path, 'original_14')}\n")
    
    # 4. 按batch运行方法2: 平均方向
    if "mean_direction" in methods_to_run:
        print(f"\n{'='*80}")
        print("METHOD 2: Mean Direction")
        print(f"{'='*80}\n")
        
        for batch_idx in range(num_batches):
            # 动量方法等使用原始 sampling_steps (20步)
            mock_args.sampling_steps = args.sampling_steps
            # 每个batch前重置seed=42以确保可复现性
            set_seed(42)
        
        start_prompt_idx = batch_idx * args.batch_size
        end_prompt_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
        batch_num_prompts = end_prompt_idx - start_prompt_idx
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{end_prompt_idx-1})...")
        
        # 准备当前batch的数据
        encoder_hidden_states_list = []
        prompt_attention_masks_list = []
        original_length_list = []
        captions_expanded = []
        
        for i in range(start_prompt_idx, end_prompt_idx):
            for _ in range(args.num_generations):
                encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                original_length_list.append(all_original_lengths[i])
                captions_expanded.append(all_captions[i])
        
        encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
        prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
        original_length = torch.tensor(original_length_list).to(device)
        
        # 运行采样 - 平均方向方法
        batch_rewards_mean, batch_latents_mean, batch_log_probs_mean, sigma_schedule_mean, batch_txt_seq_lens_mean, batch_mock_flags_mean = sample_reference_model_mean(
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
        
        batch_mock_flags_cpu = batch_mock_flags_mean.sum(dim=1).cpu().numpy()
        move_mean_batch_to_output(start_prompt_idx, batch_num_prompts, batch_mock_flags_cpu)
        
        results["mean_direction"]["rewards"].append(batch_rewards_mean)
        results["mean_direction"]["latents"].append(batch_latents_mean)
        results["mean_direction"]["log_probs"].append(batch_log_probs_mean)
        results["mean_direction"]["mock_flags"].append(batch_mock_flags_mean)
        results["mean_direction"]["txt_seq_lens"].append(batch_txt_seq_lens_mean)
        
        print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
        
        # 合并所有batch的结果
        results["mean_direction"]["rewards"] = torch.cat(results["mean_direction"]["rewards"], dim=0)
        results["mean_direction"]["latents"] = torch.cat(results["mean_direction"]["latents"], dim=0)
        results["mean_direction"]["log_probs"] = torch.cat(results["mean_direction"]["log_probs"], dim=0)
        results["mean_direction"]["mock_flags"] = torch.cat(results["mean_direction"]["mock_flags"], dim=0)
        results["mean_direction"]["txt_seq_lens"] = torch.cat(results["mean_direction"]["txt_seq_lens"], dim=0)
        
        print("Method 2 all batches completed!")
        print(f"  Images -> {os.path.join(args.output_path, 'mean_direction')}\n")
    
    # 5. 按batch运行方法3: 随机噪声
    if "noise" in methods_to_run:
        print(f"\n{'='*80}")
        print("METHOD 3: Random Noise")
        print(f"{'='*80}\n")
        
        for batch_idx in range(num_batches):
            # 动量方法等使用原始 sampling_steps (20步)
            mock_args.sampling_steps = args.sampling_steps
            # 每个batch前重置seed=42以确保可复现性
            set_seed(42)
        
        start_prompt_idx = batch_idx * args.batch_size
        end_prompt_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
        batch_num_prompts = end_prompt_idx - start_prompt_idx
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{end_prompt_idx-1})...")
        
        # 准备当前batch的数据
        encoder_hidden_states_list = []
        prompt_attention_masks_list = []
        original_length_list = []
        captions_expanded = []
        
        for i in range(start_prompt_idx, end_prompt_idx):
            for _ in range(args.num_generations):
                encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                original_length_list.append(all_original_lengths[i])
                captions_expanded.append(all_captions[i])
        
        encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
        prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
        original_length = torch.tensor(original_length_list).to(device)
        
        # 运行采样 - 随机噪声方法
        batch_rewards_noise, batch_latents_noise, batch_log_probs_noise, sigma_schedule_noise, batch_txt_seq_lens_noise, batch_mock_flags_noise = sample_reference_model_noise(
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
        
        batch_mock_flags_cpu = batch_mock_flags_noise.sum(dim=1).cpu().numpy()
        move_noise_batch_to_output(start_prompt_idx, batch_num_prompts, batch_mock_flags_cpu)
        
        results["noise"]["rewards"].append(batch_rewards_noise)
        results["noise"]["latents"].append(batch_latents_noise)
        results["noise"]["log_probs"].append(batch_log_probs_noise)
        results["noise"]["mock_flags"].append(batch_mock_flags_noise)
        results["noise"]["txt_seq_lens"].append(batch_txt_seq_lens_noise)
        
        print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
    
        # 合并所有batch的结果
        results["noise"]["rewards"] = torch.cat(results["noise"]["rewards"], dim=0)
        results["noise"]["latents"] = torch.cat(results["noise"]["latents"], dim=0)
        results["noise"]["log_probs"] = torch.cat(results["noise"]["log_probs"], dim=0)
        results["noise"]["mock_flags"] = torch.cat(results["noise"]["mock_flags"], dim=0)
        results["noise"]["txt_seq_lens"] = torch.cat(results["noise"]["txt_seq_lens"], dim=0)
        
        print("Method 3 all batches completed!")
        print(f"  Images -> {os.path.join(args.output_path, 'noise')}\n")
    
    # 6. 按batch运行方法4: 动量方法
    if "momentum" in methods_to_run:
        print(f"\n{'='*80}")
        print("METHOD 4: Momentum (Mean Direction + Momentum)")
        print(f"{'='*80}\n")
        
        for batch_idx in range(num_batches):
            # 动量方法等使用原始 sampling_steps (20步)
            mock_args.sampling_steps = args.sampling_steps
            # 每个batch前重置seed=42以确保可复现性
            set_seed(42)
        
        start_prompt_idx = batch_idx * args.batch_size
        end_prompt_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
        batch_num_prompts = end_prompt_idx - start_prompt_idx
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{end_prompt_idx-1})...")
        
        # 准备当前batch的数据
        encoder_hidden_states_list = []
        prompt_attention_masks_list = []
        original_length_list = []
        captions_expanded = []
        
        for i in range(start_prompt_idx, end_prompt_idx):
            for _ in range(args.num_generations):
                encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                original_length_list.append(all_original_lengths[i])
                captions_expanded.append(all_captions[i])
        
        encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
        prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
        original_length = torch.tensor(original_length_list).to(device)
        
        # 运行采样 - 动量方法
        batch_rewards_momentum, batch_latents_momentum, batch_log_probs_momentum, sigma_schedule_momentum, batch_txt_seq_lens_momentum, batch_mock_flags_momentum = sample_reference_model_momentum(
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
        
        batch_mock_flags_cpu = batch_mock_flags_momentum.sum(dim=1).cpu().numpy()
        move_momentum_batch_to_output(start_prompt_idx, batch_num_prompts, batch_mock_flags_cpu)
        
        results["momentum"]["rewards"].append(batch_rewards_momentum)
        results["momentum"]["latents"].append(batch_latents_momentum)
        results["momentum"]["log_probs"].append(batch_log_probs_momentum)
        results["momentum"]["mock_flags"].append(batch_mock_flags_momentum)
        results["momentum"]["txt_seq_lens"].append(batch_txt_seq_lens_momentum)
        
        print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
        
        # 合并所有batch的结果
        results["momentum"]["rewards"] = torch.cat(results["momentum"]["rewards"], dim=0)
        results["momentum"]["latents"] = torch.cat(results["momentum"]["latents"], dim=0)
        results["momentum"]["log_probs"] = torch.cat(results["momentum"]["log_probs"], dim=0)
        results["momentum"]["mock_flags"] = torch.cat(results["momentum"]["mock_flags"], dim=0)
        results["momentum"]["txt_seq_lens"] = torch.cat(results["momentum"]["txt_seq_lens"], dim=0)
        
        print("Method 4 all batches completed!")
        print(f"  Images -> {os.path.join(args.output_path, 'momentum')}\n")
    
    # 7. 按batch运行方法5: 直接复用速度
    if "reuse" in methods_to_run:
        print(f"\n{'='*80}")
        print("METHOD 5: Reuse Velocity (Direct Reuse of Previous Velocity)")
        print(f"{'='*80}\n")
        
        for batch_idx in range(num_batches):
            # 动量方法等使用原始 sampling_steps (20步)
            mock_args.sampling_steps = args.sampling_steps
            # 每个batch前重置seed=42以确保可复现性
            set_seed(42)
            
            start_prompt_idx = batch_idx * args.batch_size
            end_prompt_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
            batch_num_prompts = end_prompt_idx - start_prompt_idx
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{end_prompt_idx-1})...")
            
            # 准备当前batch的数据
            encoder_hidden_states_list = []
            prompt_attention_masks_list = []
            original_length_list = []
            captions_expanded = []
            
            for i in range(start_prompt_idx, end_prompt_idx):
                for _ in range(args.num_generations):
                    encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                    prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                    original_length_list.append(all_original_lengths[i])
                    captions_expanded.append(all_captions[i])
            
            encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
            prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
            original_length = torch.tensor(original_length_list).to(device)
            
            # 运行采样 - 直接复用速度方法
            batch_rewards_reuse, batch_latents_reuse, batch_log_probs_reuse, sigma_schedule_reuse, batch_txt_seq_lens_reuse, batch_mock_flags_reuse = sample_reference_model_reuse(
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
            
            batch_mock_flags_cpu = batch_mock_flags_reuse.sum(dim=1).cpu().numpy()
            move_reuse_batch_to_output(start_prompt_idx, batch_num_prompts, batch_mock_flags_cpu)
            
            results["reuse"]["rewards"].append(batch_rewards_reuse)
            results["reuse"]["latents"].append(batch_latents_reuse)
            results["reuse"]["log_probs"].append(batch_log_probs_reuse)
            results["reuse"]["mock_flags"].append(batch_mock_flags_reuse)
            results["reuse"]["txt_seq_lens"].append(batch_txt_seq_lens_reuse)
            
            print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
        
        # 合并所有batch的结果
        results["reuse"]["rewards"] = torch.cat(results["reuse"]["rewards"], dim=0)
        results["reuse"]["latents"] = torch.cat(results["reuse"]["latents"], dim=0)
        results["reuse"]["log_probs"] = torch.cat(results["reuse"]["log_probs"], dim=0)
        results["reuse"]["mock_flags"] = torch.cat(results["reuse"]["mock_flags"], dim=0)
        results["reuse"]["txt_seq_lens"] = torch.cat(results["reuse"]["txt_seq_lens"], dim=0)
        
        print("Method 5 all batches completed!")
        print(f"  Images -> {os.path.join(args.output_path, 'reuse')}\n")
    
    # 8. 按batch运行方法6: 自适应动量方法
    if "auto" in methods_to_run:
        print(f"\n{'='*80}")
        print("METHOD 6: Auto Momentum (Adaptive Momentum Weight)")
        print(f"{'='*80}\n")
        
        for batch_idx in range(num_batches):
            # 动量方法等使用原始 sampling_steps (20步)
            mock_args.sampling_steps = args.sampling_steps
            # 每个batch前重置seed=42以确保可复现性
            set_seed(42)
            
            start_prompt_idx = batch_idx * args.batch_size
            end_prompt_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
            batch_num_prompts = end_prompt_idx - start_prompt_idx
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{end_prompt_idx-1})...")
            
            # 准备当前batch的数据
            encoder_hidden_states_list = []
            prompt_attention_masks_list = []
            original_length_list = []
            captions_expanded = []
            
            for i in range(start_prompt_idx, end_prompt_idx):
                for _ in range(args.num_generations):
                    encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                    prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                    original_length_list.append(all_original_lengths[i])
                    captions_expanded.append(all_captions[i])
            
            encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
            prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
            original_length = torch.tensor(original_length_list).to(device)
            
            # 运行采样 - 自适应动量方法
            batch_rewards_auto, batch_latents_auto, batch_log_probs_auto, sigma_schedule_auto, batch_txt_seq_lens_auto, batch_mock_flags_auto = sample_reference_model_auto(
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
            
            batch_mock_flags_cpu = batch_mock_flags_auto.sum(dim=1).cpu().numpy()
            move_auto_batch_to_output(start_prompt_idx, batch_num_prompts, batch_mock_flags_cpu)
            
            results["auto"]["rewards"].append(batch_rewards_auto)
            results["auto"]["latents"].append(batch_latents_auto)
            results["auto"]["log_probs"].append(batch_log_probs_auto)
            results["auto"]["mock_flags"].append(batch_mock_flags_auto)
            results["auto"]["txt_seq_lens"].append(batch_txt_seq_lens_auto)
            
            print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
        
        # 合并所有batch的结果
        results["auto"]["rewards"] = torch.cat(results["auto"]["rewards"], dim=0)
        results["auto"]["latents"] = torch.cat(results["auto"]["latents"], dim=0)
        results["auto"]["log_probs"] = torch.cat(results["auto"]["log_probs"], dim=0)
        results["auto"]["mock_flags"] = torch.cat(results["auto"]["mock_flags"], dim=0)
        results["auto"]["txt_seq_lens"] = torch.cat(results["auto"]["txt_seq_lens"], dim=0)
        
        print("Method 6 all batches completed!")
        print(f"  Images -> {os.path.join(args.output_path, 'auto')}\n")

    # 8.5 按batch运行方法: Oneach (Steps=14, 全量推理 + 均值校正)
    if "oneach_14" in methods_to_run:
        print(f"\n{'='*80}")
        print("METHOD 6.5: Oneach (14 Steps - Full Inference + Mean Correction)")
        print(f"{'='*80}\n")
        
        for batch_idx in range(num_batches):
            set_seed(42)
            # 使用 eq_sampling_steps (14步)
            mock_args.sampling_steps = eq_sampling_steps
            
            start_prompt_idx = batch_idx * args.batch_size
            end_prompt_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
            batch_num_prompts = end_prompt_idx - start_prompt_idx
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_prompt_idx}-{end_prompt_idx-1})...")
            
            # 准备当前batch的数据
            encoder_hidden_states_list = []
            prompt_attention_masks_list = []
            original_length_list = []
            captions_expanded = []
            
            for i in range(start_prompt_idx, end_prompt_idx):
                for _ in range(args.num_generations):
                    encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
                    prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
                    original_length_list.append(all_original_lengths[i])
                    captions_expanded.append(all_captions[i])
            
            encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
            prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
            original_length = torch.tensor(original_length_list).to(device)
            B = encoder_hidden_states.shape[0]

            # 准备初始噪声
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
                    tqdm(range(eq_sampling_steps), desc="Sampling"),
                    sigma_schedule_14,
                    transformer,
                    encoder_hidden_states,
                    prompt_attention_masks,
                    img_shapes,
                    txt_seq_lens,
                    grpo_sample=True,
                )
            
            # 解码并保存图像
            vae.enable_tiling()
            from diffusers.image_processor import VaeImageProcessor
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
                        decoded_images[idx].save(f"./images/qwenimage_{rank}_{idx}_oneach14.png")

            # 计算 rewards
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
            
            results["oneach_14"]["rewards"].append(batch_rewards)
            results["oneach_14"]["latents"].append(all_latents_tensor)
            results["oneach_14"]["log_probs"].append(all_log_probs_tensor)
            results["oneach_14"]["txt_seq_lens"].append(torch.tensor(txt_seq_lens))
            
            print(f"Batch {batch_idx + 1}/{num_batches} completed!\n")
        
        results["oneach_14"]["rewards"] = torch.cat(results["oneach_14"]["rewards"], dim=0)
        results["oneach_14"]["latents"] = torch.cat(results["oneach_14"]["latents"], dim=0)
        results["oneach_14"]["log_probs"] = torch.cat(results["oneach_14"]["log_probs"], dim=0)
        results["oneach_14"]["txt_seq_lens"] = torch.cat(results["oneach_14"]["txt_seq_lens"], dim=0)
        
        print("Method 6.5 all batches completed!")
        print(f"  Images -> {os.path.join(args.output_path, 'oneach_14')}\n")

    # 计算guess步数统计
    num_guess_steps_mean = results["mean_direction"]["mock_flags"].sum(dim=1).cpu().numpy() if "mean_direction" in methods_to_run else None
    num_guess_steps_noise = results["noise"]["mock_flags"].sum(dim=1).cpu().numpy() if "noise" in methods_to_run else None
    num_guess_steps_momentum = results["momentum"]["mock_flags"].sum(dim=1).cpu().numpy() if "momentum" in methods_to_run else None
    num_guess_steps_reuse = results["reuse"]["mock_flags"].sum(dim=1).cpu().numpy() if "reuse" in methods_to_run else None
    num_guess_steps_auto = results["auto"]["mock_flags"].sum(dim=1).cpu().numpy() if "auto" in methods_to_run else None
    
    # 9. 计算和比较HPSv2 rewards
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
    
    # 10. 生成详细对比报告
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Total prompts: {num_prompts}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Total samples: {total_samples}")
    print(f"Random seed: 42 (fixed for each rollout)")
    
    method_names = {
        "original": "Original (No Guess - Baseline 20 steps)",
        "original_14": "Original (14 steps - Same Compute)",
        "mean_direction": "Mean Direction",
        "noise": "Random Noise",
        "momentum": "Momentum (Mean Direction + Momentum 0.5)",
        "reuse": "Reuse Velocity (Direct Reuse)",
        "auto": "Auto Momentum (Adaptive Weight)",
        "oneach_14": "Oneach (14 steps - Full Inference + Mean Correction)"
    }
    
    # 打印每个方法的统计信息
    for idx, method in enumerate(methods_to_run, 1):
        print(f"\n{'='*80}")
        print(f"Method {idx} ({method_names[method]}):")
        print(f"{'='*80}")
        
        if method == "original" or method == "original_14":
            print(f"  - Guess steps: 0 (no guess)")
        elif method == "mean_direction":
            print(f"  - Average guess steps: {num_guess_steps_mean.mean():.2f} ± {num_guess_steps_mean.std():.2f}")
        elif method == "noise":
            print(f"  - Average guess steps: {num_guess_steps_noise.mean():.2f} ± {num_guess_steps_noise.std():.2f}")
        elif method == "momentum":
            print(f"  - Average guess steps: {num_guess_steps_momentum.mean():.2f} ± {num_guess_steps_momentum.std():.2f}")
        elif method == "reuse":
            print(f"  - Average guess steps: {num_guess_steps_reuse.mean():.2f} ± {num_guess_steps_reuse.std():.2f}")
        elif method == "auto":
            print(f"  - Average guess steps: {num_guess_steps_auto.mean():.2f} ± {num_guess_steps_auto.std():.2f}")
        elif method == "oneach_14":
            print(f"  - Guess steps: 0 (no guess, full inference)")
        
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
            for method in methods_to_run:
                if method != "original" and method in rewards_cpu:
                    diff = rewards_cpu[method].mean() - baseline_mean
                    improvement = (diff / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                    print(f"  - {method_names[method]} vs Baseline:")
                    print(f"    * Difference: {diff:+.4f}")
                    print(f"    * Improvement: {improvement:+.2f}%")
        
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
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Random seed: 42 (fixed for each rollout)\n\n")
        
        for method in methods_to_run:
            f.write(f"{method_names[method]}:\n")
            if method in rewards_cpu:
                f.write(f"  HPSv2 Reward Mean: {rewards_cpu[method].mean():.4f}\n")
                f.write(f"  HPSv2 Reward Std:  {rewards_cpu[method].std():.4f}\n")
                if "original" in rewards_cpu and method != "original":
                    diff = rewards_cpu[method].mean() - rewards_cpu["original"].mean()
                    improvement = (diff / abs(rewards_cpu["original"].mean())) * 100 if rewards_cpu["original"].mean() != 0 else 0
                    f.write(f"  vs Baseline: {diff:+.4f} ({improvement:+.2f}%)\n")
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"Statistics saved to: {stats_path}")
    print(f"You can now visually compare the results in the following directories:")
    for method in methods_to_run:
        print(f"  - {os.path.join(args.output_path, method)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_comparison()
