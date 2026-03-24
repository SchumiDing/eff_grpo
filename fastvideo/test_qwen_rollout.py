import torch
import os
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
import json

# 导入 eff 脚本中的关键函数和类
from train_grpo_qwenimage_eff import sample_reference_model, sd3_time_shift

def test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/qwenimage", help="Path to qwenimage weights")
    parser.add_argument("--embeddings_path", type=str, default="data/qwenimage/rl_embeddings", help="Path to rl embeddings")
    parser.add_argument("--num_generations", type=int, default=12, help="Number of samples per prompt")
    parser.add_argument("--num_guess", type=int, default=3, help="Number of samples to guess per step")
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--output_path", type=str, default="./test_results")
    parser.add_argument("--init_same_noise", action="store_true", default=False, help="Use same noise for all samples in a group")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of embeddings to load")
    parser.add_argument("--end_idx", type=int, default=2, help="End index of embeddings to load (None means all)")
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output_path, exist_ok=True)

    # 1. 加载组件 (与训练脚本保持一致)
    print("Loading models...")
    # 从训练脚本导入正确的类
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
    
    # 2. 加载 RL embeddings (与训练脚本保持一致)
    print(f"Loading embeddings from {args.embeddings_path}...")
    json_path = os.path.join(args.embeddings_path, "videos2caption.json")
    prompt_embed_dir = os.path.join(args.embeddings_path, "prompt_embed")
    prompt_attention_mask_dir = os.path.join(args.embeddings_path, "prompt_attention_mask")
    
    with open(json_path, "r") as f:
        data_anno = json.load(f)
    
    # 确定要处理的数据范围
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
    
    # Stack embeddings
    all_prompt_embeds = torch.stack(all_prompt_embeds, dim=0)  # (N, seq_len, hidden_dim)
    all_prompt_attention_masks = torch.stack(all_prompt_attention_masks, dim=0)  # (N, seq_len)
    
    print(f"Loaded embeddings shape: {all_prompt_embeds.shape}")
    print(f"Total prompts: {len(all_captions)}")
    
    # 3. 使用 sample_reference_model 函数进行采样
    print("\n" + "="*80)
    print("FULL BATCH ROLLOUT MODE - Using sample_reference_model")
    print("="*80)
    
    num_prompts = len(all_captions)
    
    # 为了适配 sample_reference_model, 模拟 args 对象
    class MockArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # 创建 mock args (包含所有需要的参数)
    mock_args = MockArgs(
        num_guess=args.num_guess,
        num_generations=args.num_generations,
        eta=args.eta,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        init_same_noise=args.init_same_noise,
        w=args.width,
        h=args.height,
        t=1,  # 单帧
        use_hpsv2=False,
        use_hpsv3=False,
        use_pickscore=False,
    )
    
    # 每个 prompt 重复 num_generations 次 (与训练脚本保持一致)
    print(f"Expanding each prompt {args.num_generations} times...")
    encoder_hidden_states_list = []
    prompt_attention_masks_list = []
    original_length_list = []
    captions_expanded = []
    
    for i in tqdm(range(num_prompts), desc="Expanding prompts"):
        for _ in range(args.num_generations):
            encoder_hidden_states_list.append(all_prompt_embeds[i:i+1])
            prompt_attention_masks_list.append(all_prompt_attention_masks[i:i+1])
            original_length_list.append(all_original_lengths[i])
            captions_expanded.append(all_captions[i])
    
    # 拼接成最终的 tensors
    print("Concatenating all embeddings into one big batch...")
    encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0).to(device)
    prompt_attention_masks = torch.cat(prompt_attention_masks_list, dim=0).to(device)
    original_length = torch.tensor(original_length_list).to(device)
    
    total_samples = num_prompts * args.num_generations
    print(f"\n{'='*80}")
    print(f"Total samples in batch: {total_samples} ({num_prompts} prompts × {args.num_generations} generations)")
    print(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
    print(f"Prompt attention masks shape: {prompt_attention_masks.shape}")
    print(f"Original lengths (first 5): {original_length[:5].tolist()}")
    print(f"{'='*80}\n")
    
    # 调用 sample_reference_model 函数
    print(f"\n{'='*80}")
    print(f"Starting FULL BATCH sampling for {total_samples} samples...")
    print(f"{'='*80}\n")
    
    # sample_reference_model 会处理所有的采样、解码和保存
    all_rewards, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens, all_mock_flags = sample_reference_model(
        mock_args,
        device,
        transformer,
        vae,
        encoder_hidden_states,
        prompt_attention_masks,
        original_length,
        reward_model=None,  # 不使用reward model
        tokenizer=None,
        caption=captions_expanded,
        preprocess_val=None,
    )
    
    print(f"\n{'='*80}")
    print("Sampling completed!")
    print(f"{'='*80}\n")
    
    # 计算每个rollout有多少step是guess的
    num_guess_steps = all_mock_flags.sum(dim=1).cpu().numpy()  # (total_samples,)
    
    # 读取已保存的图像并重新组织
    print("Reorganizing saved images...")
    decoded_images = []
    rank = 0  # 单机测试时rank为0
    for i in range(total_samples):
        guess_count = int(num_guess_steps[i])
        img_path = f"./images/qwenimage_{rank}_{i}_guess{guess_count}.png"
        if os.path.exists(img_path):
            decoded_images.append(Image.open(img_path))
        else:
            print(f"Warning: Image {img_path} not found!")
    
    print(f"Loaded {len(decoded_images)} images successfully!\n")
    
    # 重新组织和保存图像到指定目录
    print("Reorganizing and saving images to output directory...")
    rank = 0
    for i in tqdm(range(len(decoded_images)), desc="Saving images"):
        prompt_idx = i // args.num_generations
        gen_idx = i % args.num_generations
        guess_count = int(num_guess_steps[i])
        is_mock = "guess" if all_mock_flags[i, -1] > 0.5 else "infer"
        
        # 创建子目录
        prompt_dir = os.path.join(args.output_path, f"prompt_{prompt_idx}")
        os.makedirs(prompt_dir, exist_ok=True)
        
        # 从临时位置复制到最终位置,文件名包含guess步数
        src_path = f"./images/qwenimage_{rank}_{i}_guess{guess_count}.png"
        dst_path = os.path.join(prompt_dir, f"gen_{gen_idx}_guess{guess_count}_{is_mock}.png")
        
        if os.path.exists(src_path):
            decoded_images[i].save(dst_path)
        
        if gen_idx == 0:  # 只在第一个生成时打印 caption
            tqdm.write(f"Prompt {prompt_idx}: {captions_expanded[i]}")
    
    print(f"\n{'='*80}")
    print(f"SUCCESS! All {num_prompts} prompts × {args.num_generations} generations = {total_samples} samples generated!")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_main()
