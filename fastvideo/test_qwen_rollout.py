import torch
import os
import argparse
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers.models.autoencoders import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

# 导入 eff 脚本中的关键函数
from train_grpo_qwenimage_eff import run_sample_step, pack_latents, unpack_latents, sd3_time_shift

def test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/qwenimage", help="Path to qwenimage weights")
    parser.add_argument("--prompt", type=str, default="A beautiful cat sitting on a sofa, high quality, highly detailed", help="Test prompt")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of samples in one rollout group")
    parser.add_argument("--num_guess", type=int, default=2, help="Number of samples to guess per step")
    parser.add_argument("--sampling_steps", type=int, default=25)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--output_path", type=str, default="./test_results")
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output_path, exist_ok=True)

    # 1. 加载组件 (参考 preprocess 逻辑使用 DiffusionPipeline 编码)
    print("Loading models...")
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    transformer = QwenImageTransformer2DModel.from_pretrained(args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16).to(device)
    vae = AutoencoderKLQwenImage.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    
    # 2. 编码 Prompt (参考 preprocess_qwenimage_embedding.py 逻辑)
    print(f"Encoding prompt: {args.prompt}")
    with torch.inference_mode():
        prompt_embeds, prompt_attention_mask = pipe.encode_prompt(prompt=[args.prompt])
        # 模拟预处理中的 padding 逻辑
        target_length = 1024
        original_length = prompt_embeds.shape[1]
        pad_len = target_length - original_length
        prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad_len), "constant", 0)
        prompt_attention_mask = torch.nn.functional.pad(prompt_attention_mask, (0, pad_len), "constant", 0)

    # 准备 Batch (重复 num_generations 次组成一个 rollout group)
    B = args.num_generations
    prompt_embeds = prompt_embeds.repeat(B, 1, 1).to(device)
    prompt_attention_mask = prompt_attention_mask.repeat(B, 1).to(device)
    txt_seq_lens = [original_length] * B
    img_shapes = [[(1, args.height // 8 // 2, args.width // 8 // 2)]]

    # 3. 准备噪声
    IN_CHANNELS = 16
    latent_h, latent_w = args.height // 8, args.width // 8
    input_latents = torch.randn((B, 1, IN_CHANNELS, latent_h, latent_w), device=device, dtype=torch.bfloat16)
    
    # Packing latents
    packed_height = 2 * (args.height // (8 * 2))
    packed_width = 2 * (args.width // (8 * 2))
    z_input = pack_latents(input_latents, B, IN_CHANNELS, packed_height, packed_width)

    # 4. 采样 (使用 eff 逻辑)
    print("Starting efficient sampling...")
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1).to(device)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    progress_bar = tqdm(range(args.sampling_steps), desc="Steps")

    # 为了适配 run_sample_step, 模拟 args 对象
    class MockArgs:
        def __init__(self, n_guess, n_gen, eta):
            self.num_guess = n_guess
            self.num_generations = n_gen
            self.eta = eta
    
    mock_args = MockArgs(args.num_guess, args.num_generations, args.eta)

    with torch.no_grad():
        z_final, x0_pred, all_z, all_log_probs, all_mock_flags = run_sample_step(
            mock_args,
            z_input,
            progress_bar,
            sigma_schedule,
            transformer,
            prompt_embeds,
            prompt_attention_mask,
            img_shapes,
            txt_seq_lens,
            grpo_sample=True
        )

    # 5. 解码并保存 (参考原版解码逻辑)
    print("Decoding results...")
    image_processor = VaeImageProcessor(16)
    with torch.inference_mode():
        # 解码 x0_pred (即模型预测的终点)
        latents = unpack_latents(x0_pred, args.height, args.width, 8)
        latents = latents.to(vae.dtype)
        
        # VAE 归一化
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1).to(device, torch.bfloat16)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1).to(device, torch.bfloat16)
        latents = latents / latents_std + latents_mean
        
        decoded_images = vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = image_processor.postprocess(decoded_images)

        for i, img in enumerate(images):
            is_mock = "guess" if i in torch.where(all_mock_flags[:, -1])[0] else "infer"
            save_name = os.path.join(args.output_path, f"sample_{i}_{is_mock}.png")
            img.save(save_name)
            print(f"Saved: {save_name}")

if __name__ == "__main__":
    test_main()