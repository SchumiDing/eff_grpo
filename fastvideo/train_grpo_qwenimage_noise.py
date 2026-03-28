# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

"""
对比脚本: 使用简单的噪声方式替代复杂的平均方向计算
当需要guess时,直接添加随机噪声,而不是计算平均方向
"""

# 导入所有必要的模块 (与原脚本相同)
from train_grpo_qwenimage_eff import *
from fastvideo.utils.rollout_image_dir import rollout_image_file

def run_sample_step_noise(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        prompt_attention_mask,
        img_shapes,
        txt_seq_lens,
        grpo_sample,
    ):
    """
    使用简单噪声的采样步骤
    当需要guess时,直接添加随机噪声,而不是计算平均方向
    """
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        all_mock_flags = []
        
        B = z.shape[0]
        num_generations = args.num_generations
        if B % num_generations != 0:
            raise ValueError(
                f"run_sample_step_noise(grpo): batch B={B} must be divisible by num_generations={num_generations}"
            )
        num_prompts = B // num_generations
        last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        # 跟踪每个rollout累计被guess的次数
        cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
        
        for i in progress_bar:
            sigma = sigma_schedule[i]
            d_sigma = 0 - sigma # 到达终点 0 的距离
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
            
            can_guess_mask = ~last_step_guessed_mask
            per_prompt_num_guess = 0 if i == len(progress_bar) - 1 else args.num_guess

            guess_idx_chunks = []
            for p in range(num_prompts):
                start_idx = p * num_generations
                end_idx = (p + 1) * num_generations
                local_can = torch.where(can_guess_mask[start_idx:end_idx])[0] + start_idx
                ng = min(per_prompt_num_guess, int(local_can.numel()))
                if ng > 0:
                    counts = cumulative_guess_count[local_can]
                    order = torch.argsort(counts)[:ng]
                    picked = local_can[order]
                    cumulative_guess_count[picked] += 1
                    guess_idx_chunks.append(picked)

            if guess_idx_chunks:
                current_guess_indices = torch.cat(guess_idx_chunks)
            else:
                current_guess_indices = torch.empty(0, device=z.device, dtype=torch.long)
            
            guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
            if len(current_guess_indices) > 0:
                guessed_mask[current_guess_indices] = True
            infer_mask = ~guessed_mask
            
            # 数据结构准备
            pred = torch.zeros_like(z)
            mock_flag = guessed_mask.clone().to(torch.float32)
            
            # 推理分支
            if infer_mask.any():
                transformer.eval()
                with torch.autocast("cuda", torch.bfloat16):
                    v_infer = transformer(
                        hidden_states=z[infer_mask],
                        timestep=timesteps[infer_mask] / 1000,
                        guidance=None,
                        encoder_hidden_states_mask=prompt_attention_mask[infer_mask],
                        encoder_hidden_states=encoder_hidden_states[infer_mask],
                        img_shapes=img_shapes,
                        txt_seq_lens=[txt_seq_lens[j] for j, m in enumerate(infer_mask) if m],
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                pred[infer_mask] = v_infer.to(pred.dtype)

            # 预测分支 - 使用简单噪声替代复杂计算
            if guessed_mask.any():
                # 直接生成随机噪声作为速度场
                # 噪声的标准差可以根据当前的sigma调整
                noise_scale = sigma.item() * 0.1  # 可调整的噪声强度
                v_guess = torch.randn_like(z[guessed_mask]) * noise_scale
                pred[guessed_mask] = v_guess.to(pred.dtype)
            
            # 演进
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z = z.to(torch.bfloat16)
            
            all_latents.append(z)
            all_log_probs.append(log_prob)
            all_mock_flags.append(mock_flag)
            
            last_step_guessed_mask = guessed_mask

        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, ...)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        all_mock_flags = torch.stack(all_mock_flags, dim=1) # (batch_size, num_steps)
        
        return z, latents, all_latents, all_log_probs, all_mock_flags


def sample_reference_model_noise(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    prompt_attention_masks, 
    original_length,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
):
    """
    使用简单噪声的采样函数
    """
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    all_latents = []
    all_log_probs = []
    all_mock_flags = []
    all_rewards = []  
    all_txt_seq_lens = []
    
    # 并发处理所有rollout - 准备初始噪声
    if args.init_same_noise:
        # 生成一个噪声并扩展到所有样本
        input_latents = torch.randn(
                (1, 1, IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            ).expand(B, 1, IN_CHANNELS, latent_h, latent_w)
    else:
        # 为每个sample生成不同的噪声
        input_latents = torch.randn(
                (B, 1, IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            )
    
    packed_height = 2 * (int(h) // (8 * 2))
    packed_width = 2 * (int(w) // (8 * 2))
    input_latents_new = pack_latents(input_latents, B, 16, packed_height, packed_width)
    
    # 准备所有样本的序列长度和形状信息
    # 处理不同长度的prompt - 先切片再padding (与test_qwen_rollout.py保持一致)
    encoder_hidden_states_processed = []
    prompt_attention_masks_processed = []
    
    for idx in range(B):
        actual_length = original_length[idx].item() if torch.is_tensor(original_length[idx]) else original_length[idx]
        encoder_hidden_states_processed.append(encoder_hidden_states[idx:idx+1, :actual_length, :])
        prompt_attention_masks_processed.append(prompt_attention_masks[idx:idx+1, :actual_length])
    
    # 找到最大序列长度用于padding
    max_seq_len = max(embed.shape[1] for embed in encoder_hidden_states_processed)
    
    # Padding所有embeddings到相同长度
    encoder_hidden_states_padded_list = []
    prompt_attention_masks_padded_list = []
    txt_seq_lens = []
    
    for idx in range(B):
        embed = encoder_hidden_states_processed[idx]  # (1, orig_len, hidden_dim)
        mask = prompt_attention_masks_processed[idx]  # (1, orig_len)
        
        current_len = embed.shape[1]
        if current_len < max_seq_len:
            # Pad embeddings with zeros
            pad_len = max_seq_len - current_len
            embed = torch.nn.functional.pad(embed, (0, 0, 0, pad_len), "constant", 0)
            # Pad attention mask with False (0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), "constant", 0)
        
        encoder_hidden_states_padded_list.append(embed)
        prompt_attention_masks_padded_list.append(mask)
        # txt_seq_lens使用padding后的attention mask的sum (与test_qwen_rollout.py保持一致)
        txt_seq_lens.append(int(mask.sum().item()))
    
    # 拼接成最终的tensors
    encoder_hidden_states_padded = torch.cat(encoder_hidden_states_padded_list, dim=0)
    prompt_attention_masks_padded = torch.cat(prompt_attention_masks_padded_list, dim=0)
    
    # img_shapes应该只有一个元素 (与test_qwen_rollout.py保持一致)
    img_shapes = [[(1, h // 8 // 2, w // 8// 2)]]
    
    grpo_sample = True
    progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress (Noise)")
    
    # 一次性推理所有rollout - 使用噪声版本
    with torch.no_grad():
        z, latents, all_latents_tensor, all_log_probs_tensor, all_mock_flags_tensor = run_sample_step_noise(
            args,
            input_latents_new.clone(),
            progress_bar,
            sigma_schedule,
            transformer,
            encoder_hidden_states_padded,
            prompt_attention_masks_padded,
            img_shapes,
            txt_seq_lens,
            grpo_sample,
        )
    
    # 存储结果
    all_latents.append(all_latents_tensor)
    all_log_probs.append(all_log_probs_tensor)
    all_mock_flags.append(all_mock_flags_tensor)
    all_txt_seq_lens.append(torch.tensor(txt_seq_lens))
    
    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)
    rank = int(os.environ["RANK"])
    
    # 批量解码所有latents
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents_unpacked = unpack_latents(latents, h, w, 8)
            latents_unpacked = latents_unpacked.to(vae.dtype)
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents_unpacked.device, latents_unpacked.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latents_unpacked.device, latents_unpacked.dtype
            )
            latents_unpacked = latents_unpacked / latents_std + latents_mean
            images = vae.decode(latents_unpacked, return_dict=False)[0][:, :, 0]
            decoded_images = image_processor.postprocess(images)
    
    # 计算每个rollout有多少step是guess的
    num_guess_steps = all_mock_flags_tensor.sum(dim=1).cpu().numpy()  # (B,)
    
    # 保存所有图像,文件名包含guess步数统计和noise标记
    for idx in range(B):
        guess_count = int(num_guess_steps[idx])
        decoded_images[idx].save(rollout_image_file(f"qwenimage_{rank}_{idx}_guess{guess_count}_noise.png"))
    
    # 批量计算reward
    if args.use_hpsv2:
        with torch.no_grad():
            for idx in range(B):
                image_path = decoded_images[idx]
                image = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                text = tokenizer([caption[idx]]).to(device=device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = reward_model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                all_rewards.append(hps_score)

    if args.use_hpsv3:
        with torch.no_grad():
            for idx in range(B):
                guess_count = int(num_guess_steps[idx])
                hps_score = reward_model.reward([rollout_image_file(f"qwenimage_{rank}_{idx}_guess{guess_count}_noise.png")], [caption[idx]])
                if hps_score.ndim == 2:
                    hps_score = hps_score[:,0]
                all_rewards.append(hps_score)
    
    if args.use_pickscore:
        def calc_probs(processor, model, prompt, images, device):
            # preprocess
            image_inputs = processor(
                images=images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_inputs = processor(
                text=prompt,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                # embed
                image_embs = model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            
                text_embs = model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            
                # score
                scores = (text_embs @ image_embs.T)[0]
            
            return scores
        
        for idx in range(B):
            guess_count = int(num_guess_steps[idx])
            pil_images = [Image.open(rollout_image_file(f"qwenimage_{rank}_{idx}_guess{guess_count}_noise.png"))]
            score = calc_probs(tokenizer, reward_model, caption[idx], pil_images, device)
            all_rewards.append(score)

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_mock_flags = torch.cat(all_mock_flags, dim=0)
    all_txt_seq_lens = torch.cat(all_txt_seq_lens, dim=0)
    
    # 如果没有使用reward model,创建dummy rewards
    if len(all_rewards) == 0:
        all_rewards = torch.zeros(B, device=device, dtype=torch.float32)
    else:
        all_rewards = torch.cat(all_rewards, dim=0)
    
    return all_rewards, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens, all_mock_flags


# 覆盖 `from train_grpo_qwenimage_eff import *` 带入的 sample_reference_model，否则
# `from train_grpo_qwenimage_noise import sample_reference_model` 会得到 mean 版本而非噪声版本。
sample_reference_model = sample_reference_model_noise

# 注意: 训练循环和其他函数保持不变,需要在 train_one_step 中显式调用 sample_reference_model_noise
# 或改为 from ... import sample_reference_model（此时为本模块绑定后的噪声实现）
