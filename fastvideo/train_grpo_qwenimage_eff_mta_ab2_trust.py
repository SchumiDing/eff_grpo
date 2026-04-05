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
#
# Confidence-gated AB2 rollout fork: keep the best strong-AB2 compute schedule, but
# shrink risky guessed velocities toward a smoothed group anchor when group evidence
# is noisy or sample history disagrees with the current global trend.

import argparse
import math
import os
import json
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper
from fastvideo.utils.validation import log_validation
from fastvideo.utils.rollout_image_dir import rollout_image_file
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_qwenimage_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List, Tuple
from PIL import Image
from fastvideo.utils.fsdp_util_qwenimage import fsdp_wrapper, FSDPConfig
from contextlib import contextmanager
from safetensors.torch import save_file
import copy

class FSDP_EMA:
    def __init__(self, model, decay, rank):
        self.decay = decay
        self.rank = rank
        self.ema_state_dict_rank0 = {}
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            self.ema_state_dict_rank0 = {k: v.clone() for k, v in state_dict.items()}
            main_print("--> Modern EMA handler initialized on rank 0.")

    def update(self, model):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            for key in self.ema_state_dict_rank0:
                if key in model_state_dict:
                    self.ema_state_dict_rank0[key].copy_(
                        self.decay * self.ema_state_dict_rank0[key] + (1 - self.decay) * model_state_dict[key]
                    )

    @contextmanager
    def use_ema_weights(self, model):
        backup_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        backup_state_dict_rank0 = get_model_state_dict(model, options=backup_options)

        load_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_model_state_dict(
            model,
            model_state_dict=self.ema_state_dict_rank0, 
            options=load_options
        )
        
        try:
            yield
        finally:
            restore_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            set_model_state_dict(
                model,
                model_state_dict=backup_state_dict_rank0, 
                options=restore_options
            )

def save_ema_checkpoint(ema_handler, rank, output_dir, step, epoch, config_dict):
    if rank == 0 and ema_handler is not None:
        ema_checkpoint_path = os.path.join(output_dir, f"checkpoint-ema-{step}-{epoch}")
        os.makedirs(ema_checkpoint_path, exist_ok=True)
        weight_path = os.path.join(ema_checkpoint_path ,
                                   "diffusion_pytorch_model.safetensors")
        save_file(ema_handler.ema_state_dict_rank0, weight_path)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(ema_checkpoint_path, "config.json")
        # save dict as json
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        #torch.save(ema_handler.ema_state_dict_rank0, os.path.join(ema_checkpoint_path, "ema_model.pt"))
        main_print(f"--> EMA checkpoint saved at {ema_checkpoint_path}")


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)
    

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def _velocity_mix_alpha(v_a: torch.Tensor, v_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-batch-row scalar in [0, 1] from cosine alignment (high -> trust v_a more)."""
    va = v_a.reshape(v_a.shape[0], -1)
    vb = v_b.reshape(v_b.shape[0], -1)
    num = (va * vb).sum(dim=-1)
    den = (va.norm(dim=-1) * vb.norm(dim=-1)).clamp(min=eps)
    cos = (num / den).clamp(-1.0, 1.0)
    return 0.5 * (cos + 1.0)


_MTA_TRUST_EARLY_FRAC = 0.4
_MTA_TRUST_C1 = 2.0
_MTA_TRUST_C2 = -1.0
_MTA_TRUST_GROUP_EMA = 0.75
_MTA_TRUST_DISP_SCALE = 1.0
_MTA_TRUST_MISMATCH_SCALE = 2.0
_MTA_TRUST_RADIUS = 1.15


def _row_l2_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).norm(dim=-1).clamp(min=eps)


def _clip_delta_by_radius(delta: torch.Tensor, radius: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    delta_norm = _row_l2_norm(delta, eps=eps)
    scale = (radius / delta_norm).clamp(max=1.0)
    return delta * scale.view(-1, *([1] * (delta.ndim - 1)))


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents

def run_sample_step(
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
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        all_mock_flags = []
        
        B = z.shape[0]
        num_generations = args.num_generations
        if B % num_generations != 0:
            raise ValueError(
                f"run_sample_step(grpo): batch B={B} must be divisible by num_generations={num_generations}"
            )
        num_prompts = B // num_generations
        last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        # 跟踪每个rollout累计被guess的次数
        cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
        prev_velocities = torch.zeros_like(z, dtype=torch.float32)
        prev_prev_velocities = torch.zeros_like(z, dtype=torch.float32)
        group_anchor_ema = torch.zeros_like(z, dtype=torch.float32)
        
        for i in progress_bar:
            sigma = sigma_schedule[i]
            d_sigma = 0 - sigma # 到达终点 0 的距离
            old_prev = prev_velocities.clone()
            old_pprev = prev_prev_velocities.clone()
            old_group_anchor = group_anchor_ema.clone()
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
            
            # 按 prompt 分组：每组 num_generations 个 rollout；每组内独立选最多 num_guess 个去 mock
            can_guess_mask = ~last_step_guessed_mask
            per_prompt_num_guess = 0 if i == args.sampling_steps - 1 else args.num_guess

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
            v_mean = torch.zeros_like(z, dtype=torch.float32)
            group_dispersion = torch.zeros(B, device=z.device, dtype=torch.float32)
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
                
                # --- 按rollout维度计算每个sample的平均速度场 ---
                # 遍历每个prompt，对其对应的num_generations个rollout计算平均
                for p in range(num_prompts):
                    start_idx = p * num_generations
                    end_idx = (p + 1) * num_generations
                    
                    # 获取该prompt对应的所有rollout的mask
                    group_infer_mask = infer_mask[start_idx:end_idx]
                    
                    if group_infer_mask.any():
                        # 只对推理的样本计算平均速度场
                        # 统计该组内有多少个infer样本
                        offset = infer_mask[:start_idx].sum().item()
                        count = group_infer_mask.sum().item()
                        
                        # 从v_infer中提取该组的速度场并计算平均，同时估计组内离散度。
                        group_vals = v_infer[offset : offset + count].to(torch.float32)
                        group_v_mean = group_vals.mean(dim=0)
                        v_mean[start_idx:end_idx] = group_v_mean
                        centered = group_vals - group_v_mean.unsqueeze(0)
                        mean_norm = _row_l2_norm(group_v_mean.unsqueeze(0))[0]
                        disp = centered.reshape(count, -1).norm(dim=-1).mean() / mean_norm
                        group_dispersion[start_idx:end_idx] = disp
                    else:
                        # 如果整个组都没有infer（理论上不应发生），使用零向量
                        v_mean[start_idx:end_idx] = 0.0
                
                # 根据平均速度场计算guess的预测终点
                x_L_mean = z.to(torch.float32) + v_mean * d_sigma
            else:
                x_L_mean = z.to(torch.float32).mean(dim=0, keepdim=True).expand(B, *z.shape[1:])

            if infer_mask.any():
                if i == 0:
                    group_anchor_ema.copy_(v_mean)
                else:
                    group_anchor_ema.mul_(_MTA_TRUST_GROUP_EMA).add_(
                        v_mean, alpha=(1.0 - _MTA_TRUST_GROUP_EMA)
                    )

            # 预测分支：前半段使用 smoothed group anchor + trust region；后半段只用 strong AB2。
            early_group_fusion = i < len(progress_bar) * _MTA_TRUST_EARLY_FRAC
            if guessed_mask.any():
                v_prev = old_prev[guessed_mask]
                v_pp = old_pprev[guessed_mask]
                if i >= 2:
                    v_so = _MTA_TRUST_C1 * v_prev + _MTA_TRUST_C2 * v_pp
                else:
                    v_so = v_prev
                if early_group_fusion:
                    if infer_mask.any():
                        v_group = group_anchor_ema[guessed_mask]
                    else:
                        ds = d_sigma if abs(d_sigma) > 1e-6 else 1e-6
                        v_group = (x_L_mean[guessed_mask] - z[guessed_mask].to(torch.float32)) / ds
                    alpha = _velocity_mix_alpha(v_so, v_group)
                    alpha = alpha.view(-1, *([1] * (v_so.ndim - 1)))
                    v_mid = alpha * v_so + (1.0 - alpha) * v_group
                    mismatch = _row_l2_norm(v_group - v_prev) / _row_l2_norm(v_prev)
                    conf = torch.exp(
                        -_MTA_TRUST_DISP_SCALE * group_dispersion[guessed_mask].square()
                        -_MTA_TRUST_MISMATCH_SCALE * mismatch.square()
                    )
                    ref_radius = torch.maximum(
                        _row_l2_norm(v_prev - old_group_anchor[guessed_mask]),
                        0.10 * _row_l2_norm(v_group),
                    )
                    radius = _MTA_TRUST_RADIUS * (0.35 + 0.65 * conf) * ref_radius
                    delta = _clip_delta_by_radius(v_mid - v_group, radius)
                    v_hat = v_group + delta
                else:
                    v_hat = v_so
                pred[guessed_mask] = v_hat.to(pred.dtype)

            prev_prev_velocities.copy_(old_prev)
            if infer_mask.any():
                prev_velocities[infer_mask] = pred[infer_mask].to(torch.float32)
            if guessed_mask.any():
                prev_velocities[guessed_mask] = pred[guessed_mask].to(torch.float32)
            
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

        
def grpo_one_step(
        args,
        latents,
        pre_latents,
        encoder_hidden_states, 
        prompt_attention_masks, 
        txt_seq_lens,
        img_shapes,
        transformer,
        timesteps,
        i,
        sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=latents,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_attention_masks,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=None,
            return_dict=False,
        )[0]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob



def sample_reference_model(
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
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    progress_bar = tqdm(
        range(0, sample_steps),
        desc="Sampling Progress",
        disable=_local_rank > 0,
        leave=False,
    )
    
    # 一次性推理所有rollout
    with torch.no_grad():
        z, latents, all_latents_tensor, all_log_probs_tensor, all_mock_flags_tensor = run_sample_step(
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
    
    # 保存所有图像,文件名包含guess步数统计
    for idx in range(B):
        guess_count = int(num_guess_steps[idx])
        decoded_images[idx].save(rollout_image_file(f"qwenimage_{rank}_{idx}_guess{guess_count}.png"))
    
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
                hps_score = reward_model.reward([rollout_image_file(f"qwenimage_{rank}_{idx}_guess{guess_count}.png")], [caption[idx]])
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
            pil_images = [Image.open(rollout_image_file(f"qwenimage_{rank}_{idx}_guess{guess_count}.png"))]
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


def _trust_wrap(
    early_frac: float,
    c1: float,
    c2: float,
    group_ema: float,
    disp_scale: float,
    mismatch_scale: float,
    trust_radius: float,
):
    """Return a sample_reference_model-compatible callable with temporary trust knobs."""

    def wrapped(
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
        global _MTA_TRUST_EARLY_FRAC, _MTA_TRUST_C1, _MTA_TRUST_C2
        global _MTA_TRUST_GROUP_EMA, _MTA_TRUST_DISP_SCALE
        global _MTA_TRUST_MISMATCH_SCALE, _MTA_TRUST_RADIUS
        old = (
            _MTA_TRUST_EARLY_FRAC,
            _MTA_TRUST_C1,
            _MTA_TRUST_C2,
            _MTA_TRUST_GROUP_EMA,
            _MTA_TRUST_DISP_SCALE,
            _MTA_TRUST_MISMATCH_SCALE,
            _MTA_TRUST_RADIUS,
        )
        _MTA_TRUST_EARLY_FRAC = early_frac
        _MTA_TRUST_C1 = c1
        _MTA_TRUST_C2 = c2
        _MTA_TRUST_GROUP_EMA = group_ema
        _MTA_TRUST_DISP_SCALE = disp_scale
        _MTA_TRUST_MISMATCH_SCALE = mismatch_scale
        _MTA_TRUST_RADIUS = trust_radius
        try:
            return sample_reference_model(
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
            )
        finally:
            (
                _MTA_TRUST_EARLY_FRAC,
                _MTA_TRUST_C1,
                _MTA_TRUST_C2,
                _MTA_TRUST_GROUP_EMA,
                _MTA_TRUST_DISP_SCALE,
                _MTA_TRUST_MISMATCH_SCALE,
                _MTA_TRUST_RADIUS,
            ) = old

    return wrapped


sample_reference_model_ab2_trust = _trust_wrap(0.4, 2.0, -1.0, 0.75, 1.0, 2.0, 1.15)
sample_reference_model_ab2_trust_tight = _trust_wrap(0.4, 2.0, -1.0, 0.85, 1.3, 2.5, 0.95)


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def _apply_ab2_trust_preset(preset: str) -> None:
    """Set rollout knobs for training."""
    global _MTA_TRUST_EARLY_FRAC, _MTA_TRUST_C1, _MTA_TRUST_C2
    global _MTA_TRUST_GROUP_EMA, _MTA_TRUST_DISP_SCALE
    global _MTA_TRUST_MISMATCH_SCALE, _MTA_TRUST_RADIUS
    if preset == "default":
        _MTA_TRUST_EARLY_FRAC = 0.4
        _MTA_TRUST_C1 = 2.0
        _MTA_TRUST_C2 = -1.0
        _MTA_TRUST_GROUP_EMA = 0.75
        _MTA_TRUST_DISP_SCALE = 1.0
        _MTA_TRUST_MISMATCH_SCALE = 2.0
        _MTA_TRUST_RADIUS = 1.15
    elif preset == "tight":
        _MTA_TRUST_EARLY_FRAC = 0.4
        _MTA_TRUST_C1 = 2.0
        _MTA_TRUST_C2 = -1.0
        _MTA_TRUST_GROUP_EMA = 0.85
        _MTA_TRUST_DISP_SCALE = 1.3
        _MTA_TRUST_MISMATCH_SCALE = 2.5
        _MTA_TRUST_RADIUS = 0.95
    else:
        raise ValueError(f"unknown ab2_trust_preset: {preset}")


def _batched_infer_shuffle_sel(
    mock_flags: torch.Tensor, timestep_fraction: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shuffle non-guess (infer) timesteps per batch row; take int(fraction * count) each.

    Replaces a Python loop of ``where`` + ``randperm`` with one GPU-friendly ``argsort(B, T)``.
    A custom Triton kernel is a poor fit here: typical ``T`` is small (dozens) and launch overhead
    would dominate; PyTorch already fuses comparable sort/rand patterns well.

    Returns
    -------
    sel : LongTensor (B, max_k)
        For row ``b``, ``sel[b, :k[b]]`` is a random permutation of infer timestep indices.
    k : LongTensor (B,)
        Selected count per row (0 .. count inclusive).
    """
    device = mock_flags.device
    B, T = mock_flags.shape
    infer_mask = mock_flags < 0.5
    counts = infer_mask.sum(dim=1)
    # Match single-row fallback: if no infer slots, use all timestep indices.
    if (counts == 0).any():
        infer_mask = infer_mask | (counts.unsqueeze(1) == 0)
        counts = infer_mask.sum(dim=1)

    k = (counts.to(dtype=torch.float32) * timestep_fraction).long()
    k = torch.minimum(k, counts)

    noise = torch.rand(B, T, device=device, dtype=torch.float32)
    inf = torch.tensor(float("inf"), device=device, dtype=torch.float32)
    noise = torch.where(infer_mask, noise, inf)
    perm = torch.argsort(noise, dim=-1)

    max_k = int(k.max().item())
    if max_k == 0:
        return mock_flags.new_empty((B, 0), dtype=torch.long), k
    sel = perm[:, :max_k].contiguous().long()
    return sel, k


def train_one_step(
    args,
    device,
    transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    prompt_embeds, 
    prompt_attention_masks, 
    caption, 
    original_length,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    ema_handler
):
    total_loss = 0.0
    grad_norm = torch.tensor(0.0).to(device)

    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(prompt_embeds)
        prompt_attention_masks = repeat_tensor(prompt_attention_masks)
        original_length = repeat_tensor(original_length)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    reward, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens, all_mock_flags = sample_reference_model(
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
            preprocess_val
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "mock_flags": all_mock_flags[:, :-1],
        "rewards": reward.to(torch.float32),
        "txt_seq_lens": all_txt_seq_lens,
        "encoder_hidden_states": encoder_hidden_states,
        "prompt_attention_masks": prompt_attention_masks,
        "original_length": original_length,
    }
    gathered_reward = gather_tensor(samples["rewards"])
    avg_reward = gathered_reward.mean().item()
    #计算advantage
    if args.use_group:
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages

    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)

    if getattr(args, "zero_loss_on_guess", False):
        # Shuffle only non-guess (infer) timesteps; skip mock steps in the update loop entirely.
        sel_all, k_tensor = _batched_infer_shuffle_sel(
            samples["mock_flags"], float(args.timestep_fraction)
        )
        ol = samples["original_length"]
        train_entries = []
        for b in range(batch_size):
            kb = int(k_tensor[b].item())
            row_sel = sel_all[b, :kb]
            ol_b = ol[b : b + 1] if torch.is_tensor(ol) else ol
            train_entries.append(
                {
                    "latents": samples["latents"][b : b + 1, row_sel],
                    "next_latents": samples["next_latents"][b : b + 1, row_sel],
                    "timesteps": samples["timesteps"][b : b + 1, row_sel],
                    "log_probs": samples["log_probs"][b : b + 1, row_sel],
                    "sigma_indices": row_sel,
                    "k": kb,
                    "advantages": samples["advantages"][b : b + 1],
                    "rewards": samples["rewards"][b : b + 1],
                    "encoder_hidden_states": samples["encoder_hidden_states"][b : b + 1],
                    "prompt_attention_masks": samples["prompt_attention_masks"][b : b + 1],
                    "original_length": ol_b,
                    "txt_seq_lens": samples["txt_seq_lens"][b : b + 1],
                }
            )
        for i, sample in list(enumerate(train_entries)):
            k_i = sample["k"]
            for j in range(k_i):
                clip_range = args.clip_range
                adv_clip_max = args.adv_clip_max
                sigma_i = int(sample["sigma_indices"][j].item())
                new_log_probs = grpo_one_step(
                    args,
                    sample["latents"][:, j],
                    sample["next_latents"][:, j],
                    sample["encoder_hidden_states"][:, : sample["original_length"]],
                    sample["prompt_attention_masks"][:, : sample["original_length"]],
                    sample["txt_seq_lens"],
                    [[(1, args.h // 8 // 2, args.w // 8 // 2)]],
                    transformer,
                    sample["timesteps"][:, j],
                    sigma_i,
                    sigma_schedule,
                )

                advantages = torch.clamp(
                    sample["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )

                ratio = torch.exp(new_log_probs - sample["log_probs"][:, j])

                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )

                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (
                    args.gradient_accumulation_steps * k_i
                )

                loss.backward()
                avg_loss = loss.detach().clone()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                total_loss += avg_loss.item()
            if (i + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = transformer.clip_grad_norm_(max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if dist.get_rank() % 8 == 0 and k_i > 0:
                print("reward", sample["rewards"].item())
                print("ratio", ratio)
                print("advantage", sample["advantages"].item())
                print("final loss", loss.item())
            dist.barrier()
    else:
        perms = torch.stack(
            [
                torch.randperm(len(samples["timesteps"][0]))
                for _ in range(batch_size)
            ]
        ).to(device)
        for key in ["timesteps", "latents", "next_latents", "log_probs", "mock_flags"]:
            samples[key] = samples[key][
                torch.arange(batch_size).to(device)[:, None],
                perms,
            ]
        samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
        samples_batched_list = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]
        for i, sample in list(enumerate(samples_batched_list)):
            for _ in range(train_timesteps):
                clip_range = args.clip_range
                adv_clip_max = args.adv_clip_max
                new_log_probs = grpo_one_step(
                    args,
                    sample["latents"][:, _],
                    sample["next_latents"][:, _],
                    sample["encoder_hidden_states"][:, : sample["original_length"]],
                    sample["prompt_attention_masks"][:, : sample["original_length"]],
                    sample["txt_seq_lens"],
                    [[(1, args.h // 8 // 2, args.w // 8 // 2)]],
                    transformer,
                    sample["timesteps"][:, _],
                    perms[i][_],
                    sigma_schedule,
                )

                advantages = torch.clamp(
                    sample["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )

                ratio = torch.exp(new_log_probs - sample["log_probs"][:, _])

                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )

                loss_weight = 1.0 - 0.9 * sample["mock_flags"][:, _]

                loss = torch.mean(
                    torch.maximum(unclipped_loss, clipped_loss) * loss_weight
                ) / (args.gradient_accumulation_steps * train_timesteps)

                loss.backward()
                avg_loss = loss.detach().clone()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                total_loss += avg_loss.item()
            if (i + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = transformer.clip_grad_norm_(max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if dist.get_rank() % 8 == 0:
                print("reward", sample["rewards"].item())
                print("ratio", ratio)
                print("advantage", sample["advantages"].item())
                print("final loss", loss.item())
            dist.barrier()
    return total_loss, grad_norm.item(), avg_reward


def main(args):
    rollout_dir = os.path.abspath(os.path.expanduser(os.path.normpath(args.rollout_image_dir)))
    os.makedirs(rollout_dir, exist_ok=True)
    os.environ["DANCEGRPO_ROLLOUT_IMAGE_DIR"] = rollout_dir

    _apply_ab2_trust_preset(args.ab2_trust_preset)

    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if rank == 0:
        main_print(
            f"Rollout decode scratch dir (DANCEGRPO_ROLLOUT_IMAGE_DIR): {rollout_dir}"
        )
        main_print(
            f"AB2-trust preset={args.ab2_trust_preset}: early_frac={_MTA_TRUST_EARLY_FRAC}, "
            f"c1={_MTA_TRUST_C1}, c2={_MTA_TRUST_C2}, group_ema={_MTA_TRUST_GROUP_EMA}, "
            f"disp_scale={_MTA_TRUST_DISP_SCALE}, mismatch_scale={_MTA_TRUST_MISMATCH_SCALE}, "
            f"trust_radius={_MTA_TRUST_RADIUS}; zero_loss_on_guess={getattr(args, 'zero_loss_on_guess', False)}"
        )

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    preprocess_val = None
    processor = None
    if args.use_hpsv2:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from typing import Union
        import huggingface_hub
        from hpsv2.utils import root_path, hps_version_map
        def initialize_model():
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
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        #cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location=f'cuda:{device}')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()

    if args.use_pickscore:
        from transformers import AutoProcessor, AutoModel
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        reward_model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    if args.use_hpsv3:
        from hpsv3 import HPSv3RewardInferencer
        reward_model = HPSv3RewardInferencer(device=device)

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenImageTransformerBlock
    transformer = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )

    # Setup FSDP configuration
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,  
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=torch.bfloat16,
        use_device_mesh=False, 
    )
    transformer = fsdp_wrapper(transformer, fsdp_config,)

    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    apply_fsdp_checkpointing(
            transformer, (QwenImageTransformerBlock), args.selective_checkpointing
        )

    from diffusers.models.autoencoders import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=None,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    #vae.enable_tiling()

    if rank <= 0:
        project = "qwenimage"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    train_bar_total = args.max_train_steps if args.max_train_steps is not None else 100000
    progress_bar = tqdm(
        range(0, train_bar_total),
        initial=min(init_steps, train_bar_total),
        total=train_bar_total,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )



    step_times = deque(maxlen=100)

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        for step, (prompt_embeds, prompt_attention_masks, caption, original_length) in enumerate(train_dataloader):
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_masks = prompt_attention_masks.to(device)
            start_time = time.time()
            if (step-1) % args.checkpointing_steps == 0 and step!=1:
                save_checkpoint(transformer, rank, args.output_dir,
                                step, epoch)
                if args.use_ema:
                    save_ema_checkpoint(ema_handler, rank, args.output_dir, step, epoch, dict(transformer.config))


                dist.barrier()
            if step > (args.max_train_steps+1):
                break
            loss, grad_norm, avg_reward = train_one_step(
                args,
                device, 
                transformer,
                vae,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                prompt_embeds, 
                prompt_attention_masks, 
                caption, 
                original_length,
                noise_scheduler,
                args.max_grad_norm,
                preprocess_val,
                ema_handler,
            )

            if args.use_ema and ema_handler:
                ema_handler.update(transformer)
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "reward": f"{avg_reward:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                log_data = {
                    "step": step,
                    "train_loss": loss,
                    "avg_reward": avg_reward,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                    "epoch": epoch,
                }
                
                # Log to WandB
                wandb.log(log_data, step=step)
                
                # Log to logging.jsonl
                if args.output_dir is not None:
                    log_file = os.path.join(args.output_dir, "logging.jsonl")
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_data) + "\n")



    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--rollout_image_dir",
        type=str,
        default="data/outputs/rollout_scratch_train_ab2_trust",
        help="Temporary decoded PNG directory during rollout for AB2-trust.",
    )

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    )
    parser.add_argument(
        "--use_pickscore",
        action="store_true",
        default=False,
        help="whether use pickscore as reward model",
    )
    parser.add_argument(
        "--use_hpsv3",
        action="store_true",
        default=False,
        help="whether use hpsv3 as reward model",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Enable Exponential Moving Average of model weights."
    )
    parser.add_argument(
        "--num_infer",
        type=int,
        default=12,
        help="number of samples for model inference per step",
    )
    parser.add_argument(
        "--num_guess",
        type=int,
        default=4,
        help="per prompt: max rollouts to mock-guess each denoising step (within that prompt's group of num_generations)",
    )
    parser.add_argument(
        "--ab2_trust_preset",
        type=str,
        default="default",
        choices=["default", "tight"],
        help="AB2-trust rollout hyperparams. default uses EMA anchor + trust region; "
        "tight is more conservative for tail protection.",
    )
    parser.add_argument(
        "--zero_loss_on_guess",
        action="store_true",
        default=False,
        help="If set, training shuffles and updates only non-guess (infer) timesteps; guess steps are omitted from the update loop. "
        "Loss per rollout is normalized by the number of those steps (min(timestep_fraction*T, num_infer)).",
    )

    args = parser.parse_args()
    main(args)