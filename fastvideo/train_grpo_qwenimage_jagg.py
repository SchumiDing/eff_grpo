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
# Jacobian-Aggregation (JAGG) backward acceleration fork.
#
# Core idea: for each group of `sum_sample` consecutive denoising timesteps within
# one rollout, approximate the summed gradient via linear Jacobian interpolation
# between the first and last timestep in the group.  This reduces W backward passes
# to exactly 2, while keeping all W forward passes (no-grad, cheap).
#
# Math:
#   True gradient:    G = Σ_j  s_j^T · J(z_j, t_j)
#   Approximation:    G ≈ s_base^T · J(z_0, t_0)  +  s_corr^T · J(z_{W-1}, t_{W-1})
#   where  α_j = j/(W-1),   s_base = Σ (1-α_j)·s_j,   s_corr = Σ α_j·s_j
#
# `s_j` is the per-step upstream gradient dL_j/dpred_j computed cheaply through the
# analytical flux_step + PPO formula (no transformer backward needed).

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


# ---------------------------------------------------------------------------
# EMA helpers (unchanged from base)
# ---------------------------------------------------------------------------

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
        weight_path = os.path.join(ema_checkpoint_path,
                                   "diffusion_pytorch_model.safetensors")
        save_file(ema_handler.ema_state_dict_rank0, weight_path)
        if "dtype" in config_dict:
            del config_dict["dtype"]
        config_path = os.path.join(ema_checkpoint_path, "config.json")
        import json as _json
        with open(config_path, "w") as f:
            _json.dump(config_dict, f, indent=4)
        main_print(f"--> EMA checkpoint saved at {ema_checkpoint_path}")


# ---------------------------------------------------------------------------
# Diffusion primitives (unchanged from base)
# ---------------------------------------------------------------------------

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
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
    return latents


# ---------------------------------------------------------------------------
# Rollout (unchanged from base)
# ---------------------------------------------------------------------------

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
        for i in progress_bar:
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                pred = transformer(
                    hidden_states=z,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            z, pred_original, log_prob = flux_step(
                pred, z.to(torch.float32), args.eta,
                sigmas=sigma_schedule, index=i,
                prev_sample=None, grpo=True, sde_solver=True,
            )
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        return z, latents, all_latents, all_log_probs


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
    """Single-step forward+backward (original, used as fallback)."""
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred = transformer(
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
    z, pred_original, log_prob = flux_step(
        pred, latents.to(torch.float32), args.eta,
        sigma_schedule, i,
        prev_sample=pre_latents.to(torch.float32),
        grpo=True, sde_solver=True,
    )
    return log_prob


# ---------------------------------------------------------------------------
# Jacobian-aggregated group backward  (NEW — core of JAGG)
# ---------------------------------------------------------------------------

def grpo_group_backward(
    args,
    group_latents: torch.Tensor,
    group_next_latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    prompt_attention_masks: torch.Tensor,
    txt_seq_lens,
    img_shapes,
    transformer,
    group_timesteps: torch.Tensor,
    group_sigma_indices: List[int],
    sigma_schedule: torch.Tensor,
    group_old_log_probs: torch.Tensor,
    advantages_clamped: torch.Tensor,
    clip_range: float,
    loss_norm_factor: float,
) -> Tuple[float, torch.Tensor]:
    """Jacobian-aggregated backward for W = sum_sample consecutive timesteps.

    Instead of W separate forward+backward, this performs:
      - W  forward passes WITHOUT grad  (cheap: no activation storage)
      - 2  forward passes WITH grad     (representative + perturbed)
      - 2  backward passes              (base + Taylor correction)

    The gradient approximation is exact for W <= 2 and first-order accurate
    (linear Jacobian interpolation) for W >= 3.

    Args:
        group_latents:        (1, W, ...) latents BEFORE each step
        group_next_latents:   (1, W, ...) latents AFTER each step
        group_timesteps:      (1, W) integer timestep values
        group_sigma_indices:  list[int] of length W — original indices into sigma_schedule
        group_old_log_probs:  (1, W) reference log probs from rollout
        advantages_clamped:   (1,) already-clamped advantage scalar
        loss_norm_factor:     scalar divisor (grad_accum_steps * total_train_timesteps)

    Returns:
        (total_loss_value, mean_ratio)
    """
    W = group_latents.shape[1]
    device = group_latents.device

    # ------------------------------------------------------------------
    # Phase 1 — cheap upstream gradients s_j = dL_j / dpred_j
    #
    # For each step j we run a no-grad forward through the transformer to
    # get pred_j, then differentiate the scalar PPO loss w.r.t. a detached
    # proxy of pred_j.  This only backprops through the simple flux_step +
    # PPO formula (a handful of element-wise ops), NOT through the model.
    # ------------------------------------------------------------------
    upstream_list: List[torch.Tensor] = []
    group_loss = 0.0
    ratio_list: List[torch.Tensor] = []

    for j in range(W):
        z_j = group_latents[:, j]
        z_next_j = group_next_latents[:, j]
        t_j = group_timesteps[:, j]
        sigma_idx = group_sigma_indices[j]

        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                pred_j = transformer(
                    hidden_states=z_j,
                    timestep=t_j / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_attention_masks,
                    encoder_hidden_states=encoder_hidden_states,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

        pred_proxy = pred_j.detach().to(torch.float32).requires_grad_(True)
        _, _, log_prob_j = flux_step(
            pred_proxy, z_j.to(torch.float32), args.eta,
            sigma_schedule, sigma_idx,
            prev_sample=z_next_j.to(torch.float32),
            grpo=True, sde_solver=True,
        )

        ratio_j = torch.exp(log_prob_j - group_old_log_probs[:, j])
        ratio_list.append(ratio_j.detach())

        unclipped = -advantages_clamped * ratio_j
        clipped = -advantages_clamped * torch.clamp(
            ratio_j, 1.0 - clip_range, 1.0 + clip_range,
        )
        loss_j = torch.mean(torch.maximum(unclipped, clipped)) / loss_norm_factor

        (s_j,) = torch.autograd.grad(loss_j, pred_proxy)
        upstream_list.append(s_j.detach())
        group_loss += loss_j.detach().item()

    # ------------------------------------------------------------------
    # Phase 2 — linear-interpolation aggregation
    #
    #   α_j = j / (W-1)     (0 for first, 1 for last)
    #   s_base = Σ (1 - α_j) · s_j    → upstream for representative  (z_0, t_0)
    #   s_corr = Σ      α_j  · s_j    → upstream for perturbed       (z_{W-1}, t_{W-1})
    #
    # When W=1 this degenerates to a single exact backward.
    # When W=2 s_base=s_0, s_corr=s_1, also exact.
    # ------------------------------------------------------------------
    s_stack = torch.stack(upstream_list, dim=0)  # (W, 1, ...)

    if W > 1:
        alpha = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)
    else:
        alpha = torch.zeros(1, device=device, dtype=torch.float32)

    expand_shape = (W,) + (1,) * (s_stack.ndim - 1)
    alpha = alpha.view(*expand_shape)

    s_base = ((1.0 - alpha) * s_stack).sum(dim=0)  # (1, ...)
    s_corr = (alpha * s_stack).sum(dim=0)           # (1, ...)

    # ------------------------------------------------------------------
    # Phase 3 — two with-grad forward + backward through the transformer
    #
    # Gradient contribution:
    #   G ≈ s_base^T · J(z_0, t_0)  +  s_corr^T · J(z_{W-1}, t_{W-1})
    #
    # Both .backward() calls accumulate into the same param.grad buffers.
    # ------------------------------------------------------------------
    transformer.train()

    # --- base backward at the first (representative) timestep ---
    with torch.autocast("cuda", torch.bfloat16):
        pred_rep = transformer(
            hidden_states=group_latents[:, 0],
            timestep=group_timesteps[:, 0] / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_attention_masks,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=None,
            return_dict=False,
        )[0]
    pred_rep.backward(gradient=s_base.to(pred_rep.dtype))

    # --- correction backward at the last (perturbed) timestep ---
    if W > 1:
        with torch.autocast("cuda", torch.bfloat16):
            pred_pert = transformer(
                hidden_states=group_latents[:, W - 1],
                timestep=group_timesteps[:, W - 1] / 1000,
                guidance=None,
                encoder_hidden_states_mask=prompt_attention_masks,
                encoder_hidden_states=encoder_hidden_states,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=None,
                return_dict=False,
            )[0]
        pred_pert.backward(gradient=s_corr.to(pred_pert.dtype))

    avg_ratio = torch.stack(ratio_list).mean()
    return group_loss, avg_ratio


# ---------------------------------------------------------------------------
# sample_reference_model  (unchanged from base)
# ---------------------------------------------------------------------------

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
    all_rewards = []
    all_txt_seq_lens = []

    if args.init_same_noise:
        input_latents = torch.randn(
            (1, 1, IN_CHANNELS, latent_h, latent_w),
            device=device, dtype=torch.bfloat16,
        ).expand(B, 1, IN_CHANNELS, latent_h, latent_w)
    else:
        input_latents = torch.randn(
            (B, 1, IN_CHANNELS, latent_h, latent_w),
            device=device, dtype=torch.bfloat16,
        )

    packed_height = 2 * (int(h) // (8 * 2))
    packed_width = 2 * (int(w) // (8 * 2))
    input_latents_new = pack_latents(input_latents, B, 16, packed_height, packed_width)

    encoder_hidden_states_processed = []
    prompt_attention_masks_processed = []
    for idx in range(B):
        actual_length = original_length[idx].item() if torch.is_tensor(original_length[idx]) else original_length[idx]
        encoder_hidden_states_processed.append(encoder_hidden_states[idx:idx+1, :actual_length, :])
        prompt_attention_masks_processed.append(prompt_attention_masks[idx:idx+1, :actual_length])

    max_seq_len = max(embed.shape[1] for embed in encoder_hidden_states_processed)

    encoder_hidden_states_padded_list = []
    prompt_attention_masks_padded_list = []
    txt_seq_lens = []
    for idx in range(B):
        embed = encoder_hidden_states_processed[idx]
        mask = prompt_attention_masks_processed[idx]
        current_len = embed.shape[1]
        if current_len < max_seq_len:
            pad_len = max_seq_len - current_len
            embed = torch.nn.functional.pad(embed, (0, 0, 0, pad_len), "constant", 0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), "constant", 0)
        encoder_hidden_states_padded_list.append(embed)
        prompt_attention_masks_padded_list.append(mask)
        txt_seq_lens.append(int(mask.sum().item()))

    encoder_hidden_states_padded = torch.cat(encoder_hidden_states_padded_list, dim=0)
    prompt_attention_masks_padded = torch.cat(prompt_attention_masks_padded_list, dim=0)

    img_shapes = [[(1, h // 8 // 2, w // 8 // 2)]]

    grpo_sample = True
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    progress_bar = tqdm(
        range(0, sample_steps),
        desc="Sampling Progress",
        disable=_local_rank > 0,
        leave=False,
    )

    with torch.no_grad():
        z, latents, all_latents_tensor, all_log_probs_tensor = run_sample_step(
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

    all_latents.append(all_latents_tensor)
    all_log_probs.append(all_log_probs_tensor)
    all_txt_seq_lens.append(torch.tensor(txt_seq_lens))

    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)
    rank = int(os.environ["RANK"])

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

    for idx in range(B):
        decoded_images[idx].save(rollout_image_file(f"qwenimage_{rank}_{idx}.png"))

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
                hps_score = reward_model.reward([rollout_image_file(f"qwenimage_{rank}_{idx}.png")], [caption[idx]])
                if hps_score.ndim == 2:
                    hps_score = hps_score[:, 0]
                all_rewards.append(hps_score)

    if args.use_pickscore:
        def calc_probs(processor, model, prompt, images, device):
            image_inputs = processor(
                images=images, padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            ).to(device)
            text_inputs = processor(
                text=prompt, padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                image_embs = model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                text_embs = model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                scores = (text_embs @ image_embs.T)[0]
            return scores

        for idx in range(B):
            pil_images = [Image.open(rollout_image_file(f"qwenimage_{rank}_{idx}.png"))]
            score = calc_probs(tokenizer, reward_model, caption[idx], pil_images, device)
            all_rewards.append(score)

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_txt_seq_lens = torch.cat(all_txt_seq_lens, dim=0)

    if len(all_rewards) == 0:
        all_rewards = torch.zeros(B, device=device, dtype=torch.float32)
    else:
        all_rewards = torch.cat(all_rewards, dim=0)

    return all_rewards, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


# ---------------------------------------------------------------------------
# train_one_step  (MODIFIED — JAGG backward when sum_sample >= 4)
# ---------------------------------------------------------------------------

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
    ema_handler,
):
    total_loss = 0.0
    grad_norm = torch.tensor(0.0).to(device)

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

    reward, all_latents, all_log_probs, sigma_schedule, all_txt_seq_lens = sample_reference_model(
        args, device, transformer, vae,
        encoder_hidden_states, prompt_attention_masks, original_length,
        reward_model, tokenizer, caption, preprocess_val,
    )

    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps = torch.tensor(timestep_values, device=device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-1][:, :-1],
        "next_latents": all_latents[:, 1:][:, :-1],
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
        "txt_seq_lens": all_txt_seq_lens,
        "encoder_hidden_states": encoder_hidden_states,
        "prompt_attention_masks": prompt_attention_masks,
        "original_length": original_length,
    }

    gathered_reward = gather_tensor(samples["rewards"])
    avg_reward = gathered_reward.mean().item()

    if args.use_group:
        n = len(samples["rewards"]) // args.num_generations
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
        advantages = (samples["rewards"] - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
        samples["advantages"] = advantages

    N = samples["timesteps"].shape[1]  # number of usable training timesteps
    W = args.sum_sample

    # ====================================================================
    # JAGG path: group-based Jacobian-aggregated backward
    # ====================================================================
    if W >= 4:
        assert N % W == 0, (
            f"[JAGG] Training timesteps N={N} must be divisible by sum_sample={W}. "
            f"Adjust --sampling_steps so that (sampling_steps - 1) is a multiple of {W}."
        )
        num_groups = N // W
        train_groups = max(1, int(num_groups * args.timestep_fraction))
        total_train_ts = train_groups * W

        # Group-level shuffle: permute groups, keep intra-group order
        group_perms = torch.stack([
            torch.randperm(num_groups, device=device) for _ in range(batch_size)
        ])

        perms = torch.zeros(batch_size, N, dtype=torch.long, device=device)
        for b in range(batch_size):
            for gi in range(num_groups):
                gp = group_perms[b, gi].item()
                dst_start = gi * W
                src_start = gp * W
                perms[b, dst_start:dst_start + W] = torch.arange(
                    src_start, src_start + W, device=device,
                )

        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size, device=device)[:, None], perms
            ]

        samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
        samples_batched_list = [
            dict(zip(samples_batched, x))
            for x in zip(*samples_batched.values())
        ]

        for i, sample in enumerate(samples_batched_list):
            adv_clamped = torch.clamp(
                sample["advantages"], -args.adv_clip_max, args.adv_clip_max,
            )

            for g in range(train_groups):
                s = g * W
                e = s + W

                group_sigma_indices = [
                    int(perms[i][s + j].item()) for j in range(W)
                ]

                enc_len = sample["original_length"]
                loss_val, avg_ratio = grpo_group_backward(
                    args,
                    sample["latents"][:, s:e],
                    sample["next_latents"][:, s:e],
                    sample["encoder_hidden_states"][:, :enc_len],
                    sample["prompt_attention_masks"][:, :enc_len],
                    sample["txt_seq_lens"],
                    [[(1, args.h // 8 // 2, args.w // 8 // 2)]],
                    transformer,
                    sample["timesteps"][:, s:e],
                    group_sigma_indices,
                    sigma_schedule,
                    sample["log_probs"][:, s:e],
                    adv_clamped,
                    args.clip_range,
                    args.gradient_accumulation_steps * total_train_ts,
                )

                avg_loss_t = torch.tensor(loss_val, device=device)
                dist.all_reduce(avg_loss_t, op=dist.ReduceOp.AVG)
                total_loss += avg_loss_t.item()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = transformer.clip_grad_norm_(max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if dist.get_rank() % 8 == 0:
                print("reward", sample["rewards"].item())
                print("avg_ratio (last group)", avg_ratio.item())
                print("advantage", sample["advantages"].item())
                print("group_loss (last)", loss_val)
            dist.barrier()

    # ====================================================================
    # Fallback path: original per-step backward
    # ====================================================================
    else:
        train_timesteps = int(N * args.timestep_fraction)

        perms = torch.stack([
            torch.randperm(N, device=device) for _ in range(batch_size)
        ])
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size, device=device)[:, None], perms
            ]

        samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
        samples_batched_list = [
            dict(zip(samples_batched, x))
            for x in zip(*samples_batched.values())
        ]

        for i, sample in enumerate(samples_batched_list):
            for _ in range(train_timesteps):
                clip_range = args.clip_range
                adv_clip_max = args.adv_clip_max
                new_log_probs = grpo_one_step(
                    args,
                    sample["latents"][:, _],
                    sample["next_latents"][:, _],
                    sample["encoder_hidden_states"][:, :sample["original_length"]],
                    sample["prompt_attention_masks"][:, :sample["original_length"]],
                    sample["txt_seq_lens"],
                    [[(1, args.h // 8 // 2, args.w // 8 // 2)]],
                    transformer,
                    sample["timesteps"][:, _],
                    perms[i][_],
                    sigma_schedule,
                )

                advantages = torch.clamp(
                    sample["advantages"], -adv_clip_max, adv_clip_max,
                )
                ratio = torch.exp(new_log_probs - sample["log_probs"][:, _])
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio, 1.0 - clip_range, 1.0 + clip_range,
                )
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (
                    args.gradient_accumulation_steps * train_timesteps
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

            if dist.get_rank() % 8 == 0:
                print("reward", sample["rewards"].item())
                print("ratio", ratio)
                print("advantage", sample["advantages"].item())
                print("final loss", loss.item())
            dist.barrier()

    return total_loss, grad_norm.item(), avg_reward


# ---------------------------------------------------------------------------
# main & argparse
# ---------------------------------------------------------------------------

def main(args):
    rollout_dir = os.path.abspath(os.path.expanduser(os.path.normpath(args.rollout_image_dir)))
    os.makedirs(rollout_dir, exist_ok=True)
    os.environ["DANCEGRPO_ROLLOUT_IMAGE_DIR"] = rollout_dir

    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.seed is not None:
        set_seed(args.seed + rank)

    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        print(f"Rollout decode scratch dir (DANCEGRPO_ROLLOUT_IMAGE_DIR): {rollout_dir}", flush=True)
        if args.sum_sample >= 4:
            N_train = args.sampling_steps - 1
            print(
                f"[JAGG] Jacobian-aggregation enabled: sum_sample={args.sum_sample}, "
                f"training_timesteps={N_train}, groups={N_train // args.sum_sample}, "
                f"backward_speedup≈{args.sum_sample / 2:.1f}x per group",
                flush=True,
            )
        else:
            print("[JAGG] Jacobian-aggregation DISABLED (sum_sample < 4), using original per-step backward.", flush=True)

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
                precision='amp', device=device, jit=False,
                force_quick_gelu=False, force_custom_text=False,
                force_patch_dropout=False, force_image_size=None,
                pretrained_image=False, image_mean=None, image_std=None,
                light_augmentation=True, aug_cfg={}, output_dict=True,
                with_score_predictor=False, with_region_predictor=False,
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
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

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenImageTransformerBlock
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )

    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=torch.bfloat16,
        use_device_mesh=False,
    )
    transformer = fsdp_wrapper(transformer, fsdp_config)

    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    apply_fsdp_checkpointing(
        transformer, (QwenImageTransformerBlock), args.selective_checkpointing,
    )

    from diffusers.models.autoencoders import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)

    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    main_print(f"--> model loaded")

    transformer.train()
    noise_scheduler = None

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

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
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed,
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

    if rank <= 0:
        project = "qwenimage"
        wandb.init(project=project, config=args)

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
    main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = "
        f"{sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")

    train_bar_total = args.max_train_steps if args.max_train_steps is not None else 100000
    progress_bar = tqdm(
        range(0, train_bar_total),
        initial=min(init_steps, train_bar_total),
        total=train_bar_total,
        desc="Steps",
        disable=local_rank > 0,
    )

    step_times = deque(maxlen=100)

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for step, (prompt_embeds, prompt_attention_masks, caption, original_length) in enumerate(train_dataloader):
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_masks = prompt_attention_masks.to(device)
            start_time = time.time()
            if (step - 1) % args.checkpointing_steps == 0 and step != 1:
                save_checkpoint(transformer, rank, args.output_dir, step, epoch)
                if args.use_ema:
                    save_ema_checkpoint(ema_handler, rank, args.output_dir, step, epoch, dict(transformer.config))
                dist.barrier()
            if step > (args.max_train_steps + 1):
                break
            loss, grad_norm, avg_reward = train_one_step(
                args, device, transformer, vae,
                reward_model, processor, optimizer, lr_scheduler,
                prompt_embeds, prompt_attention_masks, caption, original_length,
                noise_scheduler, args.max_grad_norm, preprocess_val, ema_handler,
            )

            if args.use_ema and ema_handler:
                ema_handler.update(transformer)

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "reward": f"{avg_reward:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
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
                wandb.log(log_data, step=step)

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
    parser.add_argument("--dataloader_num_workers", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_latent_t", type=int, default=1)
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--rollout_image_dir", type=str,
        default="data/outputs/rollout_scratch_train_grpo_qwenimage_jagg",
    )
    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--precondition_outputs", action="store_true")
    # validation & logs
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    # optimizer & scheduler & Training
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--use_cpu_offload", action="store_true")
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--train_sp_batch_size", type=int, default=1)
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    # lr_scheduler
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--master_weight_type", type=str, default="fp32")
    # GRPO training
    parser.add_argument("--h", type=int, default=None)
    parser.add_argument("--w", type=int, default=None)
    parser.add_argument("--t", type=int, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--sampler_seed", type=int, default=None)
    parser.add_argument("--loss_coef", type=float, default=1.0)
    parser.add_argument("--use_group", action="store_true", default=False)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--use_hpsv2", action="store_true", default=False)
    parser.add_argument("--use_pickscore", action="store_true", default=False)
    parser.add_argument("--use_hpsv3", action="store_true", default=False)
    parser.add_argument("--ignore_last", action="store_true", default=False)
    parser.add_argument("--init_same_noise", action="store_true", default=False)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--timestep_fraction", type=float, default=1.0)
    parser.add_argument("--clip_range", type=float, default=1e-4)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--use_ema", action="store_true")

    # ---- JAGG-specific ----
    parser.add_argument(
        "--sum_sample",
        type=int,
        default=0,
        help="Window size for Jacobian-aggregated backward.  Must be >= 4 to "
        "enable JAGG (at least 3 steps must be interpolated for meaningful "
        "speedup; W=4 gives 2x backward reduction).  Set to 0 to disable "
        "and use the original per-step backward.  (sampling_steps - 1) must "
        "be divisible by this value.",
    )

    args = parser.parse_args()

    if args.sum_sample != 0 and args.sum_sample < 4:
        parser.error("--sum_sample must be 0 (disabled) or >= 4 for meaningful speedup.")

    main(args)
