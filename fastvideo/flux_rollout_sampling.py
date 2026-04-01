"""
Flux rollout sampling with the same mock-guess strategies as Qwen Image training scripts.
Ported from train_grpo_qwenimage_{eff,eff_mta,mta,noise,eff_mta_*} run_sample_step loops;
transformer calls match train_grpo_flux.run_sample_step (FluxTransformer2DModel forward).
"""

from __future__ import annotations

import os

import torch
from diffusers.image_processor import VaeImageProcessor
from tqdm.auto import tqdm

from fastvideo.utils.rollout_image_dir import rollout_image_file
from train_grpo_flux import (
    flux_step,
    pack_latents,
    prepare_flux_txt_ids,
    prepare_latent_image_ids,
    sd3_time_shift,
    unpack_latents,
    run_sample_step as run_sample_step_flux_plain,
)


def flux_denoiser_forward(
    transformer,
    z: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    image_ids: torch.Tensor,
) -> torch.Tensor:
    """Match train_grpo_flux.run_sample_step transformer call."""
    with torch.autocast("cuda", torch.bfloat16):
        pred = transformer(
            hidden_states=z,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps / 1000,
            guidance=torch.tensor([3.5], device=z.device, dtype=torch.bfloat16),
            txt_ids=prepare_flux_txt_ids(text_ids, encoder_hidden_states),
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    return pred


def _velocity_mix_alpha(v_a: torch.Tensor, v_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    va = v_a.reshape(v_a.shape[0], -1)
    vb = v_b.reshape(v_b.shape[0], -1)
    num = (va * vb).sum(dim=-1)
    den = (va.norm(dim=-1) * vb.norm(dim=-1)).clamp(min=eps)
    cos = (num / den).clamp(-1.0, 1.0)
    return 0.5 * (cos + 1.0)


def run_sample_step_flux_mean(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    num_generations = args.num_generations
    num_prompts = B // num_generations

    for i in progress_bar:
        sigma = sigma_schedule[i]
        d_sigma = 0 - sigma
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        can_guess_mask = ~last_step_guessed_mask
        can_guess_indices = torch.where(can_guess_mask)[0]
        num_guess = min(args.num_guess, len(can_guess_indices))
        if i == len(progress_bar) - 1:
            num_guess = 0
        guess_indices_in_can = torch.randperm(len(can_guess_indices), device=z.device)[:num_guess]
        current_guess_indices = can_guess_indices[guess_indices_in_can]
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
            v_mean = torch.zeros_like(z, dtype=torch.float32)
            for p in range(num_prompts):
                start_idx = p * num_generations
                end_idx = (p + 1) * num_generations
                group_infer_mask = infer_mask[start_idx:end_idx]
                if group_infer_mask.any():
                    offset = infer_mask[:start_idx].sum().item()
                    count = group_infer_mask.sum().item()
                    group_v_mean = v_infer[offset : offset + count].mean(dim=0)
                    v_mean[start_idx:end_idx] = group_v_mean
                else:
                    v_mean[start_idx:end_idx] = 0.0
            x_L_mean = z.to(torch.float32) + v_mean * d_sigma
        else:
            x_L_mean = z.to(torch.float32).mean(dim=0, keepdim=True).expand(B, *z.shape[1:])
        if guessed_mask.any():
            v_guess = (x_L_mean[guessed_mask] - z[guessed_mask].to(torch.float32)) / (
                d_sigma if abs(d_sigma) > 1e-6 else 1e-6
            )
            pred[guessed_mask] = v_guess.to(pred.dtype)
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    all_latents = torch.stack(all_latents, dim=1)
    all_log_probs = torch.stack(all_log_probs, dim=1)
    all_mock_flags = torch.stack(all_mock_flags, dim=1)
    return z, latents, all_latents, all_log_probs, all_mock_flags


def run_sample_step_flux_noise(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    num_generations = args.num_generations
    if B % num_generations != 0:
        raise ValueError("B must be divisible by num_generations")
    num_prompts = B // num_generations
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)

    for i in progress_bar:
        sigma = sigma_schedule[i]
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
        current_guess_indices = torch.cat(guess_idx_chunks) if guess_idx_chunks else torch.empty(0, device=z.device, dtype=torch.long)
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        if len(current_guess_indices) > 0:
            guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
        if guessed_mask.any():
            noise_scale = sigma.item() * 0.1
            v_guess = torch.randn_like(z[guessed_mask]) * noise_scale
            pred[guessed_mask] = v_guess.to(pred.dtype)
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1), torch.stack(all_mock_flags, dim=1)


def _per_prompt_guess_indices(can_guess_mask, num_prompts, num_generations, per_prompt_num_guess, cumulative_guess_count, device):
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
        return torch.cat(guess_idx_chunks)
    return torch.empty(0, device=device, dtype=torch.long)


def _group_v_mean_from_infer(z, infer_mask, v_infer, num_prompts, num_generations):
    v_mean = torch.zeros_like(z, dtype=torch.float32)
    for p in range(num_prompts):
        start_idx = p * num_generations
        end_idx = (p + 1) * num_generations
        group_infer_mask = infer_mask[start_idx:end_idx]
        if group_infer_mask.any():
            offset = infer_mask[:start_idx].sum().item()
            count = group_infer_mask.sum().item()
            group_v_mean = v_infer[offset : offset + count].mean(dim=0)
            v_mean[start_idx:end_idx] = group_v_mean
        else:
            v_mean[start_idx:end_idx] = 0.0
    return v_mean


def run_sample_step_flux_reuse(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    num_generations = args.num_generations
    num_prompts = B // num_generations
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
    prev_velocities = torch.zeros_like(z, dtype=torch.float32)

    for i in progress_bar:
        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        can_guess_mask = ~last_step_guessed_mask
        per_prompt_num_guess = 0 if i == len(progress_bar) - 1 else args.num_guess
        current_guess_indices = _per_prompt_guess_indices(
            can_guess_mask, num_prompts, num_generations, per_prompt_num_guess, cumulative_guess_count, z.device
        )
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        if len(current_guess_indices) > 0:
            guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
            prev_velocities[infer_mask] = v_infer.to(torch.float32)
        if guessed_mask.any():
            pred[guessed_mask] = prev_velocities[guessed_mask].to(pred.dtype)
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1), torch.stack(all_mock_flags, dim=1)


def run_sample_step_flux_momentum(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    num_generations = args.num_generations
    num_prompts = B // num_generations
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
    prev_velocities = torch.zeros_like(z, dtype=torch.float32)
    momentum_weight = 0.5

    for i in progress_bar:
        sigma = sigma_schedule[i]
        d_sigma = 0 - sigma
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        can_guess_mask = ~last_step_guessed_mask
        per_prompt_num_guess = 0 if i == len(progress_bar) - 1 else args.num_guess
        current_guess_indices = _per_prompt_guess_indices(
            can_guess_mask, num_prompts, num_generations, per_prompt_num_guess, cumulative_guess_count, z.device
        )
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        if len(current_guess_indices) > 0:
            guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
            prev_velocities[infer_mask] = v_infer.to(torch.float32)
            v_mean = _group_v_mean_from_infer(z, infer_mask, v_infer, num_prompts, num_generations)
            x_L_mean = z.to(torch.float32) + v_mean * d_sigma
        else:
            x_L_mean = z.to(torch.float32).mean(dim=0, keepdim=True).expand(B, *z.shape[1:])
        if guessed_mask.any():
            v_guess = (x_L_mean[guessed_mask] - z[guessed_mask].to(torch.float32)) / (d_sigma if abs(d_sigma) > 1e-6 else 1e-6)
            v_guess_with_momentum = (1 - momentum_weight) * v_guess + momentum_weight * prev_velocities[guessed_mask]
            pred[guessed_mask] = v_guess_with_momentum.to(pred.dtype)
            prev_velocities[guessed_mask] = v_guess_with_momentum
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1), torch.stack(all_mock_flags, dim=1)


def run_sample_step_flux_auto(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    num_generations = args.num_generations
    num_prompts = B // num_generations
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
    prev_velocities = torch.zeros_like(z, dtype=torch.float32)

    for i in progress_bar:
        sigma = sigma_schedule[i]
        if i < len(progress_bar) * 0.5:
            ratio = (sigma / sigma_schedule[0]) * 2
        else:
            ratio = 1
        momentum_weight = 0.7 + 0.3 * ratio
        d_sigma = 0 - sigma
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        can_guess_mask = ~last_step_guessed_mask
        per_prompt_num_guess = 0 if i == args.sampling_steps - 1 else args.num_guess
        current_guess_indices = _per_prompt_guess_indices(
            can_guess_mask, num_prompts, num_generations, per_prompt_num_guess, cumulative_guess_count, z.device
        )
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        if len(current_guess_indices) > 0:
            guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
            prev_velocities[infer_mask] = v_infer.to(torch.float32)
            v_mean = _group_v_mean_from_infer(z, infer_mask, v_infer, num_prompts, num_generations)
            x_L_mean = z.to(torch.float32) + v_mean * d_sigma
        else:
            x_L_mean = z.to(torch.float32).mean(dim=0, keepdim=True).expand(B, *z.shape[1:])
        if guessed_mask.any():
            v_guess = (x_L_mean[guessed_mask] - z[guessed_mask].to(torch.float32)) / (d_sigma if abs(d_sigma) > 1e-6 else 1e-6)
            v_guess_with_momentum = (1 - momentum_weight) * v_guess + momentum_weight * prev_velocities[guessed_mask]
            pred[guessed_mask] = v_guess_with_momentum.to(pred.dtype)
            prev_velocities[guessed_mask] = v_guess_with_momentum
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1), torch.stack(all_mock_flags, dim=1)


def run_sample_step_flux_d2(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    num_generations = args.num_generations
    num_prompts = B // num_generations
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
    prev_velocities = torch.zeros_like(z, dtype=torch.float32)
    z_history = []
    sigma_history = []
    acceleration_weight_base = 0.05

    for i in progress_bar:
        sigma = sigma_schedule[i]
        if i < len(progress_bar) * 0.5:
            ratio = (sigma / sigma_schedule[0]) * 2
        else:
            ratio = 1
        momentum_weight = 0.6 + 0.4 * ratio
        acceleration_weight = acceleration_weight_base / max(momentum_weight, 1.0)
        early_guess_phase = i < len(progress_bar) * 0.5
        d_sigma = 0 - sigma
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        can_guess_mask = ~last_step_guessed_mask
        per_prompt_num_guess = 0 if i == len(progress_bar) - 1 else args.num_guess
        current_guess_indices = _per_prompt_guess_indices(
            can_guess_mask, num_prompts, num_generations, per_prompt_num_guess, cumulative_guess_count, z.device
        )
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        if len(current_guess_indices) > 0:
            guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
            prev_velocities[infer_mask] = v_infer.to(torch.float32)
            v_mean = _group_v_mean_from_infer(z, infer_mask, v_infer, num_prompts, num_generations)
            x_L_mean = z.to(torch.float32) + v_mean * d_sigma
        else:
            x_L_mean = z.to(torch.float32).mean(dim=0, keepdim=True).expand(B, *z.shape[1:])
        if guessed_mask.any():
            v_guess = (x_L_mean[guessed_mask] - z[guessed_mask].to(torch.float32)) / (d_sigma if abs(d_sigma) > 1e-6 else 1e-6)
            if not early_guess_phase:
                v_reuse = prev_velocities[guessed_mask]
                pred[guessed_mask] = v_reuse.to(pred.dtype)
                prev_velocities[guessed_mask] = v_reuse
            else:
                if len(z_history) >= 2:
                    prev_z_1 = z_history[-1]
                    prev_z_2 = z_history[-2]
                    sigma_1 = sigma_history[-1]
                    sigma_2 = sigma_history[-2]
                    dt_1 = sigma_2 - sigma_1
                    dt_2 = sigma_1 - sigma
                    velocity_1 = (prev_z_1 - prev_z_2) / (dt_1 if abs(dt_1) > 1e-6 else 1e-6)
                    velocity_2 = (z.to(torch.float32) - prev_z_1) / (dt_2 if abs(dt_2) > 1e-6 else 1e-6)
                    avg_dt = (abs(dt_1) + abs(dt_2)) / 2
                    acceleration = (velocity_2 - velocity_1) / (avg_dt if avg_dt > 1e-6 else 1e-6)
                    v_so = (
                        v_guess
                        + momentum_weight * velocity_2[guessed_mask]
                        + 0.5 * acceleration_weight * acceleration[guessed_mask]
                    )
                    v_out = (1 - momentum_weight) * v_so + momentum_weight * prev_velocities[guessed_mask]
                    pred[guessed_mask] = v_out.to(pred.dtype)
                    prev_velocities[guessed_mask] = v_out
                else:
                    v_guess_with_momentum = (1 - momentum_weight) * v_guess + momentum_weight * prev_velocities[guessed_mask]
                    pred[guessed_mask] = v_guess_with_momentum.to(pred.dtype)
                    prev_velocities[guessed_mask] = v_guess_with_momentum
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        z_history.append(z.clone().to(torch.float32))
        sigma_history.append(sigma)
        if len(z_history) > 3:
            z_history.pop(0)
            sigma_history.pop(0)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1), torch.stack(all_mock_flags, dim=1)


def run_sample_step_flux_alpha(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    num_generations = args.num_generations
    num_prompts = B // num_generations
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
    prev_velocities = torch.zeros_like(z, dtype=torch.float32)

    for i in progress_bar:
        sigma = sigma_schedule[i]
        d_sigma = 0 - sigma
        old_prev = prev_velocities.clone()
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        can_guess_mask = ~last_step_guessed_mask
        per_prompt_num_guess = 0 if i == args.sampling_steps - 1 else args.num_guess
        current_guess_indices = _per_prompt_guess_indices(
            can_guess_mask, num_prompts, num_generations, per_prompt_num_guess, cumulative_guess_count, z.device
        )
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        if len(current_guess_indices) > 0:
            guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        v_mean = torch.zeros_like(z, dtype=torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
            v_mean = _group_v_mean_from_infer(z, infer_mask, v_infer, num_prompts, num_generations)
            x_L_mean = z.to(torch.float32) + v_mean * d_sigma
        else:
            x_L_mean = z.to(torch.float32).mean(dim=0, keepdim=True).expand(B, *z.shape[1:])
        early_group_fusion = i < len(progress_bar) * 0.5
        if guessed_mask.any():
            v_prev = old_prev[guessed_mask]
            if early_group_fusion:
                if infer_mask.any():
                    v_group = v_mean[guessed_mask]
                else:
                    ds = d_sigma if abs(d_sigma) > 1e-6 else 1e-6
                    v_group = (x_L_mean[guessed_mask] - z[guessed_mask].to(torch.float32)) / ds
                alpha = _velocity_mix_alpha(v_prev, v_group)
                alpha = alpha.view(-1, *([1] * (v_prev.ndim - 1)))
                v_hat = alpha * v_prev + (1.0 - alpha) * v_group
            else:
                v_hat = v_prev
            pred[guessed_mask] = v_hat.to(pred.dtype)
        if infer_mask.any():
            prev_velocities[infer_mask] = pred[infer_mask].to(torch.float32)
        if guessed_mask.any():
            prev_velocities[guessed_mask] = pred[guessed_mask].to(torch.float32)
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1), torch.stack(all_mock_flags, dim=1)


def run_sample_step_flux_ab2(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    num_generations = args.num_generations
    num_prompts = B // num_generations
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
    prev_velocities = torch.zeros_like(z, dtype=torch.float32)
    prev_prev_velocities = torch.zeros_like(z, dtype=torch.float32)

    for i in progress_bar:
        sigma = sigma_schedule[i]
        d_sigma = 0 - sigma
        old_prev = prev_velocities.clone()
        old_pprev = prev_prev_velocities.clone()
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        can_guess_mask = ~last_step_guessed_mask
        per_prompt_num_guess = 0 if i == args.sampling_steps - 1 else args.num_guess
        current_guess_indices = _per_prompt_guess_indices(
            can_guess_mask, num_prompts, num_generations, per_prompt_num_guess, cumulative_guess_count, z.device
        )
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        if len(current_guess_indices) > 0:
            guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        v_mean = torch.zeros_like(z, dtype=torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
            v_mean = _group_v_mean_from_infer(z, infer_mask, v_infer, num_prompts, num_generations)
            x_L_mean = z.to(torch.float32) + v_mean * d_sigma
        else:
            x_L_mean = z.to(torch.float32).mean(dim=0, keepdim=True).expand(B, *z.shape[1:])
        early_group_fusion = i < len(progress_bar) * 0.5
        if guessed_mask.any():
            v_prev = old_prev[guessed_mask]
            v_pp = old_pprev[guessed_mask]
            v_so = 1.5 * v_prev - 0.5 * v_pp if i >= 2 else v_prev
            if early_group_fusion:
                if infer_mask.any():
                    v_group = v_mean[guessed_mask]
                else:
                    ds = d_sigma if abs(d_sigma) > 1e-6 else 1e-6
                    v_group = (x_L_mean[guessed_mask] - z[guessed_mask].to(torch.float32)) / ds
                alpha = _velocity_mix_alpha(v_so, v_group)
                alpha = alpha.view(-1, *([1] * (v_so.ndim - 1)))
                v_hat = alpha * v_so + (1.0 - alpha) * v_group
            else:
                v_hat = v_so
            pred[guessed_mask] = v_hat.to(pred.dtype)
        prev_prev_velocities.copy_(old_prev)
        if infer_mask.any():
            prev_velocities[infer_mask] = pred[infer_mask].to(torch.float32)
        if guessed_mask.any():
            prev_velocities[guessed_mask] = pred[guessed_mask].to(torch.float32)
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1), torch.stack(all_mock_flags, dim=1)


def run_sample_step_flux_varguess(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    grpo_sample,
):
    if not grpo_sample:
        raise NotImplementedError
    all_latents = [z]
    all_log_probs = []
    all_mock_flags = []
    B = z.shape[0]
    num_generations = args.num_generations
    num_prompts = B // num_generations
    last_step_guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=z.device, dtype=torch.long)
    prev_velocities = torch.zeros_like(z, dtype=torch.float32)
    last_pred_v = torch.zeros_like(z, dtype=torch.float32)

    for i in progress_bar:
        sigma = sigma_schedule[i]
        if i < len(progress_bar) * 0.5:
            ratio = (sigma / sigma_schedule[0]) * 2
        else:
            ratio = 1
        momentum_weight = 0.7 + 0.3 * ratio
        d_sigma = 0 - sigma
        old_prev = prev_velocities.clone()
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        can_guess_mask = ~last_step_guessed_mask
        cap_last = i == args.sampling_steps - 1
        num_guess_max = args.num_guess
        num_guess_min = getattr(args, "num_guess_min", 0)
        var_tau = getattr(args, "var_guess_tau", 1e-4)
        guess_idx_chunks = []
        for p in range(num_prompts):
            start_idx = p * num_generations
            end_idx = (p + 1) * num_generations
            local_can = torch.where(can_guess_mask[start_idx:end_idx])[0] + start_idx
            if cap_last or int(local_can.numel()) == 0:
                ng = 0
            else:
                gstack = last_pred_v[start_idx:end_idx].reshape(num_generations, -1)
                var_g = gstack.var(dim=0, unbiased=False).mean()
                weight = (1.0 / (1.0 + var_g / var_tau)).item()
                ng = num_guess_min + (num_guess_max - num_guess_min) * weight
                ng = int(round(ng))
                ng = max(num_guess_min, min(ng, num_guess_max, int(local_can.numel())))
            if ng > 0:
                counts = cumulative_guess_count[local_can]
                order = torch.argsort(counts)[:ng]
                picked = local_can[order]
                cumulative_guess_count[picked] += 1
                guess_idx_chunks.append(picked)
        current_guess_indices = torch.cat(guess_idx_chunks) if guess_idx_chunks else torch.empty(0, device=z.device, dtype=torch.long)
        guessed_mask = torch.zeros(B, device=z.device, dtype=torch.bool)
        if len(current_guess_indices) > 0:
            guessed_mask[current_guess_indices] = True
        infer_mask = ~guessed_mask
        pred = torch.zeros_like(z)
        mock_flag = guessed_mask.clone().to(torch.float32)
        if infer_mask.any():
            transformer.eval()
            v_infer = flux_denoiser_forward(
                transformer,
                z[infer_mask],
                timesteps[infer_mask],
                encoder_hidden_states[infer_mask],
                pooled_prompt_embeds[infer_mask],
                text_ids[infer_mask],
                image_ids,
            )
            pred[infer_mask] = v_infer.to(pred.dtype)
            prev_velocities[infer_mask] = v_infer.to(torch.float32)
            v_mean = _group_v_mean_from_infer(z, infer_mask, v_infer, num_prompts, num_generations)
            x_L_mean = z.to(torch.float32) + v_mean * d_sigma
        else:
            x_L_mean = z.to(torch.float32).mean(dim=0, keepdim=True).expand(B, *z.shape[1:])
        early_group_fusion = i < len(progress_bar) * 0.5
        if guessed_mask.any():
            if early_group_fusion:
                v_guess = (x_L_mean[guessed_mask] - z[guessed_mask].to(torch.float32)) / (
                    d_sigma if abs(d_sigma) > 1e-6 else 1e-6
                )
                v_guess_with_momentum = (1 - momentum_weight) * v_guess + momentum_weight * prev_velocities[guessed_mask]
                pred[guessed_mask] = v_guess_with_momentum.to(pred.dtype)
                prev_velocities[guessed_mask] = v_guess_with_momentum
            else:
                v_reuse = old_prev[guessed_mask]
                pred[guessed_mask] = v_reuse.to(pred.dtype)
                prev_velocities[guessed_mask] = v_reuse
        last_pred_v.copy_(pred.to(torch.float32))
        z, pred_original, log_prob = flux_step(
            pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True
        )
        z = z.to(torch.bfloat16)
        all_latents.append(z)
        all_log_probs.append(log_prob)
        all_mock_flags.append(mock_flag)
        last_step_guessed_mask = guessed_mask
    latents = pred_original
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1), torch.stack(all_mock_flags, dim=1)


def _flux_decode_save_hps(
    args,
    device,
    vae,
    latents_final,
    all_mock_flags_tensor,
    caption,
    reward_model,
    tokenizer,
    preprocess_val,
    rank: int,
    file_prefix: str,
    *,
    guess_in_filename: bool = True,
):
    """Decode final latents for each rollout, save PNGs, return HPS scores stacked."""
    w, h = args.w, args.h
    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)
    num_guess_steps = all_mock_flags_tensor.sum(dim=1).cpu().numpy()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents_u = unpack_latents(latents_final, h, w, 8)
            latents_u = (latents_u / 0.3611) + 0.1159
            image = vae.decode(latents_u, return_dict=False)[0]
            decoded_images = image_processor.postprocess(image)
    B = latents_final.shape[0]
    all_rewards = []
    for idx in range(B):
        guess_count = int(num_guess_steps[idx])
        if guess_in_filename:
            path = rollout_image_file(f"{file_prefix}_{rank}_{idx}_guess{guess_count}.png")
        else:
            path = rollout_image_file(f"{file_prefix}_{rank}_{idx}.png")
        decoded_images[idx].save(path)
        if args.use_hpsv2 and reward_model is not None:
            with torch.no_grad():
                im = preprocess_val(decoded_images[idx]).unsqueeze(0).to(device=device, non_blocking=True)
                text = tokenizer([caption[idx]]).to(device=device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    outputs = reward_model(im, text)
                    logits = outputs["image_features"] @ outputs["text_features"].T
                    hps_score = torch.diagonal(logits)
                all_rewards.append(hps_score)
    if not all_rewards:
        return torch.zeros(B, device=device, dtype=torch.float32)
    return torch.cat(all_rewards, dim=0)


def sample_reference_model_flux_plain(
    args,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
):
    """No mock-guess: same stepping as train_grpo_flux.run_sample_step, full batch B."""
    w, h = args.w, args.h
    sample_steps = args.sampling_steps
    sigma_schedule = sd3_time_shift(args.shift, torch.linspace(1, 0, sample_steps + 1, device=device))
    B = encoder_hidden_states.shape[0]
    IN_CHANNELS = 16
    latent_w, latent_h = w // 8, h // 8
    if args.init_same_noise:
        input_latents = torch.randn((1, IN_CHANNELS, latent_h, latent_w), device=device, dtype=torch.bfloat16).expand(
            B, -1, -1, -1
        )
    else:
        input_latents = torch.randn((B, IN_CHANNELS, latent_h, latent_w), device=device, dtype=torch.bfloat16)
    z0 = pack_latents(input_latents, B, IN_CHANNELS, latent_h, latent_w)
    image_ids = prepare_latent_image_ids(B, latent_h // 2, latent_w // 2, device, torch.bfloat16)
    rank = int(os.environ.get("RANK", "0"))
    with torch.no_grad():
        z, latents, all_latents, all_log_probs = run_sample_step_flux_plain(
            args,
            z0,
            tqdm(range(sample_steps), desc="Flux original", leave=False),
            sigma_schedule,
            transformer,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            image_ids,
            True,
        )
    mock_zeros = torch.zeros(B, sample_steps, device=device, dtype=torch.float32)
    rewards = _flux_decode_save_hps(
        args,
        device,
        vae,
        latents,
        mock_zeros,
        caption,
        reward_model,
        tokenizer,
        preprocess_val,
        rank,
        "flux",
        guess_in_filename=False,
    )
    return rewards, all_latents, all_log_probs, sigma_schedule, torch.zeros(B, dtype=torch.long, device=device)


def sample_reference_model_flux_mock(
    run_step_fn,
    args,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
    *,
    tag: str = "mock",
):
    w, h = args.w, args.h
    sample_steps = args.sampling_steps
    sigma_schedule = sd3_time_shift(args.shift, torch.linspace(1, 0, sample_steps + 1, device=device))
    B = encoder_hidden_states.shape[0]
    IN_CHANNELS = 16
    latent_w, latent_h = w // 8, h // 8
    if args.init_same_noise:
        input_latents = torch.randn((1, IN_CHANNELS, latent_h, latent_w), device=device, dtype=torch.bfloat16).expand(
            B, -1, -1, -1
        )
    else:
        input_latents = torch.randn((B, IN_CHANNELS, latent_h, latent_w), device=device, dtype=torch.bfloat16)
    z0 = pack_latents(input_latents, B, IN_CHANNELS, latent_h, latent_w)
    image_ids = prepare_latent_image_ids(B, latent_h // 2, latent_w // 2, device, torch.bfloat16)
    rank = int(os.environ.get("RANK", "0"))
    with torch.no_grad():
        z, latents, all_latents, all_log_probs, all_mock_flags = run_step_fn(
            args,
            z0,
            tqdm(range(sample_steps), desc=f"Flux {tag}", leave=False),
            sigma_schedule,
            transformer,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            image_ids,
            True,
        )
    rewards = _flux_decode_save_hps(
        args, device, vae, latents, all_mock_flags, caption, reward_model, tokenizer, preprocess_val, rank, "flux"
    )
    txt_seq = torch.zeros(B, dtype=torch.long, device=device)
    return rewards, all_latents, all_log_probs, sigma_schedule, txt_seq, all_mock_flags


def sample_reference_model_flux_mean(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_mean, *a, tag="mean", **k)


def sample_reference_model_flux_noise(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_noise, *a, tag="noise", **k)


def sample_reference_model_flux_reuse(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_reuse, *a, tag="reuse", **k)


def sample_reference_model_flux_momentum(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_momentum, *a, tag="momentum", **k)


def sample_reference_model_flux_auto(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_auto, *a, tag="auto", **k)


def sample_reference_model_flux_d2(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_d2, *a, tag="d2", **k)


def sample_reference_model_flux_alpha(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_alpha, *a, tag="alpha", **k)


def sample_reference_model_flux_ab2(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_ab2, *a, tag="ab2", **k)


def sample_reference_model_flux_varguess(*a, **k):
    return sample_reference_model_flux_mock(run_sample_step_flux_varguess, *a, tag="varguess", **k)
