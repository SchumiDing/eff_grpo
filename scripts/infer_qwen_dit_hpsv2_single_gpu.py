#!/usr/bin/env python3
# Copyright: DanceGRPO / user script — Qwen-Image (diffusers) inference with a DiT-only checkpoint
# folder, precomputed text embeddings, and optional HPSv2 evaluation.
#
# Denoising follows the official QwenImagePipeline (FlowMatchEulerDiscreteScheduler + transformer
# forward + scheduler.step). This avoids FastVideo GRPO rollout (stochastic flux_step + scratch PNGs).
#
# Single GPU (from repo root):
#   export PYTHONPATH=/path/to/DanceGRPO
#   python scripts/infer_qwen_dit_hpsv2_single_gpu.py \
#     --dit_checkpoint data/outputs/grpo_eff_auto_mta/checkpoint-26-0 \
#     --base_model data/qwenimage \
#     --embeddings_path data/qwenimage/rl_embeddings_drawbench \
#     --output_dir outputs/drawbench_ckpt26_eval
#
# Four GPUs + FSDP (DiT sharded; VAE decode / IO on rank 0 only; VAE tiling off):
#   torchrun --standalone --nproc_per_node=4 scripts/infer_qwen_dit_hpsv2_single_gpu.py \
#     --dit_checkpoint ... --base_model ... --embeddings_path ... --output_dir ...
#
# Optional: add HPSv2 to PYTHONPATH or `pip install -e /path/to/HPSv2`.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_distributed() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). world_size==1 means no distributed run."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    return rank, ws, local_rank


def _ensure_paths() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen-Image: DiT checkpoint + RL embeddings, official scheduler denoise, optional HPSv2; "
        "multi-GPU via torchrun + FSDP on DiT."
    )
    p.add_argument(
        "--dit_checkpoint",
        type=str,
        required=True,
        help="Folder with config.json + diffusion_pytorch_model.safetensors (DiT only).",
    )
    p.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Full Qwen-Image diffusers tree (VAE + scheduler). Default: <repo>/data/qwenimage",
    )
    p.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="RL embedding dir with videos2caption.json, prompt_embed/, prompt_attention_mask/.",
    )
    p.add_argument("--output_dir", type=str, required=True, help="Images + metrics JSONL.")
    p.add_argument("--batch_size", type=int, default=1, help="Prompts per forward (VRAM bound).")
    p.add_argument("--sampling_steps", type=int, default=20)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None, help="Exclusive global end index (default: all).")
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Single-GPU only. Under torchrun, each rank uses cuda:LOCAL_RANK (this flag is ignored).",
    )
    p.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="FULL_SHARD",
        choices=("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"),
        help="FSDP sharding for DiT when WORLD_SIZE>1 (torchrun).",
    )
    p.add_argument(
        "--max_sequence_length",
        type=int,
        default=1024,
        help="Truncate prompt embeds (must match RL preprocessing; pipeline default caps at 1024).",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Only for guidance-distilled checkpoints (transformer.config.guidance_embeds). Else ignored.",
    )
    p.add_argument(
        "--hps_version",
        type=str,
        default="v2.1",
        choices=("v2.0", "v2.1"),
        help="HPSv2 checkpoint tag when using HF download.",
    )
    p.add_argument(
        "--hps_checkpoint",
        type=str,
        default=None,
        help="Path to HPS_v2*.pt (e.g. hps_ckpt/HPS_v2.1_compressed.pt). Default: download via huggingface_hub.",
    )
    p.add_argument(
        "--hps_open_clip",
        type=str,
        default=None,
        help="Optional path to open_clip_pytorch_model.bin; default: <repo>/hps_ckpt/open_clip_pytorch_model.bin if present.",
    )
    p.add_argument(
        "--no_hps",
        action="store_true",
        help="Only save images, skip HPSv2 (faster, no hpsv2 install needed).",
    )
    p.add_argument(
        "--attn",
        type=str,
        default="auto",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        help="Transformer attention implementation.",
    )
    return p.parse_args()


def _load_transformer(dit_checkpoint: str, attn: str):
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    kwargs = {"torch_dtype": torch.bfloat16}
    if attn == "auto":
        try:
            return QwenImageTransformer2DModel.from_pretrained(
                dit_checkpoint, attn_implementation="flash_attention_2", **kwargs
            )
        except Exception:
            return QwenImageTransformer2DModel.from_pretrained(
                dit_checkpoint, attn_implementation="sdpa", **kwargs
            )
    return QwenImageTransformer2DModel.from_pretrained(
        dit_checkpoint, attn_implementation=attn, **kwargs
    )


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _init_hps(
    device: str,
    hps_version: str,
    hps_checkpoint: str | None,
    open_clip_checkpoint: str | None,
):
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import hps_version_map

    oc = open_clip_checkpoint if (open_clip_checkpoint and os.path.isfile(open_clip_checkpoint)) else None

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        oc,
        precision="amp",
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
        with_region_predictor=False,
    )
    cp = hps_checkpoint
    if cp is None:
        import huggingface_hub

        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    ckpt = _safe_torch_load(cp)
    model.load_state_dict(ckpt["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()
    return model, preprocess_val, tokenizer


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
    return latents


def _prepare_prompt_batch(
    encoder_hidden_states: torch.Tensor,
    prompt_attention_masks: torch.Tensor,
    original_length: torch.Tensor,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Slice to true length, pad batch, align with QwenImagePipeline.encode_prompt.

    Some diffusers builds require explicit ``txt_seq_lens`` for RoPE (``pos_embed``); we always
    return per-sample valid lengths from the mask (same convention as ``train_grpo_qwenimage``).
    """
    b = encoder_hidden_states.shape[0]
    processed_e = []
    processed_m = []
    for idx in range(b):
        actual_length = int(original_length[idx].item())
        processed_e.append(encoder_hidden_states[idx : idx + 1, :actual_length, :])
        processed_m.append(prompt_attention_masks[idx : idx + 1, :actual_length])
    max_seq_len = max(x.shape[1] for x in processed_e)
    padded_e = []
    padded_m = []
    for idx in range(b):
        embed = processed_e[idx]
        mask = processed_m[idx]
        if embed.shape[1] < max_seq_len:
            pad_len = max_seq_len - embed.shape[1]
            embed = torch.nn.functional.pad(embed, (0, 0, 0, pad_len), "constant", 0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), "constant", 0)
        padded_e.append(embed)
        padded_m.append(mask)
    pe = torch.cat(padded_e, dim=0)
    pm = torch.cat(padded_m, dim=0)
    pe = pe[:, :max_sequence_length]
    pm = pm[:, :max_sequence_length].float()
    txt_seq_lens = [int(pm[i].sum().item()) for i in range(b)]
    pe = pe.to(dtype=torch.bfloat16)
    return pe, pm, txt_seq_lens


def _qwen_official_denoise(
    *,
    transformer,
    scheduler,
    vae,
    image_processor,
    vae_scale_factor: int,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    txt_seq_lens: list[int],
    height: int,
    width: int,
    num_inference_steps: int,
    device: str,
    generator: torch.Generator | None,
    guidance_scale: float | None,
    rank: int = 0,
    decode_rank: int = 0,
    noise_seed: int | None = None,
) -> list | None:
    """Match diffusers QwenImagePipeline.__call__ denoising + VAE decode (batch)."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps
    from diffusers.utils.torch_utils import randn_tensor

    num_channels_latents = transformer.config.in_channels // 4
    h_in = 2 * (int(height) // (vae_scale_factor * 2))
    w_in = 2 * (int(width) // (vae_scale_factor * 2))
    batch_size = prompt_embeds.shape[0]
    shape = (batch_size, 1, num_channels_latents, h_in, w_in)
    if noise_seed is not None:
        gen_cpu = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        latents = randn_tensor(shape, generator=gen_cpu, device="cpu", dtype=prompt_embeds.dtype)
        latents = latents.to(device)
    else:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
    latents = _pack_latents(latents, batch_size, num_channels_latents, h_in, w_in)

    triple = (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2)
    img_shapes = [[triple] for _ in range(batch_size)]

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )

    if getattr(transformer.config, "guidance_embeds", False):
        if guidance_scale is None:
            raise ValueError("This checkpoint expects --guidance_scale (guidance-distilled).")
        g = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)
    else:
        g = None

    scheduler.set_begin_index(0)
    
    with torch.no_grad():

        for t in tqdm(timesteps, total=len(timesteps), desc="Sampling"):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            # Do not use transformer.cache_context() here: some diffusers/torch combos break
            # ``with cache_context(...)`` (not a stdlib-compatible context manager).
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=g,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        if rank != decode_rank:
            return None

        assert vae is not None and image_processor is not None
        latents = _unpack_latents(latents, height, width, vae_scale_factor)
        latents = latents.to(vae.dtype)
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            images = vae.decode(latents, return_dict=False)[0][:, :, 0]
        decoded = image_processor.postprocess(images, output_type="pil")
    return decoded


def _hps_scores_batch(
    reward_model,
    preprocess_val,
    tokenizer,
    device: str,
    pil_images: list,
    captions: list[str],
) -> torch.Tensor:
    scores = []
    with torch.no_grad():
        for pil, cap in zip(pil_images, captions):
            image = preprocess_val(pil).unsqueeze(0).to(device=device, non_blocking=True)
            text = tokenizer([cap]).to(device=device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = reward_model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image)
            scores.append(hps_score.float())
    return torch.cat(scores, dim=0)


def main() -> None:
    args = _parse_args()
    _ensure_paths()

    if args.base_model is None:
        args.base_model = str(_repo_root() / "data" / "qwenimage")

    dit_ckpt = os.path.abspath(os.path.expanduser(args.dit_checkpoint))
    base_model = os.path.abspath(os.path.expanduser(args.base_model))
    emb_root = os.path.abspath(os.path.expanduser(args.embeddings_path))
    out_dir = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    rank, world_size, local_rank = _setup_distributed()
    is_main = rank == 0
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    torch.manual_seed(args.seed)

    hps_root = os.environ.get("HPSV2_ROOT", str(_repo_root().parent / "HPSv2"))
    if os.path.isdir(hps_root) and hps_root not in sys.path:
        sys.path.insert(0, hps_root)

    from diffusers.image_processor import VaeImageProcessor
    from diffusers.models.autoencoders import AutoencoderKLQwenImage
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from fastvideo.utils.fsdp_util_qwenimage import FSDPConfig, fsdp_wrapper

    if world_size > 1:
        device = f"cuda:{local_rank}"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if is_main:
        print(f"Loading DiT from {dit_ckpt} ...")
    transformer = _load_transformer(dit_ckpt, args.attn).to(device)
    transformer.eval()
    if world_size > 1:
        fsdp_config = FSDPConfig(
            sharding_strategy=args.fsdp_sharding_strategy,
            backward_prefetch="BACKWARD_PRE",
            cpu_offload=False,
            mixed_precision_dtype=torch.bfloat16,
            use_device_mesh=False,
        )
        if is_main:
            print(f"Wrapping DiT with FSDP ({args.fsdp_sharding_strategy}, world_size={world_size}) ...")
        transformer = fsdp_wrapper(transformer, fsdp_config)

    vae = None
    image_processor = None
    vae_sf = 8
    if is_main:
        print(f"Loading VAE + scheduler from {base_model} ...")
        vae = AutoencoderKLQwenImage.from_pretrained(
            base_model, subfolder="vae", torch_dtype=torch.bfloat16
        ).to(device)
        if hasattr(vae, "disable_tiling"):
            vae.disable_tiling()
        vae_sf = 2 ** len(vae.temperal_downsample) if getattr(vae, "temperal_downsample", None) is not None else 8
        image_processor = VaeImageProcessor(vae_scale_factor=vae_sf * 2)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler")
    if world_size > 1:
        vae_sf_buf = torch.tensor([vae_sf], device=device, dtype=torch.long)
        dist.broadcast(vae_sf_buf, src=0)
        vae_sf = int(vae_sf_buf.item())

    reward_model = None
    preprocess_val = None
    tokenizer = None
    if is_main and not args.no_hps:
        print("Loading HPSv2 ...")
        oc_path = args.hps_open_clip
        if oc_path is None:
            _oc = _repo_root() / "hps_ckpt" / "open_clip_pytorch_model.bin"
            oc_path = str(_oc) if _oc.is_file() else None
        reward_model, preprocess_val, tokenizer = _init_hps(
            device, args.hps_version, args.hps_checkpoint, oc_path
        )

    json_path = os.path.join(emb_root, "videos2caption.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data_anno = json.load(f)

    end = args.end_idx if args.end_idx is not None else len(data_anno)
    data_anno = data_anno[args.start_idx : end]
    n = len(data_anno)
    if is_main:
        print(f"Prompts in this run: {n} (indices {args.start_idx}..{end - 1})")

    prompt_embed_dir = os.path.join(emb_root, "prompt_embed")
    mask_dir = os.path.join(emb_root, "prompt_attention_mask")

    metrics_path = out_dir / "hps_metrics.jsonl"
    agg_scores: list[float] = []

    bs = max(1, args.batch_size)
    global_offset = args.start_idx

    for start in range(0, n, bs):
        batch = data_anno[start : start + bs]
        pe_list = []
        pm_list = []
        caps = []
        orig_lens = []
        for item in batch:
            pe = _safe_torch_load(os.path.join(prompt_embed_dir, item["prompt_embed_path"]))
            pm = _safe_torch_load(os.path.join(mask_dir, item["prompt_attention_mask"]))
            pe_list.append(pe)
            pm_list.append(pm)
            caps.append(item["caption"])
            orig_lens.append(int(item["original_length"]))

        prompt_embeds = torch.stack(pe_list, dim=0).to(device)
        prompt_attention_masks = torch.stack(pm_list, dim=0).to(device)
        original_length = torch.tensor(orig_lens, device=device, dtype=torch.long)

        pe, pm, txt_seq_lens = _prepare_prompt_batch(
            prompt_embeds,
            prompt_attention_masks,
            original_length,
            args.max_sequence_length,
        )

        gen_device = device if device.startswith("cuda") else "cpu"
        gen = torch.Generator(device=gen_device).manual_seed(int(args.seed + global_offset + start))
        noise_seed = int(args.seed + global_offset + start) if world_size > 1 else None

        pil_images = _qwen_official_denoise(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            image_processor=image_processor,
            vae_scale_factor=vae_sf,
            prompt_embeds=pe,
            prompt_embeds_mask=pm,
            txt_seq_lens=txt_seq_lens,
            height=args.height,
            width=args.width,
            num_inference_steps=args.sampling_steps,
            device=device,
            generator=gen,
            guidance_scale=args.guidance_scale,
            rank=rank,
            decode_rank=0,
            noise_seed=noise_seed,
        )

        if is_main and pil_images is not None:
            if not args.no_hps:
                rewards_cpu = _hps_scores_batch(
                    reward_model, preprocess_val, tokenizer, device, pil_images, caps
                ).cpu()
            else:
                rewards_cpu = None

            for j, item in enumerate(batch):
                gi = global_offset + start + j
                dst = out_dir / f"{gi:05d}.png"
                pil_images[j].save(dst)
                score = float(rewards_cpu[j].item()) if rewards_cpu is not None else float("nan")
                if rewards_cpu is not None:
                    agg_scores.append(score)
                rec = {
                    "index": gi,
                    "image": str(dst),
                    "caption": item["caption"],
                    "hps": score,
                }
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"Batch {start // bs + 1}/{(n + bs - 1) // bs} done (saved {len(batch)} images).")

        if world_size > 1:
            dist.barrier()

    if is_main and agg_scores:
        mean_hps = sum(agg_scores) / len(agg_scores)
        summary = {"num_images": len(agg_scores), "mean_hps": mean_hps}
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Mean HPSv2 ({args.hps_version}): {mean_hps:.4f} over {len(agg_scores)} images.")
    elif is_main:
        print("Done (HPS skipped).")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
