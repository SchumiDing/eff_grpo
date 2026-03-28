#!/usr/bin/env python3
# Copyright: DanceGRPO / user script — single-GPU Qwen-Image (diffusers) inference with
# a DiT-only checkpoint folder, precomputed text embeddings, and HPSv2 evaluation.
#
# Usage (from repo root, with your conda env that has diffusers + torch):
#   export PYTHONPATH=/path/to/DanceGRPO
#   python scripts/infer_qwen_dit_hpsv2_single_gpu.py \
#     --dit_checkpoint data/outputs/grpo_eff_auto_mta/checkpoint-26-0 \
#     --base_model data/qwenimage \
#     --embeddings_path data/qwenimage/rl_embeddings_drawbench \
#     --output_dir outputs/drawbench_ckpt26_eval
#
# Optional: add HPSv2 to PYTHONPATH or `pip install -e /path/to/HPSv2`.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_paths() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-GPU Qwen-Image: load DiT checkpoint + RL embeddings, sample, HPSv2 score."
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
        help="Full Qwen-Image diffusers tree (VAE + scheduler metadata). Default: <repo>/data/qwenimage",
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
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None, help="Exclusive global end index (default: all).")
    p.add_argument("--device", type=str, default="cuda:0")
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


@dataclass
class SampleArgs:
    w: int
    h: int
    t: int
    sampling_steps: int
    shift: float
    eta: float
    init_same_noise: bool
    use_hpsv2: bool
    use_hpsv3: bool
    use_pickscore: bool


def _load_transformer(dit_checkpoint: str, attn: str):
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    kwargs = {"torch_dtype": __import__("torch").bfloat16}
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
    import torch

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
    import torch
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import hps_version_map

    # Local file (e.g. hps_ckpt/open_clip_pytorch_model.bin) avoids any pretrained tag download.
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


def main() -> None:
    args = _parse_args()
    _ensure_paths()

    if args.base_model is None:
        args.base_model = str(_repo_root() / "data" / "qwenimage")

    dit_ckpt = os.path.abspath(os.path.expanduser(args.dit_checkpoint))
    base_model = os.path.abspath(os.path.expanduser(args.base_model))
    emb_root = os.path.abspath(os.path.expanduser(args.embeddings_path))
    out_dir = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-process env (train_grpo_qwenimage expects RANK for filenames)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    scratch = out_dir / "_rollout_scratch"
    scratch.mkdir(exist_ok=True)
    os.environ["DANCEGRPO_ROLLOUT_IMAGE_DIR"] = str(scratch)

    import torch
    from diffusers.models.autoencoders import AutoencoderKLQwenImage

    torch.manual_seed(args.seed)

    # HPSv2 on path: repo sibling or pip install
    hps_root = os.environ.get("HPSV2_ROOT", str(_repo_root().parent / "HPSv2"))
    if os.path.isdir(hps_root) and hps_root not in sys.path:
        sys.path.insert(0, hps_root)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print(f"Loading DiT from {dit_ckpt} ...")
    transformer = _load_transformer(dit_ckpt, args.attn).to(device)
    transformer.eval()

    print(f"Loading VAE from {base_model} (subfolder vae) ...")
    vae = AutoencoderKLQwenImage.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.bfloat16).to(
        device
    )

    reward_model = None
    preprocess_val = None
    tokenizer = None
    if not args.no_hps:
        print("Loading HPSv2 ...")
        oc_path = args.hps_open_clip
        if oc_path is None:
            _oc = _repo_root() / "hps_ckpt" / "open_clip_pytorch_model.bin"
            oc_path = str(_oc) if _oc.is_file() else None
        reward_model, preprocess_val, tokenizer = _init_hps(
            device, args.hps_version, args.hps_checkpoint, oc_path
        )

    from fastvideo.utils.rollout_image_dir import rollout_image_file
    from fastvideo.train_grpo_qwenimage import sample_reference_model

    json_path = os.path.join(emb_root, "videos2caption.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data_anno = json.load(f)

    end = args.end_idx if args.end_idx is not None else len(data_anno)
    data_anno = data_anno[args.start_idx : end]
    n = len(data_anno)
    print(f"Prompts in this run: {n} (indices {args.start_idx}..{end - 1})")

    prompt_embed_dir = os.path.join(emb_root, "prompt_embed")
    mask_dir = os.path.join(emb_root, "prompt_attention_mask")

    metrics_path = out_dir / "hps_metrics.jsonl"
    agg_scores: list[float] = []

    mock = SampleArgs(
        w=args.width,
        h=args.height,
        t=1,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        eta=args.eta,
        init_same_noise=False,
        use_hpsv2=not args.no_hps,
        use_hpsv3=False,
        use_pickscore=False,
    )

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

        # Clear scratch PNGs from previous batch
        for p in scratch.glob("qwenimage_*.png"):
            p.unlink(missing_ok=True)

        rewards, _, _, _, _ = sample_reference_model(
            mock,
            device,
            transformer,
            vae,
            prompt_embeds,
            prompt_attention_masks,
            original_length,
            reward_model,
            tokenizer,
            caps,
            preprocess_val,
        )
        rewards_cpu = rewards.detach().float().cpu()

        for j, item in enumerate(batch):
            gi = global_offset + start + j
            src = rollout_image_file(f"qwenimage_0_{j}.png")
            dst = out_dir / f"{gi:05d}.png"
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Expected rollout image missing: {src}")
            shutil.move(src, dst)
            score = float(rewards_cpu[j].item()) if not args.no_hps else float("nan")
            if not args.no_hps:
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

    if agg_scores:
        mean_hps = sum(agg_scores) / len(agg_scores)
        summary = {"num_images": len(agg_scores), "mean_hps": mean_hps}
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Mean HPSv2 ({args.hps_version}): {mean_hps:.4f} over {len(agg_scores)} images.")
    else:
        print("Done (HPS skipped).")

    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
