#!/usr/bin/env python3
"""Read logging.jsonl under a run root and save training analysis figures there."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def col(rows: list[dict[str, Any]], key: str) -> np.ndarray | None:
    if not rows or key not in rows[0]:
        return None
    try:
        return np.array([r[key] for r in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="valid")


def plot_reward(
    steps: np.ndarray,
    reward: np.ndarray,
    out: Path,
    smooth_window: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, reward, alpha=0.35, label="avg_reward (per step)", color="C0")
    if len(reward) >= smooth_window:
        sm = moving_average(reward, smooth_window)
        sm_steps = steps[smooth_window - 1 :]
        ax.plot(sm_steps, sm, lw=2, label=f"moving avg (w={smooth_window})", color="C1")
    ax.set_xlabel("step")
    ax.set_ylabel("avg_reward")
    ax.set_title("Average reward")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_loss(steps: np.ndarray, loss: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, loss, lw=0.8, color="C2")
    ax.set_xlabel("step")
    ax.set_ylabel("train_loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_loss_log_abs(steps: np.ndarray, loss: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.maximum(np.abs(loss), 1e-12)
    ax.semilogy(steps, y, lw=0.8, color="C2")
    ax.set_xlabel("step")
    ax.set_ylabel("|train_loss| (log scale)")
    ax.set_title("Training loss magnitude (log)")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_grad_norm(steps: np.ndarray, grad_norm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, grad_norm, lw=0.8, color="C3")
    ax.set_xlabel("step")
    ax.set_ylabel("grad_norm")
    ax.set_title("Gradient norm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_timing(
    steps: np.ndarray,
    step_time: np.ndarray | None,
    avg_step_time: np.ndarray | None,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if step_time is not None:
        ax.plot(steps, step_time, alpha=0.5, lw=0.8, label="step_time")
    if avg_step_time is not None:
        ax.plot(steps, avg_step_time, lw=1.5, label="avg_step_time")
    ax.set_xlabel("step")
    ax.set_ylabel("seconds")
    ax.set_title("Step timing")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_lr(steps: np.ndarray, lr: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, lr, lw=1.0, color="C4")
    ax.set_xlabel("step")
    ax.set_ylabel("learning_rate")
    ax.set_title("Learning rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dashboard(
    steps: np.ndarray,
    rows: list[dict[str, Any]],
    out: Path,
    smooth_window: int = 10,
) -> None:
    reward = col(rows, "avg_reward")
    loss = col(rows, "train_loss")
    grad_norm = col(rows, "grad_norm")
    step_time = col(rows, "step_time")
    avg_step_time = col(rows, "avg_step_time")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax0, ax1, ax2, ax3 = axes.ravel()

    if reward is not None:
        ax0.plot(steps, reward, alpha=0.35, color="C0")
        if len(reward) >= smooth_window:
            sm = moving_average(reward, smooth_window)
            ax0.plot(steps[smooth_window - 1 :], sm, lw=2, color="C1")
    ax0.set_title("avg_reward")
    ax0.set_xlabel("step")
    ax0.grid(True, alpha=0.3)

    if loss is not None:
        ax1.plot(steps, loss, lw=0.6, color="C2")
    ax1.set_title("train_loss")
    ax1.set_xlabel("step")
    ax1.grid(True, alpha=0.3)

    if grad_norm is not None:
        ax2.plot(steps, grad_norm, lw=0.6, color="C3")
    ax2.set_title("grad_norm")
    ax2.set_xlabel("step")
    ax2.grid(True, alpha=0.3)

    if step_time is not None:
        ax3.plot(steps, step_time, alpha=0.45, lw=0.6, label="step_time")
    if avg_step_time is not None:
        ax3.plot(steps, avg_step_time, lw=1.2, label="avg_step_time")
    ax3.set_title("timing (s)")
    ax3.set_xlabel("step")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Training overview", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RL training logs from logging.jsonl")
    parser.add_argument(
        "root_path",
        nargs="?",
        default="/mnt/shared-storage-user/mineru4s/dingruiyi/DanceGRPO/data/outputs/grpo_standard",
        type=str,
        help="Directory containing logging.jsonl; figures are written here",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Moving average window for reward curves (default: 10)",
    )
    args = parser.parse_args()
    root = Path(os.path.expanduser(args.root_path)).resolve()
    log_path = root / "logging.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing {log_path}")

    rows = load_jsonl(log_path)
    if not rows:
        raise ValueError(f"Empty log: {log_path}")

    steps = col(rows, "step")
    if steps is None:
        steps = np.arange(len(rows), dtype=np.float64)

    outputs = [
        ("analysis_reward.png", lambda: plot_reward(steps, col(rows, "avg_reward"), root / "analysis_reward.png", args.smooth)),
        ("analysis_loss.png", lambda: plot_loss(steps, col(rows, "train_loss"), root / "analysis_loss.png")),
        ("analysis_loss_log_abs.png", lambda: plot_loss_log_abs(steps, col(rows, "train_loss"), root / "analysis_loss_log_abs.png")),
        ("analysis_grad_norm.png", lambda: plot_grad_norm(steps, col(rows, "grad_norm"), root / "analysis_grad_norm.png")),
        ("analysis_timing.png", lambda: plot_timing(steps, col(rows, "step_time"), col(rows, "avg_step_time"), root / "analysis_timing.png")),
        ("analysis_lr.png", lambda: plot_lr(steps, col(rows, "learning_rate"), root / "analysis_lr.png")),
        ("analysis_dashboard.png", lambda: plot_dashboard(steps, rows, root / "analysis_dashboard.png", args.smooth)),
    ]

    for name, fn in outputs:
        path = root / name
        try:
            if name == "analysis_reward.png" and col(rows, "avg_reward") is None:
                continue
            if name == "analysis_loss.png" and col(rows, "train_loss") is None:
                continue
            if name == "analysis_loss_log_abs.png" and col(rows, "train_loss") is None:
                continue
            if name == "analysis_grad_norm.png" and col(rows, "grad_norm") is None:
                continue
            if name == "analysis_timing.png":
                if col(rows, "step_time") is None and col(rows, "avg_step_time") is None:
                    continue
            if name == "analysis_lr.png" and col(rows, "learning_rate") is None:
                continue
            fn()
            print(f"Wrote {path}")
        except Exception as e:
            print(f"Skip {name}: {e}")

    print(f"Done. Outputs under {root}")


if __name__ == "__main__":
    main()
