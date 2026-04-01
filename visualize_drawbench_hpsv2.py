#!/usr/bin/env python3
"""Parse COMPARISON SUMMARY from rollout logs and plot HPSv2 mean (excluding original baselines).

Rollouts visualized here are from the Qwen-Image generation model (DrawBench comparison logs).
For FLUX.1 logs, use `visualize_drawbench_flux_hpsv2.py`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LINE = re.compile(r"^Method\s+\d+\s+\((.*)\):\s*$")
MEAN_RE = re.compile(r"\*\s+Mean:\s*([0-9.]+)")
SAVE_RE = re.compile(r"Results saved to:\s*(.+?)\s*$", re.MULTILINE)


def parse_comparison_summary(text: str) -> list[tuple[str, float, str]]:
    """Return list of (method_title, hpsv2_mean, save_dir_line) per method block."""
    idx = text.find("COMPARISON SUMMARY")
    if idx >= 0:
        text = text[idx:]

    rows: list[tuple[str, float, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        mm = METHOD_LINE.match(lines[i].strip())
        if not mm:
            i += 1
            continue
        title = mm.group(1).strip()
        i += 1
        if i < len(lines) and set(lines[i].strip()) == {"="}:
            i += 1
        body_lines: list[str] = []
        while i < len(lines):
            if METHOD_LINE.match(lines[i].strip()):
                break
            if lines[i].strip().startswith("Reward Comparison:"):
                break
            if set(lines[i].strip()) == {"="} and len(lines[i].strip()) >= 20:
                nxt = i + 1
                if nxt < len(lines) and (
                    METHOD_LINE.match(lines[nxt].strip())
                    or lines[nxt].strip().startswith("Reward Comparison:")
                ):
                    break
            body_lines.append(lines[i])
            i += 1
        body = "\n".join(body_lines)
        m_mean = MEAN_RE.search(body)
        m_save = SAVE_RE.search(body)
        if not m_mean or not m_save:
            continue
        rows.append((title, float(m_mean.group(1)), m_save.group(1).strip()))
    return rows


def is_original_baseline(save_path: str, title: str) -> bool:
    folder = Path(save_path).name
    if folder in ("original", "original_14"):
        return True
    if title.strip().lower().startswith("original"):
        return True
    return False


def plot_hpsv2_means(
    labels: list[str],
    means: list[float],
    out_path: Path,
    title: str = "HPSv2 mean (DrawBench comparison, non-baseline methods)",
    xlim: tuple[float, float] | None = None,
) -> None:
    order = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
    labels_o = [labels[i] for i in order]
    means_o = [means[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.45 * len(labels_o))))
    y = range(len(labels_o))
    bars = ax.barh(y, means_o, color="steelblue", edgecolor="white", linewidth=0.6)
    ax.set_yticks(list(y), labels_o, fontsize=9)
    ax.set_xlabel("HPSv2 reward (mean)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.35)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0.24, 0.26)

    for bar, v in zip(bars, means_o):
        ax.text(
            v + 0.0003,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            ha="left",
            fontsize=8,
            color="0.25",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "log",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "drawbench2.out",
        help="Path to log containing COMPARISON SUMMARY (default: ./drawbench2.out)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <log_stem>_hpsv2_non_baseline.png next to log)",
    )
    args = p.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    all_rows = parse_comparison_summary(text)
    if not all_rows:
        raise SystemExit(f"No method blocks parsed from {args.log}")

    filtered = [(t, m, s) for t, m, s in all_rows if not is_original_baseline(s, t)]
    if not filtered:
        raise SystemExit("After excluding original baselines, nothing left to plot.")

    labels = [t for t, _, _ in filtered]
    means = [m for _, m, _ in filtered]

    out = args.out
    if out is None:
        out = args.log.with_name(f"{args.log.stem}_hpsv2_non_baseline.png")

    plot_hpsv2_means(
        labels,
        means,
        out,
        title="HPSv2 mean — Qwen-Image (DrawBench, non-baseline methods)",
    )
    print(f"Wrote {out} ({len(filtered)} methods)")


if __name__ == "__main__":
    main()
