#!/usr/bin/env python3
"""Parse COMPARISON SUMMARY from FLUX.1 DrawBench rollout logs and plot HPSv2 mean (no baselines).

Same log format as `visualize_drawbench_hpsv2.py`, but scores are for FLUX.1 generations.
Qwen-Image results use `visualize_drawbench_hpsv2.py` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visualize_drawbench_hpsv2 import (
    is_original_baseline,
    parse_comparison_summary,
    plot_hpsv2_means,
)


def _xlim_for_means(means: list[float], label_pad: float = 0.0012) -> tuple[float, float]:
    lo, hi = min(means), max(means)
    span = max(hi - lo, 0.004)
    return (lo - span * 0.2, hi + span * 0.55 + label_pad)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "log",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "drawbench_flux.out",
        help="Path to FLUX.1 log with COMPARISON SUMMARY (default: ./drawbench_flux.out)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <log_stem>_hpsv2_flux1_non_baseline.png next to log)",
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
        out = args.log.with_name(f"{args.log.stem}_hpsv2_flux1_non_baseline.png")

    plot_hpsv2_means(
        labels,
        means,
        out,
        title="HPSv2 mean — FLUX.1 (DrawBench, non-baseline methods)",
        xlim=_xlim_for_means(means),
    )
    print(f"Wrote {out} ({len(filtered)} methods)")


if __name__ == "__main__":
    main()
