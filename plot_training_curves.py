#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) == 0:
        return list(values)
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=float) / float(window)
    return list(np.convolve(values, kernel, mode="same"))


def load_trainer_state(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_plot(
    trainer_state: dict,
    window: int,
    title: str = "Training/Eval Curves",
):
    history = trainer_state.get("log_history", [])

    steps = [h["step"] for h in history if "loss" in h]
    losses = [float(h["loss"]) for h in history if "loss" in h]
    smoothed = moving_average(losses, window)

    eval_steps = [h["step"] for h in history if "eval_loss" in h]
    eval_losses = [float(h["eval_loss"]) for h in history if "eval_loss" in h]

    acc_steps = [h["step"] for h in history if "eval_accuracy" in h]
    acc_vals = [float(h["eval_accuracy"]) * 100.0 for h in history if "eval_accuracy" in h]

    best_step = trainer_state.get("best_global_step")

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))

    if steps:
        ax1.plot(steps, losses, color="#9bbad8", alpha=0.45, label="train_loss")
        ax1.plot(steps, smoothed, color="#1f77b4", linewidth=2.0, label="train_loss (smoothed)")
    if eval_steps:
        ax1.plot(
            eval_steps,
            eval_losses,
            "o-",
            color="#ff7f0e",
            linewidth=2.0,
            markersize=6,
            label="eval_loss",
        )

    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, alpha=0.25)
    ax1.set_title(title, fontsize=14, pad=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = [], []

    # Secondary axis for accuracy if present
    if acc_steps:
        ax2 = ax1.twinx()
        ax2.plot(
            acc_steps,
            acc_vals,
            "s--",
            color="#2ca02c",
            linewidth=2.0,
            markersize=6,
            label="eval_accuracy (%)",
        )
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        lines2, labels2 = ax2.get_legend_handles_labels()

    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

    if best_step is not None:
        ymin, ymax = ax1.get_ylim()
        ax1.axvline(best_step, color="#d62728", linestyle="--", alpha=0.6)
        ax1.annotate(
            f"best @ {best_step}",
            xy=(best_step, ymin + 0.05 * (ymax - ymin)),
            xytext=(5, 0),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=9,
            color="#d62728",
        )

    plt.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training/eval curves from trainer_state.json")
    p.add_argument("--trainer_state", type=str, required=True, help="Path to trainer_state.json")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for figures")
    p.add_argument("--png", type=str, default="training_curve_enhanced.png")
    p.add_argument("--svg", type=str, default="training_curve_enhanced.svg")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--window", type=int, default=7, help="Moving average window for smoothing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts_path = Path(args.trainer_state)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_trainer_state(ts_path)
    fig = build_plot(state, window=int(args.window))

    png_path = out_dir / args.png
    svg_path = out_dir / args.svg
    fig.savefig(png_path, dpi=int(args.dpi))
    fig.savefig(svg_path)
    print(str(png_path))
    print(str(svg_path))


if __name__ == "__main__":
    main()


