#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def main() -> None:
    # Qualitative comparison matrix.
    # Values: 2=High, 1=Medium, 0=Low, -1=N/A
    # Keep this general and constraint-driven (VRAM, speed, math
    # specialization, tokenizer support, availability).
    models = [
        "Qwen2.5-Math-1.5B",
        "Mistral 7B Instruct",
        "Llama 3.1 8B Instruct",
        "Phi-3 Mini (3.8B)",
    ]

    criteria = [
        "Math specialization",
        "Tokenizer math/LaTeX coverage",
        "Fine-tune VRAM need (4-bit LoRA)",
        "Single-GPU throughput",
        "Open weights & license fit",
    ]

    # Scores are intentionally qualitative; prefer conservative, widely
    # accepted expectations
    grid = np.array([
        # Qwen2.5-Math-1.5B
        [2, 2, 2, 2, 2],
        # Mistral 7B Instruct
        [1, 1, 1, 1, 2],
        # Llama 3.1 8B Instruct
        [1, 1, 1, 1, 2],
        # Phi-3 Mini (3.8B)
        [1, 1, 2, 2, 2],
    ], dtype=float)

    # Build a colormap: use traffic-light colors for qualitative cells
    fig, ax = plt.subplots(figsize=(10, 5.8))

    # Create a background grid for qualitative criteria (all columns)
    qualitative = grid
    # Use distinct, presentation-friendly colors:
    # -1: gray, 0: red, 1: yellow, 2: green
    color_map = {
        -1: (0.88, 0.88, 0.88),   # N/A
        0: (0.90, 0.45, 0.45),    # Low
        1: (0.99, 0.80, 0.20),    # Medium
        2: (0.30, 0.72, 0.35),    # High
    }

    # Draw qualitative cells explicitly to guarantee color-label alignment
    n_rows, n_cols = qualitative.shape
    for i in range(n_rows):
        for j in range(n_cols):
            score = int(qualitative[i, j])
            cell_color = color_map.get(score, (0.95, 0.95, 0.95))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cell_color))

    # Grid lines
    total_cols = qualitative.shape[1]
    for x in range(total_cols + 1):
        ax.axvline(x, color="white", linewidth=1)
    for y in range(len(models) + 1):
        ax.axhline(y, color="white", linewidth=1)

    # Text annotations
    for i in range(len(models)):
        for j in range(qualitative.shape[1]):
            score = int(qualitative[i, j])
            label = {2: "High", 1: "Med", 0: "Low", -1: "N/A"}.get(
                score, ""
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=10,
            )

    # Axes cosmetics
    ax.set_xticks(np.arange(total_cols) + 0.5)
    ax.set_yticks(np.arange(len(models)) + 0.5)
    ax.set_xticklabels(criteria, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlim(0, total_cols)
    ax.set_ylim(0, len(models))
    ax.invert_yaxis()
    ax.set_title(
        "Why Qwen2.5‑Math‑1.5B as Base Model",
        fontsize=14,
        pad=12,
    )
    ax.tick_params(left=False, bottom=False)

    # Legend for qualitative colors
    legend_patches = [
        Patch(facecolor=color_map[2], label="High"),
        Patch(facecolor=color_map[1], label="Medium"),
        Patch(facecolor=color_map[0], label="Low"),
        Patch(facecolor=color_map[-1], label="N/A"),
    ]
    # Place legend near bottom center above footnote
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        title="Qualitative score",
    )

    # Footnote
    foot = (
        "Qualitative scores reflect project constraints (math focus, VRAM, "
        "single‑GPU speed, licensing)."
    )
    plt.figtext(0.01, 0.01, foot, ha="left", va="bottom", fontsize=8)

    out_dir = Path("trained_math_model_qwen_run2")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "model_choice_comparison.png"
    svg = out_dir / "model_choice_comparison.svg"
    # Leave bottom margin for legend + footnote
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(png, dpi=300)
    plt.savefig(svg)
    print(str(png))
    print(str(svg))


if __name__ == "__main__":
    main()
