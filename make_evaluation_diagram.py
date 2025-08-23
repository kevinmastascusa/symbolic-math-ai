#!/usr/bin/env python3
"""
Generate a clean evaluation diagram (SVG/PNG) from eval_all.json for slides.

Reads key metrics (EM and BLEU/METEOR/ROUGE) and per-dataset EM, then renders
two figures:
  1) A compact scoreboard with overall metrics
  2) A grouped bar chart per dataset showing EM

Outputs:
  - trained_math_model_qwen_run2/eval_scoreboard.svg/png
  - trained_math_model_qwen_run2/eval_datasets.svg/png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np


def load_eval(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_pct(x: float) -> str:
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "-"


def draw_scoreboard(data: Dict[str, Any], out_dir: Path) -> None:
    overall_em = data.get("overall_exact_match")
    metrics = data.get("overall_metrics", {})

    labels = [
        "Exact Match",
        "BLEU-4",
        "METEOR",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
    ]
    values = [
        format_pct(overall_em),
        f"{metrics.get('bleu', 0):.2f}",
        f"{metrics.get('meteor', 0):.3f}",
        f"{metrics.get('rouge1', 0):.3f}",
        f"{metrics.get('rouge2', 0):.3f}",
        f"{metrics.get('rougeL', 0):.3f}",
    ]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
    ax.axis("off")
    ax.set_title("Overall Evaluation", fontsize=18, pad=12)

    table_data = [[l, v] for l, v in zip(labels, values)]
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colColours=["#f1f3f5", "#f1f3f5"],
    )
    table.scale(1.2, 1.6)
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    out_svg = out_dir / "eval_scoreboard.svg"
    out_png = out_dir / "eval_scoreboard.png"
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.2, dpi=300)
    plt.close(fig)


def draw_datasets(data: Dict[str, Any], out_dir: Path) -> None:
    per = data.get("per_dataset", {})
    ds_names: List[str] = ["gsm8k", "mathqa", "svamp", "math500"]
    xs = np.arange(len(ds_names))
    em_vals = [float(per.get(n, {}).get("exact_match", 0.0)) * 100.0 for n in ds_names]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars = ax.bar(xs, em_vals, color="#7db3ff", edgecolor="#2b4c7e")
    ax.set_xticks(xs, [n.upper() for n in ds_names])
    ax.set_ylim(0, max(100.0, max(em_vals) + 10))
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Per-dataset EM")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    for b, val in zip(bars, em_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

    out_svg = out_dir / "eval_datasets.svg"
    out_png = out_dir / "eval_datasets.png"
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.2, dpi=300)
    plt.close(fig)


def main() -> None:
    eval_path = Path("eval_all.json")
    if not eval_path.exists():
        raise SystemExit("eval_all.json not found. Run evaluation first.")
    data = load_eval(eval_path)

    out_dir = Path("trained_math_model_qwen_run2")
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_scoreboard(data, out_dir)
    draw_datasets(data, out_dir)
    print("Saved diagrams to:", out_dir)


if __name__ == "__main__":
    main()


