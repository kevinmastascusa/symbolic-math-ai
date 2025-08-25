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
    except (TypeError, ValueError):
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
    overall_em_val = float(overall_em) if overall_em is not None else 0.0
    values = [
        format_pct(overall_em_val),
        f"{metrics.get('bleu', 0):.2f}",
        f"{metrics.get('meteor', 0):.3f}",
        f"{metrics.get('rouge1', 0):.3f}",
        f"{metrics.get('rouge2', 0):.3f}",
        f"{metrics.get('rougeL', 0):.3f}",
    ]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
    ax.axis("off")
    ax.set_title("Overall Evaluation", fontsize=18, pad=12)

    table_data = [[label, val] for label, val in zip(labels, values)]
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
    em_vals = [
        float(per.get(n, {}).get("exact_match", 0.0)) * 100.0
        for n in ds_names
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars = ax.bar(xs, em_vals, color="#7db3ff", edgecolor="#2b4c7e")
    ax.set_xticks(xs, [n.upper() for n in ds_names])
    ax.set_ylim(0, max(100.0, max(em_vals) + 10))
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Per-dataset EM")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    for b, val in zip(bars, em_vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    out_svg = out_dir / "eval_datasets.svg"
    out_png = out_dir / "eval_datasets.png"
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.2, dpi=300)
    plt.close(fig)


def draw_reasoning(data: Dict[str, Any], out_dir: Path) -> None:
    """Render per-dataset reasoning statistics (rs_* fields).

    Metrics shown per dataset:
      - Avg equations per example = rs_equations / total
      - Has any equation (%) = rs_examples_with_any_eq / total * 100
      - Solvable (%) = rs_solvable / total * 100
      - Consistent (%) = rs_consistent / total * 100
    """
    per = data.get("per_dataset", {})
    ds_names: List[str] = ["gsm8k", "mathqa", "svamp", "math500"]

    totals = [max(1, int(per.get(n, {}).get("total", 0))) for n in ds_names]
    rs_eq = [float(per.get(n, {}).get("rs_equations", 0.0)) for n in ds_names]
    rs_has = [
        float(per.get(n, {}).get("rs_examples_with_any_eq", 0.0))
        for n in ds_names
    ]
    rs_solv = [float(per.get(n, {}).get("rs_solvable", 0.0)) for n in ds_names]
    rs_cons = [float(per.get(n, {}).get("rs_consistent", 0.0)) for n in ds_names]

    eq_per_ex = [r / t for r, t in zip(rs_eq, totals)]
    has_pct = [r / t * 100.0 for r, t in zip(rs_has, totals)]
    solv_pct = [r / t * 100.0 for r, t in zip(rs_solv, totals)]
    cons_pct = [r / t * 100.0 for r, t in zip(rs_cons, totals)]

    x = np.arange(len(ds_names))
    width = 0.2

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), dpi=200)

    # Top: percentage-based metrics
    ax = ax_top
    b1 = ax.bar(
        x - 1.5 * width,
        has_pct,
        width,
        label="Has Eq (%)",
        color="#6aa6ff",
    )
    b2 = ax.bar(
        x - 0.5 * width,
        solv_pct,
        width,
        label="Solvable (%)",
        color="#8bd3c7",
    )
    b3 = ax.bar(
        x + 0.5 * width,
        cons_pct,
        width,
        label="Consistent (%)",
        color="#ffb480",
    )
    ax.set_xticks(x, [n.upper() for n in ds_names])
    ax.set_ylabel("Percent of examples (%)")
    ax.set_title("Per-dataset Reasoning Stats (percent)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(ncols=3, loc="upper right")
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 1.0,
                f"{b.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Bottom: average number of equations per example
    ax2 = ax_bot
    b4 = ax2.bar(
        x, eq_per_ex, width=0.6, color="#b28dff", edgecolor="#4b3f72"
    )
    ax2.set_xticks(x, [n.upper() for n in ds_names])
    ax2.set_ylabel("Avg equations/example")
    ax2.set_title("Avg extracted equations per example")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    for b, r, t in zip(b4, rs_eq, totals):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{r:.0f}/{t}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_svg = out_dir / "eval_reasoning.svg"
    out_png = out_dir / "eval_reasoning.png"
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
    draw_reasoning(data, out_dir)
    print("Saved diagrams to:", out_dir)


if __name__ == "__main__":
    main()


