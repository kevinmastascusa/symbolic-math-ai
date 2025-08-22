#!/usr/bin/env python3
"""Generate a Methods diagram for the project using Matplotlib.

Pipeline: Data → Preprocessing → EDA → Model Choice → Fine‑tuning → Metrics
→ Evaluation → App/Deployment

Outputs:
- trained_math_model_qwen_run2/methods_diagram.png
- trained_math_model_qwen_run2/methods_diagram.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass

import matplotlib

# Headless backend for non‑interactive render in CI/servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def draw_box(
    ax: plt.Axes,
    center_xy: Tuple[float, float],
    text: str,
    width: float = 2.4,
    height: float = 1.2,
    facecolor: Tuple[float, float, float] = (0.96, 0.96, 0.96),
    edgecolor: Tuple[float, float, float] = (0.25, 0.25, 0.25),
    fontsize: int = 11,
) -> None:
    """Draw a rounded box with centered text at the given location."""
    cx, cy = center_xy
    x0 = cx - width / 2.0
    y0 = cy - height / 2.0
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
        zorder=5,
    )


@dataclass
class NodeSpec:
    xy: Tuple[float, float]
    title: str
    details: List[str]
    w: float
    h: float


def draw_node(
    ax: plt.Axes,
    center_xy: Tuple[float, float],
    title: str,
    details: List[str] | None,
    width: float,
    height: float,
    *,
    title_size: int = 12,
    detail_size: int = 10,
) -> None:
    """Draw a rounded box with a title and left-aligned bullet details."""
    cx, cy = center_xy
    x0 = cx - width / 2.0
    y0 = cy - height / 2.0
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        facecolor=(0.96, 0.96, 0.96),
        edgecolor=(0.25, 0.25, 0.25),
        zorder=1,
    )
    ax.add_patch(patch)

    # Title centered near top
    ax.text(
        cx,
        cy + (height * 0.22),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        zorder=2,
    )

    if details:
        lines = [f"• {ln}" for ln in details]
        # Left padding for details
        ax.text(
            x0 + 0.18,
            cy - (height * 0.10),
            "\n".join(lines),
            ha="left",
            va="center",
            fontsize=detail_size,
            zorder=2,
            wrap=True,
        )


def arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> None:
    """Draw a simple arrow from start to end coordinates."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="0.25"),
    )


def main() -> None:
    ax = plt.subplots(figsize=(20, 10))[1]

    # Layout coordinates (x, y)
    # Left to right main path on y = 5
    data_xy = (2.0, 5.0)
    prep_xy = (7.0, 5.0)
    eda_xy = (12.0, 5.0)
    choice_xy = (17.0, 5.0)
    train_xy = (22.0, 5.0)
    eval_xy = (27.0, 5.0)
    app_xy = (32.0, 5.0)

    # Side nodes
    metrics_xy = (22.0, 3.0)
    hub_xy = (27.0, 3.0)

    # Boxes as titled nodes (left-aligned bullet details)
    nodes: Dict[str, NodeSpec] = {
        "data": NodeSpec(
            data_xy,
            "Datasets",
            ["GSM8K, MathQA", "SVAMP, Math500, custom"],
            3.6,
            1.8,
        ),
        "prep": NodeSpec(
            prep_xy,
            "Preprocessing",
            ["LaTeX cleanup", "number normalization", "train/val split"],
            3.9,
            2.1,
        ),
        "eda": NodeSpec(
            eda_xy,
            "EDA",
            ["operator mix", "leakage & difficulty checks"],
            3.6,
            1.9,
        ),
        "choice": NodeSpec(
            choice_xy,
            "Model Choice",
            ["Qwen2.5‑Math‑1.5B", "open weights, tokenizer"],
            3.7,
            1.9,
        ),
        "train": NodeSpec(
            train_xy,
            "Fine‑tune",
            ["4‑bit QLoRA", "single‑GPU throughput", "checkpoints"],
            3.9,
            2.2,
        ),
        "eval": NodeSpec(
            eval_xy,
            "Evaluation",
            ["EM on GSM8K", "MathQA, SVAMP, Math500"],
            3.8,
            1.9,
        ),
        "app": NodeSpec(
            app_xy,
            "Inference & App",
            ["Streamlit UI", "SymPy reasoning"],
            3.6,
            1.9,
        ),
    }

    for key in ["data", "prep", "eda", "choice", "train", "eval", "app"]:
        n = nodes[key]
        draw_node(ax, n.xy, n.title, n.details, n.w, n.h)

    # Auxiliary
    draw_node(
        ax,
        metrics_xy,
        "Training Metrics",
        ["loss/accuracy curves", "metrics.json"],
        3.2,
        1.9,
    )
    draw_node(
        ax,
        hub_xy,
        "Model Artifact",
        ["Upload to Hugging Face Hub", "(optional)"],
        3.6,
        2.0,
    )

    # Arrows along main path (computed from node widths)
    def right_edge(k: str) -> float:
        n = nodes[k]
        return n.xy[0] + n.w / 2.0

    def left_edge(k: str) -> float:
        n = nodes[k]
        return n.xy[0] - n.w / 2.0

    g = 0.6
    arrow(
        ax,
        (right_edge("data") + g, data_xy[1]),
        (left_edge("prep") - g, prep_xy[1]),
    )
    arrow(
        ax,
        (right_edge("prep") + g, prep_xy[1]),
        (left_edge("eda") - g, eda_xy[1]),
    )
    arrow(
        ax,
        (right_edge("eda") + g, eda_xy[1]),
        (left_edge("choice") - g, choice_xy[1]),
    )
    arrow(
        ax,
        (right_edge("choice") + g, choice_xy[1]),
        (left_edge("train") - g, train_xy[1]),
    )
    arrow(
        ax,
        (right_edge("train") + g, train_xy[1]),
        (left_edge("eval") - g, eval_xy[1]),
    )
    arrow(
        ax,
        (right_edge("eval") + g, eval_xy[1]),
        (left_edge("app") - g, app_xy[1]),
    )

    # Arrows to side nodes
    arrow(
        ax,
        (train_xy[0], train_xy[1] - 0.6),
        (metrics_xy[0], metrics_xy[1] + 0.55),
    )
    arrow(
        ax,
        (eval_xy[0] - 0.2, eval_xy[1] - 0.6),
        (hub_xy[0] + 0.4, hub_xy[1] + 0.55),
    )

    # Style
    ax.set_xlim(0, 36.0)
    ax.set_ylim(2.0, 8.0)
    # Allow autoscaling aspect so text has more vertical space
    ax.axis("off")
    ax.set_title(
        "Symbolic‑Math AI: Methods Overview",
        fontsize=14,
        pad=14,
    )

    out_dir = Path("trained_math_model_qwen_run2")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "methods_diagram.png"
    svg = out_dir / "methods_diagram.svg"
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(png, dpi=300)
    plt.savefig(svg)
    print(str(png))
    print(str(svg))


if __name__ == "__main__":
    main()
