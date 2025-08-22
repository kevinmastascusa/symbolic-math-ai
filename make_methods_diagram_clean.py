#!/usr/bin/env python3
"""Build a clean Methods diagram with clear flow and no text overflow.

Outputs under trained_math_model_qwen_run2/:
- methods_diagram_clean.png
- methods_diagram_clean.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def draw_box(
    ax: plt.Axes,
    bottom_left: Tuple[float, float],
    width: float,
    height: float,
    title: str,
    bullets: List[str],
) -> Tuple[float, float, float, float]:
    """Draw a rounded box with a bold title and left-aligned bullets.

    Returns the (x0, y0, x1, y1) bounds for arrow anchoring.
    """
    x0, y0 = bottom_left
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

    # Render header and bullets ABOVE the box with padding
    pad_x = 0.6
    pad_y = 0.8
    cx = x0 + width / 2.0
    title_y = y0 + height + pad_y + 0.8
    # Bullets should be INSIDE the box, near the top with padding
    bullets_top_y = y0 + height - pad_y

    ax.text(
        cx,
        title_y,
        title,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        zorder=2,
    )

    lines = [f"• {b}" for b in bullets]
    ax.text(
        x0 + pad_x,
        bullets_top_y,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        zorder=2,
    )
    return (x0, y0, x0 + width, y0 + height)


def arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="0.25"),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(28, 14))

    # Coordinate system will be set after we know total width
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Layout constants
    box_w = 20.0
    box_h = 12.0
    gap_x = 8.0
    base_y = 60.0

    # Main stages (7 boxes)
    titles = [
        "Datasets",
        "Preprocessing",
        "EDA",
        "Model Choice",
        "Fine‑tune",
        "Evaluation",
        "Inference & App",
    ]
    bullets = [
        ["GSM8K, MathQA", "SVAMP, Math500, custom"],
        ["LaTeX cleanup", "number normalization", "train/val split"],
        ["operator mix", "leakage & difficulty checks"],
        ["Qwen2.5‑Math‑1.5B", "open weights, tokenizer"],
        ["4‑bit QLoRA", "single‑GPU throughput", "checkpoints"],
        ["EM on GSM8K", "MathQA, SVAMP, Math500"],
        ["Streamlit UI", "SymPy reasoning"],
    ]

    boxes: List[Tuple[float, float, float, float]] = []
    left_margin = 2.0
    right_margin = 2.0
    x = left_margin
    for i in range(len(titles)):
        boxes.append(
            draw_box(ax, (x, base_y), box_w, box_h, titles[i], bullets[i])
        )
        x += box_w + gap_x

    total_width = (
        left_margin
        + len(titles) * box_w
        + (len(titles) - 1) * gap_x
        + right_margin
    )
    ax.set_xlim(0, total_width)

    # Bottom nodes
    bottom_gap = 18.0
    bottom_y = base_y - box_h - bottom_gap
    tm = draw_box(
        ax,
        (boxes[4][0], bottom_y),
        24.0,
        12.0,
        "Training Metrics",
        ["loss/accuracy curves", "metrics.json"],
    )
    ma = draw_box(
        ax,
        (boxes[5][0] + 2.0, bottom_y),
        26.0,
        12.0,
        "Model Artifact",
        ["Upload to Hugging Face Hub", "(optional)"],
    )

    # Flow arrows left-to-right between top boxes
    for i in range(len(boxes) - 1):
        right = (boxes[i][2], (boxes[i][1] + boxes[i][3]) / 2.0)
        left = (boxes[i + 1][0], (boxes[i + 1][1] + boxes[i + 1][3]) / 2.0)
        arrow(ax, (right[0] + 0.8, right[1]), (left[0] - 0.8, left[1]))

    # Down arrows to bottom boxes
    mid_finetune = ((boxes[4][0] + boxes[4][2]) / 2.0, boxes[4][1])
    mid_eval = ((boxes[5][0] + boxes[5][2]) / 2.0, boxes[5][1])
    arrow(
        ax,
        (mid_finetune[0], mid_finetune[1] - 0.5),
        (tm[0] + (tm[2] - tm[0]) / 2.0, tm[3]),
    )
    arrow(
        ax,
        (mid_eval[0], mid_eval[1] - 0.5),
        (ma[0] + (ma[2] - ma[0]) / 2.0, ma[3]),
    )

    ax.set_title(
        "Symbolic‑Math AI: Methods Overview",
        fontsize=18,
        pad=18,
    )

    out = Path("trained_math_model_qwen_run2")
    out.mkdir(parents=True, exist_ok=True)
    (out / "methods_diagram_clean.png").write_bytes(b"")  # ensure path exists
    png = out / "methods_diagram_clean.png"
    svg = out / "methods_diagram_clean.svg"
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(svg, bbox_inches="tight", pad_inches=0.2)
    print(str(png))
    print(str(svg))


if __name__ == "__main__":
    main()
