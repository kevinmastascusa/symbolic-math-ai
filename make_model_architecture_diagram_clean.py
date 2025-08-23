#!/usr/bin/env python3
"""
Create a simplified Model Architecture diagram focused on readability.

Outputs:
- trained_math_model_qwen_run2/model_architecture_diagram_clean.png
- trained_math_model_qwen_run2/model_architecture_diagram_clean.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402


def draw_box(
    ax: Axes,
    xy: Tuple[float, float],
    w: float,
    h: float,
    title: str,
    *,
    title_size: int = 22,
    fc=(0.96, 0.96, 0.96),
    ec=(0.25, 0.25, 0.25),
) -> Tuple[float, float, float, float]:
    x0, y0 = xy
    patch = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        linewidth=1.4,
        facecolor=fc,
        edgecolor=ec,
        zorder=1,
    )
    ax.add_patch(patch)

    cx = x0 + w / 2.0
    cy = y0 + h / 2.0
    ax.text(
        cx,
        cy,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        zorder=2,
    )
    return (x0, y0, x0 + w, y0 + h)


def arrow(
    ax: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.6, color="0.25"),
    )


def mid_y(b: Tuple[float, float, float, float]) -> float:
    return (b[1] + b[3]) / 2.0


def mid_x(b: Tuple[float, float, float, float]) -> float:
    return (b[0] + b[2]) / 2.0


def center_of(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (mid_x(b), mid_y(b))


def main() -> None:
    fig, ax = plt.subplots(figsize=(34, 19))
    ax.axis("off")

    base_y = 50.0
    x = 3.0
    gap = 5.0

    # Main pipeline with short labels only
    input_box = draw_box(ax, (x, base_y), 12.0, 8.0, "Input Text")
    x = input_box[2] + gap
    tok = draw_box(ax, (x, base_y), 14.0, 8.0, "Tokenizer")
    x = tok[2] + gap
    ids = draw_box(ax, (x, base_y), 10.0, 8.0, "Token IDs")
    x = ids[2] + gap
    emb = draw_box(ax, (x, base_y), 16.0, 8.0, "Embeddings")
    x = emb[2] + gap
    stack = draw_box(ax, (x, base_y - 9.0), 42.0, 26.0, "")
    x = stack[2] + gap
    lm = draw_box(ax, (x, base_y), 16.0, 8.0, "LM Head")
    x = lm[2] + gap
    logits = draw_box(ax, (x, base_y), 12.0, 8.0, "Logits")
    x = logits[2] + gap
    out = draw_box(ax, (x, base_y), 16.0, 8.0, "Generated Text")

    # Flow arrows
    arrow(
        ax,
        (input_box[2] + 0.8, mid_y(input_box)),
        (tok[0] - 0.8, mid_y(tok)),
    )
    arrow(ax, (tok[2] + 0.8, mid_y(tok)), (ids[0] - 0.8, mid_y(ids)))
    arrow(ax, (ids[2] + 0.8, mid_y(ids)), (emb[0] - 0.8, mid_y(emb)))
    arrow(ax, (emb[2] + 0.8, mid_y(emb)), (stack[0] - 0.8, mid_y(stack)))
    arrow(ax, (stack[2] + 0.8, mid_y(stack)), (lm[0] - 0.8, mid_y(lm)))
    arrow(ax, (lm[2] + 0.8, mid_y(lm)), (logits[0] - 0.8, mid_y(logits)))
    arrow(
        ax,
        (logits[2] + 0.8, mid_y(logits)),
        (out[0] - 0.8, mid_y(out)),
    )

    # Inside-stack detail: Self-Attention and MLP (grouped, no micro-boxes)
    inner_margin_x = 2.0
    inner_margin_y = 3.0
    inner_y = stack[1] + inner_margin_y
    inner_h = (stack[3] - stack[1]) - inner_margin_y * 2

    attn_w = 20.0
    mlp_w = 20.0
    gap_inner = 2.0
    attn = draw_box(
        ax,
        (stack[0] + inner_margin_x, inner_y),
        attn_w,
        inner_h,
        "Self-Attention",
        title_size=24,
    )
    mlp = draw_box(
        ax,
        (attn[2] + gap_inner, inner_y),
        mlp_w,
        inner_h,
        "MLP",
        title_size=24,
    )

    # Caption for the whole stack placed above to avoid overlap
    ax.text(
        mid_x(stack),
        stack[3] + 0.8,
        "Transformer Layers",
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold",
    )

    # Inline labels for LoRA and Quantization (clean placement, no arrows)
    badge_w = 12.0
    badge_h = 3.0
    # LoRA badges just above the attention and MLP boxes
    draw_box(
        ax,
        (mid_x(attn) - badge_w / 2.0, attn[3] + 0.6),
        badge_w,
        badge_h,
        "LoRA",
        title_size=18,
        fc=(0.90, 0.98, 0.90),
    )
    draw_box(
        ax,
        (mid_x(mlp) - badge_w / 2.0, mlp[3] + 0.6),
        badge_w,
        badge_h,
        "LoRA",
        title_size=18,
        fc=(0.90, 0.98, 0.90),
    )
    # Quantization ribbon inside stack near the bottom
    ribbon_margin = 1.4
    ribbon_w = (stack[2] - stack[0]) - ribbon_margin * 2
    draw_box(
        ax,
        (stack[0] + ribbon_margin, stack[1] + ribbon_margin),
        ribbon_w,
        2.6,
        "4-bit quantization (NF4)",
        title_size=16,
        fc=(0.90, 0.94, 0.99),
    )

    # Title
    ax.set_xlim(0, out[2] + 4.0)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Symbolic-Math AI: Model Architecture",
        fontsize=26,
        pad=22,
    )

    out_dir = Path("trained_math_model_qwen_run2")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "model_architecture_diagram_clean.png"
    svg = out_dir / "model_architecture_diagram_clean.svg"
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(svg, bbox_inches="tight", pad_inches=0.2)
    print(str(png))
    print(str(svg))


if __name__ == "__main__":
    main()
