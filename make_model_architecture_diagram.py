#!/usr/bin/env python3
"""
Generate a Model Architecture diagram (NOT a methods overview).

Focus: Tokenizer → Embedding → Transformer Stack (Self-Attention + MLP
with LoRA) → LM Head, with side components for Tree-of-Thoughts and
SymPy verification.

Outputs:
- trained_math_model_qwen_run2/model_architecture_diagram.png
- trained_math_model_qwen_run2/model_architecture_diagram.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

# Headless backend for non-interactive render
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    w: float,
    h: float,
    title: str,
    subtitle: str | None = None,
    *,
    fc=(0.96, 0.96, 0.96),
    ec=(0.25, 0.25, 0.25),
    title_size: int = 16,
    subtitle_size: int = 12,
    bold: bool = True,
) -> Tuple[float, float, float, float]:
    x0, y0 = xy
    patch = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        facecolor=fc,
        edgecolor=ec,
        zorder=1,
    )
    ax.add_patch(patch)

    cx = x0 + w / 2.0
    top = y0 + h - 0.35
    ax.text(
        cx,
        top,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold" if bold else None,
        zorder=2,
    )
    if subtitle:
        ax.text(
            cx,
            y0 + h - 1.0,
            subtitle,
            ha="center",
            va="top",
            fontsize=subtitle_size,
            zorder=2,
            wrap=True,
        )
    return (x0, y0, x0 + w, y0 + h)


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
    fig, ax = plt.subplots(figsize=(28, 16))
    ax.axis("off")

    # Horizontal pipeline baseline
    base_y = 50.0
    x = 2.0
    gap = 3.0

    # Left inputs
    tokens = box(
        ax,
        (x, base_y),
        10.0,
        6.0,
        "Input Text",
        "Tokenized with chat template",
    )
    x = tokens[2] + gap
    tok = box(
        ax,
        (x, base_y),
        10.5,
        6.0,
        "Tokenizer",
        "Qwen2.5 tokenizer (LaTeX-friendly)",
    )
    x = tok[2] + gap
    ids = box(ax, (x, base_y), 7.0, 6.0, "Token IDs", "int32 sequences")
    x = ids[2] + gap
    emb = box(
        ax,
        (x, base_y),
        12.0,
        6.0,
        "Embedding Layer",
        "Learned token + positional embeddings",
    )

    # Transformer stack mega box
    x = emb[2] + gap
    stack = box(
        ax,
        (x, base_y - 6.5),
        36.0,
        19.0,
        "Transformer Stack (N layers)",
        "Self-Attention + MLP per layer, residual connections, layer norms",
    )

    # Inside the stack: depict one representative layer with LoRA locations
    layer_x0 = stack[0] + 1.0
    layer_y0 = stack[1] + 2.0
    layer_w = 34.0
    layer_h = 14.5
    layer = box(
        ax,
        (layer_x0, layer_y0),
        layer_w,
        layer_h,
        "Representative Layer",
        "LoRA on q/k/v/o and MLP gate/up/down (4-bit QLoRA)",
        fc=(0.985, 0.985, 0.985),
    )

    # Attention and MLP sub-boxes
    attn = box(
        ax,
        (layer[0] + 1.0, layer[1] + 2.0),
        15.5,
        9.5,
        "Self-Attention",
        "q_proj, k_proj, v_proj, o_proj\nLoRA adapters injected",
    )
    mlp = box(
        ax,
        (attn[2] + 1.5, attn[1]),
        14.5,
        9.5,
        "MLP",
        "gate_proj, up_proj, down_proj\nLoRA adapters injected",
    )
    box(
        ax,
        (mlp[2] + 1.0, attn[1]),
        2.5,
        9.5,
        "Norms",
        "pre/post LN",
    )

    # LoRA note
    box(
        ax,
        (layer[0] + 1.0, layer[1] + 0.5),
        layer_w - 2.0,
        1.2,
        "LoRA config",
        "r=8, alpha=16, dropout=0.05; target: q/k/v/o, gate/up/down",
        fc=(0.94, 0.97, 0.94),
    )

    # Quantization note
    box(
        ax,
        (stack[0] + 1.0, stack[1] + 0.6),
        17.0,
        1.5,
        "Quantization",
        "4-bit NF4 (bitsandbytes) with bfloat16/float16 compute",
        fc=(0.94, 0.96, 0.99),
    )

    # Output head
    x = stack[2] + gap
    lm = box(
        ax,
        (x, base_y),
        12.0,
        6.0,
        "LM Head",
        "Linear projection to vocab logits",
    )
    x = lm[2] + gap
    logits = box(ax, (x, base_y), 8.0, 6.0, "Logits", "softmax decoding")
    x = logits[2] + gap
    text = box(
        ax,
        (x, base_y),
        10.0,
        6.0,
        "Generated Text",
        "answers/rationales",
    )

    # Main flow arrows
    def mid_y(b):
        return (b[1] + b[3]) / 2.0

    arrow(ax, (tokens[2] + 0.6, mid_y(tokens)), (tok[0] - 0.6, mid_y(tok)))
    arrow(ax, (tok[2] + 0.6, mid_y(tok)), (ids[0] - 0.6, mid_y(ids)))
    arrow(ax, (ids[2] + 0.6, mid_y(ids)), (emb[0] - 0.6, mid_y(emb)))
    arrow(ax, (emb[2] + 0.6, mid_y(emb)), (stack[0] - 0.6, mid_y(stack)))
    arrow(ax, (stack[2] + 0.6, mid_y(stack)), (lm[0] - 0.6, mid_y(lm)))
    arrow(ax, (lm[2] + 0.6, mid_y(lm)), (logits[0] - 0.6, mid_y(logits)))
    arrow(ax, (logits[2] + 0.6, mid_y(logits)), (text[0] - 0.6, mid_y(text)))

    # Side components: Tree-of-Thoughts controller and SymPy verification
    tot = box(
        ax,
        (lm[0] + 2.0, stack[1] - 10.0),
        20.0,
        6.5,
        "Tree-of-Thoughts Controller",
        "Iterative generation, scoring, best-path selection",
    )
    sym = box(
        ax,
        (tot[2] + 2.0, tot[1]),
        18.0,
        6.5,
        "SymPy Verifier",
        "Extracts/solves equations, checks final numerical/formal consistency",
    )

    # Arrows to/from side components
    arrow(
        ax,
        ((lm[0] + lm[2]) / 2.0, lm[1] - 0.6),
        ((tot[0] + tot[2]) / 2.0, tot[3] + 0.2),
    )
    arrow(
        ax,
        (((logits[0] + text[2]) / 2.0), text[1] - 0.6),
        ((sym[0] + sym[2]) / 2.0, sym[3] + 0.2),
    )

    # Title
    ax.set_xlim(0, text[2] + 2.0)
    ax.set_ylim(0, 90)
    ax.set_title(
        "Symbolic-Math AI: Model Architecture",
        fontsize=22,
        pad=18,
    )

    out = Path("trained_math_model_qwen_run2")
    out.mkdir(parents=True, exist_ok=True)
    png = out / "model_architecture_diagram.png"
    svg = out / "model_architecture_diagram.svg"
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(svg, bbox_inches="tight", pad_inches=0.2)
    print(str(png))
    print(str(svg))


if __name__ == "__main__":
    main()
