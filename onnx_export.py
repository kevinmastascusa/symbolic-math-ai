#!/usr/bin/env python3
"""
Export a trained HF CausalLM model to ONNX and optimize it for CUDA with
ONNX Runtime via Optimum.

Usage:
  python onnx_export.py \
    --model_dir ./trained_model_mixtral \
    --onnx_dir ./onnx_model_mixtral

Notes:
- Requires: optimum[onnxruntime], onnxruntime-gpu, onnx
- Works with decoder-only models inheriting AutoModelForCausalLM
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import OptimizationConfig


def export_to_onnx(
    model_dir: str,
    onnx_dir: str,
    fp16: bool = True,
    optimize: bool = True,
) -> None:
    model_dir_path = Path(model_dir)
    onnx_dir_path = Path(onnx_dir)
    onnx_dir_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir_path)

    # Export to ONNX and load ORT model
    model = ORTModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir_path,
        export=True,
        use_io_binding=True,
        file_name="model.onnx",
        provider=(
            "CUDAExecutionProvider" if fp16 else "CPUExecutionProvider"
        ),
    )

    if optimize:
        opt_config = OptimizationConfig(optimization_level=99)
        model.optimize(opt_config)

    # Save ONNX + tokenizer
    model.save_pretrained(onnx_dir_path)
    tokenizer.save_pretrained(onnx_dir_path)

    print(f"Exported and saved ONNX model to: {onnx_dir_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export HF CausalLM to ONNX for CUDA with Optimum"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained HF model directory",
    )
    parser.add_argument(
        "--onnx_dir",
        type=str,
        required=True,
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Disable fp16; use CPU provider"
    )
    parser.add_argument(
        "--no_optimize", action="store_true", help="Disable graph optimization"
    )
    args = parser.parse_args()

    export_to_onnx(
        model_dir=args.model_dir,
        onnx_dir=args.onnx_dir,
        fp16=not args.no_fp16,
        optimize=not args.no_optimize,
    )


if __name__ == "__main__":
    main()

