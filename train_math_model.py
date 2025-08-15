#!/usr/bin/env python3
"""
Standalone training script for Symbolic Math AI (PyTorch-only).

- Uses preprocessed datasets from `Dataset/`
- Trains a math-focused base model (default: Qwen/Qwen2.5-Math-1.5B)
- Evaluates token-loss and a simple exact-match (EM) metric
-
Run:
  python train_math_model.py \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --output_dir ./trained_math_model_qwen \
    --batch_size 1 --epochs 2 --max_length 512 \
    --max_samples_per_dataset 1000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch

from symbolic_math_ai import (
    TrainingConfig,
    MathModelTrainer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Symbolic Math AI (PyTorch)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HF model id to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_math_model_qwen",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=1000,
        help="Cap samples per dataset",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients to simulate larger batch with less GPU memory",
    )
    parser.add_argument(
        "--quantize_4bit",
        action="store_true",
        help="Enable 4-bit loading via bitsandbytes to reduce VRAM",
    )
    parser.add_argument(
        "--no_grad_ckpt",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA even if quantization is enabled",
    )
    return parser.parse_args()


def extract_final_answer(text: str) -> str:
    import re
    matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    return matches[-1] if matches else text.strip().lower()


def compute_exact_match(trainer: MathModelTrainer, n: int = 200) -> Dict[str, Any]:
    _, val_dataset = trainer.prepare_datasets()
    model = trainer.model
    tok = trainer.tokenizer
    model.eval()
    em_count = 0
    total = 0

    with torch.no_grad():
        for i in range(min(n, len(val_dataset))):
            item = val_dataset[i]
            inp = item["input_ids"].unsqueeze(0).to(model.device)
            attn = item["attention_mask"].unsqueeze(0).to(model.device)
            # Temporarily enable cache for generation to save compute
            prev_use_cache = getattr(model.config, "use_cache", True)
            if hasattr(model, "config"):
                model.config.use_cache = True
            gen = model.generate(
                input_ids=inp,
                attention_mask=attn,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.eos_token_id,
            )
            if hasattr(model, "config"):
                model.config.use_cache = prev_use_cache
            out = tok.decode(gen[0], skip_special_tokens=True)
            pred = extract_final_answer(out)
            gold_ids = item["labels"]
            gold_txt = tok.decode(
                gold_ids[gold_ids != -100], skip_special_tokens=True
            )
            gold = extract_final_answer(gold_txt)
            em_count += int(pred == gold)
            total += 1

    return {"exact_match": (em_count / max(total, 1))}


def main() -> None:
    args = parse_args()

    cfg = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_length=args.max_length,
        max_samples_per_dataset=args.max_samples_per_dataset,
        eval_steps=200,
        save_steps=200,
        use_quantization=args.quantize_4bit,
        use_gradient_checkpointing=(not args.no_grad_ckpt),
        use_lora=(not args.no_lora),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        tot_max_depth=3,
        tot_max_children=3,
    )

    trainer = MathModelTrainer(cfg)
    model = trainer.train_model()
    if model is None:
        raise RuntimeError("Training did not return a model instance")
    eval_res = trainer.evaluate_model(model)

    em = compute_exact_match(trainer, n=50)

    out_dir = Path(args.output_dir)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"eval": eval_res, "exact_match": em}, f, indent=2)

    print("Saved metrics to:", metrics_path)
    print("Model saved to:", out_dir)


if __name__ == "__main__":
    main()


