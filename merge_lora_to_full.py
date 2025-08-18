#!/usr/bin/env python3
"""
Merge a LoRA adapter into the base model, save a standalone full model,
and upload it to the Hugging Face Hub.

Usage (defaults are set for this project):
  python merge_lora_to_full.py \
    --base Qwen/Qwen2.5-Math-1.5B \
    --adapter Kevinmastascusa/symbolic-math-qwen2p5-1p5b-lora \
    --out_dir qwen2p5-math-1p5b-merged \
    --repo_id Kevinmastascusa/qwen2p5-math-1p5b-merged \
    [--private]

Requires environment token: HF_TOKEN or HUGGING_FACE_HUB_TOKEN
"""

import argparse
import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi, HfFolder


def get_hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or HfFolder.get_token()
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model"
    )
    p.add_argument(
        "--base",
        default="Qwen/Qwen2.5-Math-1.5B",
        type=str,
        help="Base model id",
    )
    p.add_argument(
        "--adapter",
        default="Kevinmastascusa/symbolic-math-qwen2p5-1p5b-lora",
        type=str,
        help="LoRA adapter repo or local folder",
    )
    p.add_argument(
        "--out_dir",
        default="qwen2p5-math-1p5b-merged",
        type=str,
        help="Local output folder",
    )
    p.add_argument(
        "--repo_id",
        default="Kevinmastascusa/qwen2p5-math-1p5b-merged",
        type=str,
        help="New Hub repo id for merged model",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create Hub repo as private",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = get_hf_token()
    if token is None:
        print("No HF token found. Set HF_TOKEN or run huggingface-cli login.")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer tokenizer from adapter repo to preserve exact vocab and special
    # tokens
    print(f"Loading tokenizer from '{args.adapter}' ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter, trust_remote_code=True
    )

    print(
        f"Loading base model '{args.base}' on CPU (this can take a while) ..."
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )

    # IMPORTANT: Align base model vocab with adapter tokenizer BEFORE
    # loading the adapter to avoid size-mismatch errors on embed_tokens
    # and lm_head when the adapter saved these modules (e.g., via
    # modules_to_save).
    try:
        target_vocab_size = len(tokenizer)
        if getattr(base.config, "vocab_size", None) != target_vocab_size:
            base.resize_token_embeddings(target_vocab_size)
            base.config.vocab_size = target_vocab_size
    except Exception:
        # Proceed even if resizing is a no-op for this architecture
        pass

    print(f"Attaching LoRA adapter from '{args.adapter}' and merging ...")
    lora = PeftModel.from_pretrained(base, args.adapter)
    merged = lora.merge_and_unload()

    # Align embeddings and config with tokenizer
    try:
        merged.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    try:
        merged.config.vocab_size = len(tokenizer)
        if tokenizer.pad_token_id is not None:
            merged.config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            merged.config.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    print(f"Saving merged model to '{out_dir}' ...")
    merged.save_pretrained(
        out_dir,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(out_dir)

    print(
        f"Uploading to Hub repo '{args.repo_id}' (private={args.private}) ..."
    )
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        commit_message="Upload merged full model",
    )
    repo_url = f"https://huggingface.co/{args.repo_id}"
    print("Done. View at:", repo_url)


if __name__ == "__main__":
    main()
