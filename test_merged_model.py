#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick test for merged full model")
    p.add_argument(
        "--repo",
        type=str,
        default="Kevinmastascusa/qwen2p5-math-1p5b-merged",
        help="HF repo or local dir for the merged model",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Solve 2x+5=13",
        help="Prompt to test",
    )
    p.add_argument(
        "--quant4bit",
        action="store_true",
        help="Load model in 4-bit (requires bitsandbytes)",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading tokenizer: {args.repo}")
    tok = AutoTokenizer.from_pretrained(args.repo, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        if args.quant4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb
        else:
            model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(args.repo, **model_kwargs)

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Running generation on:", device)
    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    print("=== Output ===")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()


