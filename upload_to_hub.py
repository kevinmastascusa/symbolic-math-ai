#!/usr/bin/env python3
"""
Upload the trained LoRA adapter folder to Hugging Face Hub.

Usage (after setting HF token via `huggingface-cli login` or HF_TOKEN env):
  python upload_to_hub.py [optional_adapter_dir]

Defaults to folder: trained_math_model_qwen_run2
Repo: kevinmastascusa/symbolic-math-qwen2p5-1p5b-lora
"""

import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, HfFolder


def get_hf_token() -> Optional[str]:
    # Prefer env var if present, else use cached token
    return os.environ.get("HF_TOKEN") or HfFolder.get_token()


def main() -> None:
    adapter_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "trained_math_model_qwen_run2"
    )
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        print(f"Adapter directory not found: {adapter_dir}")
        sys.exit(1)

    repo_id = "Kevinmastascusa/symbolic-math-qwen2p5-1p5b-lora"

    token = get_hf_token()
    api = HfApi(token=token) if token else HfApi()

    if token is None:
        print("No Hugging Face token found. Run one of these first:")
        print(
            "  - huggingface-cli login (recommended)\n"
            "  - set HF_TOKEN=hf_... (Windows cmd)"
        )
        # Proceeding without token will fail for private repos;
        # may work for public if anonymous allowed

    print(
        f"Creating (or updating) repo: {repo_id} (private=True)"
    )
    api.create_repo(repo_id, private=True, exist_ok=True)

    # Touch a timestamp file so there's always something to commit
    try:
        from datetime import datetime
        ts_path = adapter_dir / ".last_upload"
        ts_path.write_text(datetime.utcnow().isoformat() + "Z")
    except Exception:
        pass

    print(f"Uploading folder '{adapter_dir}' to {repo_id} ...")
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        commit_message="Upload LoRA adapter",
    )

    print("Done. View repo at: https://huggingface.co/" + repo_id)


if __name__ == "__main__":
    main()
