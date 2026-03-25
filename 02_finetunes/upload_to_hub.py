"""Upload a merged model and its checkpoints to the HuggingFace Hub.

Usage:
    python upload_to_hub.py <model_dir> <repo_id> [--private]

The main branch gets the final merged model (top-level files).
Each checkpoint-* subdirectory is uploaded as a separate branch.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload model + checkpoints to HF Hub")
    parser.add_argument("model_dir", type=str, help="Path to the output directory (e.g. checkpoints/)")
    parser.add_argument("repo_id", type=str, help="HF repo id (e.g. user/model-name)")
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    api = HfApi()

    # Create the repo (no-op if it already exists)
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)

    # Upload the final merged model (top-level files only) to main branch
    print(f"Uploading final model to {args.repo_id} (main branch)...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        # Exclude checkpoint subdirectories from the main branch upload
        ignore_patterns=["checkpoint-*/**"],
    )
    print("Main branch done.")

    # Upload each checkpoint as a separate branch
    checkpoints = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    for ckpt in checkpoints:
        branch = ckpt.name  # e.g. "checkpoint-1755"
        print(f"Uploading {branch} as branch '{branch}'...")
        api.create_branch(repo_id=args.repo_id, branch=branch, exist_ok=True)
        api.upload_folder(
            folder_path=str(ckpt),
            repo_id=args.repo_id,
            revision=branch,
        )
        print(f"Branch {branch} done.")

    print(f"All uploads complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
