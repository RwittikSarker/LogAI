"""Download a model INTO the project folder (./models/<name>).

Usage:
    python -m scripts.download_model
    python -m scripts.download_model --repo Qwen/Qwen3.5-2B --name qwen3.5-2b

This is the only script allowed to touch the network.
"""
from __future__ import annotations

import os

os.environ["LOGAI_ALLOW_DOWNLOAD"] = "1"  # must be set before importing config

import argparse
import sys
from pathlib import Path

from app import config  # noqa: E402  (redirects all caches into the project)
from huggingface_hub import snapshot_download  # noqa: E402

# We only run text inference with transformers: skip formats we will never use.
IGNORE = ["*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot", "original/*"]


def folder_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and ".cache" not in p.parts:
            total += p.stat().st_size
    return total / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=config.DEFAULT_REPO_ID)
    ap.add_argument("--name", default=None, help="folder name under ./models")
    args = ap.parse_args()

    name = args.name or args.repo.split("/")[-1].lower()
    target = (config.MODELS_DIR / name).resolve()
    try:
        target.relative_to(config.ROOT)
    except ValueError:
        sys.exit(f"Refusing to download outside the project folder: {target}")

    print(f"Downloading {args.repo}\n  into {target}")
    snapshot_download(repo_id=args.repo, local_dir=str(target), ignore_patterns=IGNORE)

    weights = sorted(target.glob("*.safetensors"))
    if not weights:
        sys.exit("Download finished but no .safetensors files were found - check the repo id.")
    print(f"\nOK: {len(weights)} weight file(s), {folder_size_gb(target):.2f} GB total in {target}")


if __name__ == "__main__":
    main()
