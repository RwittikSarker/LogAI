"""One-time: write a native float32 copy of the model to ./models/<name>-fp32.

Why: this CPU has no bf16 instructions, so bf16 is ~4x slower than fp32. But
asking transformers for fp32 from a bf16 checkpoint leaves the memory-mapped
bf16 pages resident next to the converted copy (~1.5 GB wasted). A checkpoint
that is already fp32 on disk is mapped directly, with no duplicate.

This script itself peaks near 5 GB (it does the slow conversion). That is fine:
it is a setup step, not part of the app.

Usage:
    python -m scripts.convert_fp32
    python -m scripts.benchmark --model-dir models/qwen3.5-0.8b-fp32 --dtype float32
"""
from __future__ import annotations

from app import config  # noqa: F401  (MUST be first)

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(config.DEFAULT_MODEL_DIR))
    ap.add_argument("--dst", default=str(config.MODELS_DIR / "qwen3.5-0.8b-fp32"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing destination")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst).resolve()
    try:
        dst.relative_to(config.ROOT)
    except ValueError:
        sys.exit(f"Refusing to write outside the project folder: {dst}")
    if not src.exists():
        sys.exit(f"{src} not found. Run:  python -m scripts.download_model")
    if dst.exists() and any(dst.glob("*.safetensors")) and not args.force:
        sys.exit(f"{dst} already has weights. Use --force to overwrite.")

    print(f"Loading {src} as float32 ...")
    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, low_cpu_mem_usage=True)
    print(f"  loaded in {time.perf_counter() - t:.1f}s, dtype={model.dtype}")

    print(f"Saving to {dst} ...")
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(dst)

    files = [p for p in dst.iterdir() if p.is_file()]
    total_gb = sum(p.stat().st_size for p in files) / 1024**3
    print(f"\nOK: {len(files)} files, {total_gb:.2f} GB in {dst}")
    print("Next:  python -m scripts.benchmark --model-dir", dst, "--dtype float32")


if __name__ == "__main__":
    main()
