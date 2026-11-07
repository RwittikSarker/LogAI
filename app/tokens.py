"""Token counting behind one tiny interface.

The chunker only needs `count(text) -> int`. It never imports transformers, so
it can be unit-tested with ApproxTokenCounter and does not care which inference
backend (transformers or a llama.cpp fallback) is used: it is handed the real
tokenizer of whichever model is active.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Protocol

from app import config

log = logging.getLogger("logai.tokens")


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    """Deterministic stand-in for tests and for machines without the model."""

    def __init__(self, chars_per_token: float = 3.0) -> None:
        self.chars_per_token = chars_per_token
        self.name = f"approx({chars_per_token:g} chars/token)"

    def count(self, text: str) -> int:
        return max(1, math.ceil(len(text) / self.chars_per_token))


class HFTokenCounter:
    """Exact counts from the model's own tokenizer (loaded in-process)."""

    def __init__(self, model_dir: Path) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.name = f"hf:{Path(model_dir).name}"

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


def get_counter(model_dir: Optional[Path] = None, allow_approx: bool = True) -> TokenCounter:
    model_dir = Path(model_dir) if model_dir else config.ACTIVE_MODEL_DIR
    if not model_dir.exists():
        # The tokenizer is identical in the bf16 download, so fall back to it.
        model_dir = config.DEFAULT_MODEL_DIR
    if model_dir.exists():
        return HFTokenCounter(model_dir)
    if not allow_approx:
        raise FileNotFoundError("No model folder found; run scripts/download_model.py")
    log.warning("No tokenizer found; using the approximate counter (not exact!).")
    return ApproxTokenCounter()
