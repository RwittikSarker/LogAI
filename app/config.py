"""Central configuration.

IMPORT THIS MODULE FIRST in every entry point, before torch / transformers /
huggingface_hub. Importing it has a side effect on purpose: it redirects every
cache and temp directory into the project folder, so nothing is written to the
user's home directory, and it switches the HF libraries to offline mode.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = ROOT / ".cache"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"

DEFAULT_REPO_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_MODEL_DIR = MODELS_DIR / "qwen3.5-0.8b"  # as downloaded (bf16 checkpoint)

# The model the app actually runs: native fp32 copy made by scripts/convert_fp32.py
# (this CPU has no bf16 instructions; see AI_USAGE.md section 7 for the numbers).
ACTIVE_MODEL_DIR = MODELS_DIR / "qwen3.5-0.8b-fp32"
ACTIVE_DTYPE = "float32"
SAMPLES_DIR = ROOT / "samples"  # small committed example logs

# The assignment's limits.
RAM_LIMIT_GB = 4.0
CPU_THREADS = int(os.environ.get("LOGAI_THREADS", "2"))  # target machine: 2 vCPU
PROJECT_ROOT = ROOT

def ensure_dirs() -> None:
    for d in (CACHE_DIR, MODELS_DIR, DATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _pin_everything_to_project() -> None:
    ensure_dirs()
    hf_home = CACHE_DIR / "huggingface"
    redirects = {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_home / "hub",
        "HF_XET_CACHE": hf_home / "xet",
        "TORCH_HOME": CACHE_DIR / "torch",
        "XDG_CACHE_HOME": CACHE_DIR / "xdg",
        "TRITON_CACHE_DIR": CACHE_DIR / "triton",
        "TMPDIR": CACHE_DIR / "tmp",
    }
    for key, path in redirects.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)  # override, never setdefault
    tempfile.tempdir = str(redirects["TMPDIR"])  # tempfile caches its choice

    # Thread settings must be in the environment before torch is imported.
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Offline by default: proves the app works without network after setup.
    # Only scripts/download_model.py sets LOGAI_ALLOW_DOWNLOAD=1.
    if os.environ.get("LOGAI_ALLOW_DOWNLOAD") == "1":
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_pin_everything_to_project()
