"""Benchmark: can this machine run the model within our limits?

Measures, for one model + dtype:
  * RAM after loading and peak RAM (via RamMonitor, sampled every 0.1 s)
  * prefill speed (prompt processing) at a realistic prompt size
  * decode speed (tokens/second)
  * the estimated time of one analysis call and of a whole 20k-token log

Usage (from the project root, venv active):
    python -m scripts.benchmark                    # bfloat16, 2 threads
    python -m scripts.benchmark --dtype float32    # compare
"""
from __future__ import annotations

from app import config  # noqa: F401  (MUST be first: pins caches, threads, offline)

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.ram_monitor import RamMonitor

# Planned per-call budget (see AI_USAGE.md): ~1500 prompt tokens, ~300 output.
PLAN_PROMPT_TOKENS = 1500
PLAN_OUTPUT_TOKENS = 300
PLAN_CALLS = 23
GATE_SECONDS_PER_CALL = 90

TEMPLATES = [
    '203.0.113.{x} - - [07/Nov/2026:10:{t:02d}:11 +0000] "GET /index.php?id={n} HTTP/1.1" 200 512',
    '198.51.100.{x} - - [07/Nov/2026:10:{t:02d}:12 +0000] "POST /login HTTP/1.1" 401 128',
    'Nov  7 10:{t:02d}:13 web01 sshd[{n}]: Failed password for admin from 192.0.2.{x} port 4{t:02d}2 ssh2',
    '10.0.0.{x} - - [07/Nov/2026:10:{t:02d}:14 +0000] "GET /static/app.js HTTP/1.1" 304 0',
]


def cpu_info() -> dict:
    model, flags = "unknown", set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and model == "unknown":
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    wanted = ["avx2", "avx512f", "avx512_bf16", "amx_bf16", "amx_tile", "avx512_vnni"]
    return {
        "model": model,
        "logical_cpus": __import__("os").cpu_count(),
        "flags": {w: (w in flags) for w in wanted},
    }


def build_prompt_ids(tokenizer, n_tokens: int):
    lines = [
        f"L{i}|" + TEMPLATES[i % len(TEMPLATES)].format(n=i, x=i % 250 + 1, t=i % 60)
        for i in range(1, n_tokens // 10 + 60)
    ]
    ids = tokenizer("\n".join(lines), return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] < n_tokens:
        raise RuntimeError("Synthetic prompt too short; increase the line count.")
    return ids[:, :n_tokens]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=str(config.DEFAULT_MODEL_DIR))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--threads", type=int, default=config.CPU_THREADS)
    ap.add_argument("--prompt-tokens", type=int, default=PLAN_PROMPT_TOKENS)
    ap.add_argument("--new-tokens", type=int, default=48, help="tokens generated to time decoding")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"{model_dir} not found. Run:  python -m scripts.download_model")

    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    cpu = cpu_info()
    print(f"CPU: {cpu['model']} ({cpu['logical_cpus']} logical CPUs; using {args.threads} threads)")
    print("CPU flags:", ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in cpu["flags"].items()))
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | python {platform.python_version()}")

    result: dict = {"dtype": args.dtype, "threads": args.threads, "cpu": cpu}
    csv_path = config.LOGS_DIR / f"ram_benchmark_{args.dtype}.csv"

    with RamMonitor(interval=0.1, csv_path=csv_path) as mon:
        mon.mark("start")

        tok = AutoTokenizer.from_pretrained(model_dir)
        mon.mark("tokenizer_loaded")

        print(f"\nLoading model ({args.dtype}) ...")
        t = time.perf_counter()
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, low_cpu_mem_usage=True)
        except Exception as exc:
            print(f"\nModel load failed: {exc!r}")
            print("If this says the architecture is unknown, upgrade transformers (pip install -U transformers).")
            raise
        model.eval()
        load_s = time.perf_counter() - t
        n_params = sum(p.numel() for p in model.parameters())
        param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        mon.mark("model_loaded")
        print(f"  loaded in {load_s:.1f}s | {n_params / 1e9:.2f}B params | {param_gb:.2f} GB of weights")

        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def generate(ids, new_tokens: int) -> float:
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=new_tokens,
                    min_new_tokens=new_tokens,  # don't stop early on EOS
                    do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=pad,
                )
            return time.perf_counter() - t0

        print("Warm-up ...")
        generate(build_prompt_ids(tok, 32), 4)
        mon.mark("after_warmup")

        ids = build_prompt_ids(tok, args.prompt_tokens)
        print(f"Prefill: {args.prompt_tokens} prompt tokens + 1 new token ...")
        prefill_s = generate(ids, 1)
        mon.mark("after_prefill")
        print(f"  {prefill_s:.1f}s  ({args.prompt_tokens / prefill_s:.0f} prompt tok/s)")

        print(f"Decode: same prompt + {args.new_tokens} new tokens ...")
        total_s = generate(ids, args.new_tokens)
        mon.mark("after_decode")
        decode_s = max(total_s - prefill_s, 1e-6)
        decode_tps = (args.new_tokens - 1) / decode_s
        print(f"  {decode_tps:.2f} tokens/s")

        snap = mon.snapshot()

    est_call = prefill_s * (PLAN_PROMPT_TOKENS / args.prompt_tokens) + PLAN_OUTPUT_TOKENS / decode_tps
    est_total_min = est_call * PLAN_CALLS / 60
    peak_gb = snap["peak_mb"] / 1024

    result.update(
        load_seconds=round(load_s, 1), params_billion=round(n_params / 1e9, 3), weights_gb=round(param_gb, 2),
        prefill_seconds=round(prefill_s, 2), prefill_tok_per_s=round(args.prompt_tokens / prefill_s, 1),
        decode_tok_per_s=round(decode_tps, 2), est_seconds_per_call=round(est_call, 1),
        est_minutes_full_log=round(est_total_min, 1), ram=snap,
    )
    out_json = config.LOGS_DIR / f"benchmark_{args.dtype}.json"
    out_json.write_text(json.dumps(result, indent=2))

    print("\n================ RESULT ================")
    print(f"RAM after load : {snap['marks']['model_loaded']['rss_mb']:.0f} MB")
    print(f"RAM peak       : {snap['peak_mb']:.0f} MB  ({peak_gb:.2f} GB of {config.RAM_LIMIT_GB:.0f} GB limit)")
    print(f"One call ({PLAN_PROMPT_TOKENS} in / {PLAN_OUTPUT_TOKENS} out): ~{est_call:.0f}s")
    print(f"Whole log ({PLAN_CALLS} calls): ~{est_total_min:.0f} min")
    print(f"Saved: {out_json}  and  {csv_path}")

    if peak_gb > 0.85 * config.RAM_LIMIT_GB:
        print("WARNING: peak RAM is above 85% of the 4 GB limit.")
    if est_call > GATE_SECONDS_PER_CALL:
        print(f"WARNING: a call takes more than {GATE_SECONDS_PER_CALL}s - revisit budgets (or the llama.cpp fallback).")
    if not (cpu["flags"]["avx512_bf16"] or cpu["flags"]["amx_bf16"]) and args.dtype == "bfloat16":
        print("NOTE: this CPU has no bf16 instructions; compare with --dtype float32 (watch the RAM peak).")


if __name__ == "__main__":
    main()
