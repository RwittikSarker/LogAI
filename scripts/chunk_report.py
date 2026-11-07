"""Show how a log file is split, and verify the plan.

Usage:  python -m scripts.chunk_report samples/sample_attack.log
        python -m scripts.chunk_report samples/sample_attack.log --show 3
"""
from __future__ import annotations

from app import config  # noqa: F401  (MUST be first)

import argparse
import sys
from pathlib import Path

from app.chunker import DEFAULT_MAX_CHUNK_TOKENS, render_chunk, split_log
from app.ram_monitor import RamMonitor
from app.tokens import get_counter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--show", type=int, default=None, help="print the rendered text of this chunk index")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_file():
        sys.exit(f"{path} not found")
    counter = get_counter(args.model_dir)

    with RamMonitor(interval=0.1) as mon:
        chunks = split_log(path, counter, max_tokens=args.max_tokens)
        peak = mon.snapshot()["peak_mb"]

    print(f"counter: {counter.name} | budget per chunk: {args.max_tokens} tokens\n")
    print(f"{'idx':>3}  {'lines':<13} {'n':>4} {'tokens':>6} {'entries':>7}  flags")
    for c in chunks:
        flags = ("SPLIT " if c.split_entry else "") + ("OVER" if c.tokens > args.max_tokens else "")
        print(f"{c.index:>3}  {c.start_line}-{c.end_line:<{max(1, 12 - len(str(c.start_line)))}} {c.line_count:>4} "
              f"{c.tokens:>6} {c.entries:>7}  {flags}")

    # Verification 1: contiguous coverage. Verification 2: per-line sums vs whole-text recount.
    contiguous = all(b.start_line == a.end_line + 1 for a, b in zip(chunks, chunks[1:]))
    worst = 0.0
    for c in chunks:
        recount = counter.count(render_chunk(path, c))
        worst = max(worst, (recount - c.tokens) / max(c.tokens, 1))
    print(f"\nchunks: {len(chunks)} | lines: {chunks[-1].end_line if chunks else 0} | "
          f"tokens (sum): {sum(c.tokens for c in chunks)} | largest chunk: {max((c.tokens for c in chunks), default=0)}")
    print(f"contiguous coverage: {'OK' if contiguous else 'GAP!'} | "
          f"worst whole-text recount vs estimate: {100 * worst:+.1f}% | python RSS peak: {peak:.0f} MB")

    if args.show is not None:
        print(f"\n----- chunk {args.show} -----")
        print(render_chunk(path, chunks[args.show]))


if __name__ == "__main__":
    main()
