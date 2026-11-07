"""Continuous RAM monitor.

A daemon thread samples the resident memory (RSS) of this process, plus any
child processes (so a llama.cpp server launched by us would be counted too),
every `interval` seconds. It tracks:

  * current and peak RSS
  * a resettable "window" peak (e.g. peak during one model call)
  * named marks ("model_loaded", ...) so we can see where memory goes
  * an optional CSV time series in ./logs for later inspection

It only OBSERVES. Python cannot enforce a memory cap, so it warns (log + optional
callback) when usage crosses `warn_fraction` of the limit.

RSS counts memory-mapped model pages that are resident, which is also what a
cgroup limit would count, so it is the right number to compare against 4 GB.
Peaks shorter than `interval` can be missed; use a small interval when
benchmarking.
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import psutil

from app import config

log = logging.getLogger("logai.ram")
MB = 1024 * 1024


class RamMonitor:
    def __init__(
        self,
        limit_gb: float = config.RAM_LIMIT_GB,
        interval: float = 0.5,
        warn_fraction: float = 0.85,
        csv_path: Optional[Path] = None,
        include_children: bool = True,
        on_warn: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.limit_bytes = int(limit_gb * 1024**3)
        self.warn_bytes = int(self.limit_bytes * warn_fraction)
        self.interval = interval
        self.include_children = include_children
        self.on_warn = on_warn

        self._proc = psutil.Process()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._csv_path = Path(csv_path) if csv_path else None
        self._csv_file = None
        self._csv_writer = None

        self._t0 = time.monotonic()
        self._current = 0
        self._peak = 0
        self._window_peak = 0
        self._samples = 0
        self._warned = False
        self._marks: dict[str, dict] = {}

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> "RamMonitor":
        if self._thread and self._thread.is_alive():
            return self
        self._t0 = time.monotonic()
        self._stop.clear()
        if self._csv_path:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["elapsed_s", "rss_mb", "peak_mb"])
        self._sample()
        self._thread = threading.Thread(target=self._run, name="ram-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()
        with self._lock:
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def __enter__(self) -> "RamMonitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- sampling --------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:  # monitoring must never kill the app
                log.exception("RAM sample failed")

    def _read_rss(self) -> int:
        total = self._proc.memory_info().rss
        if self.include_children:
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        return total

    def _sample(self) -> int:
        rss = self._read_rss()
        crossed = False
        with self._lock:
            self._current = rss
            self._samples += 1
            self._peak = max(self._peak, rss)
            self._window_peak = max(self._window_peak, rss)
            if rss >= self.warn_bytes and not self._warned:
                self._warned = crossed = True
            elif rss < self.warn_bytes * 0.95:
                self._warned = False  # re-arm after dropping back down
            if self._csv_writer:
                self._csv_writer.writerow(
                    [f"{time.monotonic() - self._t0:.2f}", f"{rss / MB:.1f}", f"{self._peak / MB:.1f}"]
                )
                self._csv_file.flush()
        if crossed:
            snap = self.snapshot()
            log.warning(
                "RAM %.0f MB is above %.0f%% of the %.0f MB limit",
                snap["rss_mb"], 100 * self.warn_bytes / self.limit_bytes, snap["limit_mb"],
            )
            if self.on_warn:
                try:
                    self.on_warn(snap)
                except Exception:
                    log.exception("on_warn callback failed")
        return rss

    # ---- public API ------------------------------------------------------
    def mark(self, label: str) -> dict:
        """Sample right now and remember the reading under `label`."""
        self._sample()
        with self._lock:
            entry = {"rss_mb": round(self._current / MB, 1), "peak_mb": round(self._peak / MB, 1)}
            self._marks[label] = entry
        return entry

    def reset_window(self) -> float:
        """Return the peak (MB) since the last reset, then start a new window."""
        self._sample()
        with self._lock:
            peak = self._window_peak
            self._window_peak = self._current
        return round(peak / MB, 1)

    def snapshot(self) -> dict:
        with self._lock:
            cur, peak, win = self._current, self._peak, self._window_peak
            samples, marks = self._samples, dict(self._marks)
        vm = psutil.virtual_memory()
        if cur >= self.limit_bytes:
            status = "exceeded"
        elif cur >= self.warn_bytes:
            status = "warn"
        else:
            status = "ok"
        return {
            "rss_mb": round(cur / MB, 1),
            "peak_mb": round(peak / MB, 1),
            "window_peak_mb": round(win / MB, 1),
            "limit_mb": round(self.limit_bytes / MB, 1),
            "used_pct_of_limit": round(100 * cur / self.limit_bytes, 1),
            "peak_exceeded_limit": peak > self.limit_bytes,
            "status": status,
            "system_available_mb": round(vm.available / MB, 1),
            "elapsed_s": round(time.monotonic() - self._t0, 1),
            "samples": samples,
            "marks": marks,
        }
