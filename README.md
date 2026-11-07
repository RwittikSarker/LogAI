# Local Offline Security Log Analyzer

An offline, resource-constrained AI-powered log analysis system designed to detect multi-stage security threats from raw server logs. The system runs locally using a **Qwen 3.5 0.8B** model operating strictly under a **4 GB RAM limit** and **2 vCPU threads**, without GPU acceleration.

---

## Technical Overview & Model Details

- **Core LLM:** `Qwen/Qwen3.5-0.8B`
- **Execution Mode:** Native `fp32` (Float32 precision)
- **Quantization/Conversion:** Converted locally from default `bf16` to `fp32` via `scripts/convert_fp32.py` to bypass slow CPU emulation on non-AVX-512/bf16 hardware.
- **Context Window Limit:** 2,048 tokens maximum per call.
- **Memory Footprint:** Peak RAM usage tracked at ~3.62 GB to ~3.70 GB (under the 4.0 GB hard limit).
- **Execution Speed:** ~6.1 tokens/sec decode, ~140 tokens/sec prefill on 2 threads.

---

## Log Splitting & Analysis Methodology

Large logs (e.g., ~20,000 tokens) exceed the model's single-call 2,048 context window. To solve this without hitting out-of-memory errors or missing long-term threats:

1. **Offset-Based Streaming Chunker (`app/chunker.py`):**
   - Reads the log file via disk offsets rather than loading the entire file into Python memory.
   - Divides log lines into chunks capped at ~900 tokens.
   - Prefixes every line with `L<no>|` for accurate line-level evidence tracking.

2. **Rolling State Ledger (`app/ledger.py`):**
   - Maintains cross-chunk continuity without relying on overlapping lines (which cause double-counting).
   - Passes a compact `context_update` summary from chunk to chunk stored in SQLite (`logs/analysis.db`).
   - Links multi-stage attacks over time (e.g., initial brute-force attempts on chunk 3 linked to a successful root compromise on chunk 18).

3. **Structured JSON Extraction (`app/parser.py`):**
   - Parses structured findings (attack type, severity, target IP/user, line evidence, action) with automatic fuzzy JSON repair for small model output quirks.

---

## Requirements & Setup

### Prerequisites
- Python 3.10+
- PyTorch (CPU version)
- Transformers
- Flask

### Installation Steps

1. **Clone/Setup Directory:**
   ```bash
   git clone <repository-url>
   cd security-log-analyzer
   ```

2. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install torch transformers flask
   ```

3. **Convert Model Checkpoint to Native FP32:**
   *(Ensure model files exist under `models/qwen3.5-0.8b`)*
   ```bash
   python scripts/convert_fp32.py
   ```

4. **Launch the Application:**
   ```bash
   python -m app.main
   ```

5. **Access the Web Interface:**
   Open `http://localhost:5000` in your browser, upload a log file, and click **Start Analysis**.

---

## System Architecture

```text
       [ Raw Log File ]
               │
      (app/chunker.py)  ── Offset-based chunking (~900 tokens)
               │
               ▼
     [ Analysis Pipeline ] ◄── (app/ledger.py) Passes context from previous chunk
               │
       (app/backend.py) ── Runs Qwen 0.8B (fp32, 2 CPU threads)
               │
       (app/parser.py)  ── Structured JSON parsing & fuzzy repair
               │
               ▼
      [ Flask Dashboard ] ── Real-time RAM monitor & Security Report UI
```

---

## Known Limits & Trade-offs

1. **Processing Time:**
   - Because generation runs purely on 2 CPU threads without GPU acceleration, processing large log files (~20k tokens / ~50 chunks) takes approximately 20 to 30 minutes total (~25–40 seconds per chunk).
2. **Deterministic Output Constraints:**
   - Inference uses low temperature (`temperature=0.1`) and greedy decoding to ensure deterministic analytical outputs.
   - Token budget for output generation is capped at `max_new_tokens=150` to maintain fast response times and avoid endless generation loops.
3. **RAM Upper Bound:**
   - Running background applications alongside the model might push system RAM near the 4 GB limit. The app includes a background RAM monitor (`ram_monitor.py`) to observe peak consumption.