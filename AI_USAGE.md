# AI Usage Log

## 1. Tools & Coding Assistants Used

- **Claude (Anthropic):** Used for initial project scoping, designing the offset-based log chunker (`chunker.py`), setting up the background RAM monitor daemon (`ram_monitor.py`), benchmarking scripts, and discovering the PyTorch lazy memory-mapping behavior.
- **Gemini (Google):** Used for integrating pipeline modules, building the Flask web dashboard (`main.py`), implementing robust JSON parsing and repair (`parser.py`), resolving debugging issues (RAM monitor wrappers, thread limits, `eos_token_id` generation hang fixes), and generating this documentation.

---

## 2. Token Budget Plan (2,048 Token Limit)

The application strictly respects the model's **2,048-token context window limit** per call, including prompt instructions, rolling context, log text, and model output.

- **System Instructions & Rolling Context:** ~300 tokens, carrying forward state across chunks to link multi-stage attacks such as brute-force followed by compromise.
- **Log Chunk Size (`max_tokens`):** ~900 tokens, parsed dynamically with line prefixes like `L<no>|`.
- **Model Output Generation (`max_new_tokens`):** Up to 150 tokens, capped to preserve fast response times and strict adherence to JSON schema limits.
- **Total Peak Per Call:** ~1,350 tokens, well within the 2,048-token hard cap.

---

## 3. Local Model Prompt Template & Evolution

The prompt template enforces structured analytical output and helps prevent hallucination or instruction injection from malicious log data:

```text
You are an expert security log analyst. Analyze the following log chunk for security threats (brute-force, SQL injection, scanning, rate anomalies, unauthorized access).

ROLLING CONTEXT FROM PREVIOUS CHUNKS:
{rolling_context}

CURRENT LOG CHUNK ({chunk_index}/{total_chunks}):
{chunk_text}

INSTRUCTIONS:
1. Identify any attacks or suspicious activities in this chunk.
2. Link evidence to previous rolling context if relevant.
3. Limit to max 3 findings. Keep reasons concise.
4. Output STRICT JSON format matching this schema:
{{
  "context_update": "Summary of active threats or state to carry forward",
  "findings": [
    {{
      "attack_type": "string",
      "severity": "Low|Medium|High|Critical",
      "target_id": "IP or username",
      "evidence": ["L12| ..."],
      "reason": "short explanation",
      "action": "recommended mitigation"
    }}
  ]
}}
JSON OUTPUT:
```

---

## 4. AI Mistakes Fixed & Design Adjustments

### The bf16 CPU Emulation Bottleneck

- **Initial AI Suggestion:** Use bf16 precision to save memory.
- **Error / Mistake:** Measurements showed that the target CPU lacks native bf16 instructions, so matmuls were emulated and generation slowed to ~1.84 tok/s, making a full run take ~85 minutes.
- **Correction:** Discovered PyTorch lazy memory-mapping behavior and created a one-time script (`convert_fp32.py`) to convert the checkpoint to native fp32, restoring speed to ~6 tok/s while staying under the 4 GB RAM cap (peak 3.62 GB).

### Chunk Overlap Double-Counting

- **Initial AI Suggestion:** Include a 2-line overlap between chunks to maintain continuity.
- **Error / Mistake:** The overlap caused lines to be counted twice in the SQLite state ledger counters.
- **Correction:** Removed chunk overlap entirely and relied instead on offset-based streaming and the SQLite rolling `context_update` ledger to carry continuity forward.

### Greedy Decoding Generation Hangs

- **Initial AI Suggestion:** Use default greedy generation (`do_sample=False`) for maximum analytical determinism.
- **Error / Mistake:** Without an explicit `eos_token_id`, smaller models could become trapped in repetitive token loops while formatting structured JSON, stalling the pipeline indefinitely.
- **Correction:** Explicitly configured `eos_token_id=self.tokenizer.eos_token_id` and tightened `max_new_tokens` to 150 to guarantee clean termination.

---

## 5. Prompt & Reply Log (Core Interactions)

### Interaction 1: Architecture & Resource Constraints (Claude Session)

- **User Prompt:** “Build a web app that uses a downloaded AI model to find possible attacks in server logs... Resource limit: Server RAM 4 GB total, CPU: 2 vCPU, no GPU, Log file: ~20,000 tokens, Model context: 2,048 tokens per call.”
- **Assistant Reply:** Outlined the local architecture plan: offset-based chunking with no extra log copies, a background RAM daemon (`ram_monitor.py`), local model download with restricted offline caching, and benchmark scripts.

### Interaction 2: Memory & Dtype Decision Gate (Claude Session)

- **User Prompt:** “Both results are in, and together they explain the memory behaviour... bf16 prefill runs at 26 tok/s and decode at 1.84 tok/s... fp32 is about 4x faster.”
- **Assistant Reply:** Analyzed benchmark results (3.70 GB peak RAM, ~60s per call), recommended creating `convert_fp32.py` for native fp32 checkpoint loading to eliminate duplicate memory-mapped pages, and confirmed that transformers on fp32 meets the go/no-go decision gate.

### Interaction 3: Pipeline & Chunker Integration (Gemini Session)

- **User Prompt:** Provide project details and the rest of the modules (`backend.py`, `parser.py`, `ledger.py`, `pipeline.py`, `main.py`) needed to run the app.
- **Assistant Reply:** Provided the complete modular Flask backend, JSON repair parser, SQLite ledger for cross-chunk attack linking (brute-force → compromise), and integrated the user's custom offset-based `chunker.py`.

### Interaction 4: Performance & Hang Diagnosis (Gemini Session)

- **User Prompt:** “How long is this gonna take? It says it'll take a minute, but it's been 20 minutes... 127.0.0.1 - - [07/Nov/2026 21:03:00] 'GET /api/ram HTTP/1.1' 200 - ... it's been almost an hour.”
- **Assistant Reply:** Diagnosed the greedy decoding hang caused by a missing `eos_token_id` and optimized `max_new_tokens` to 150, enabling smooth, non-blocking chunk processing and real-time terminal progress logging.
