# app/pipeline.py
import os
from app.chunker import split_log, render_chunk
from app.backend import LLMBackend
from app.parser import parse_llm_json
from app.ledger import AnalysisLedger

class ModelTokenCounterAdapter:
    """Adapts the backend tokenizer to the TokenCounter protocol expected by split_log."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

PROMPT_TEMPLATE = """You are an expert security log analyst. Analyze the following log chunk for security threats (brute-force, SQL injection, scanning, rate anomalies, unauthorized access).

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
JSON OUTPUT:"""

class AnalysisPipeline:
    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.ledger = AnalysisLedger()
        self.counter = ModelTokenCounterAdapter(backend.tokenizer)

    def run(self, log_file_path: str, progress_callback=None) -> dict:
        self.ledger.clear()
        
        chunks = split_log(log_file_path, counter=self.counter, max_tokens=900)
        total_chunks = len(chunks)
        print(f"\n[+] Starting analysis: {total_chunks} chunks total.")

        for chunk in chunks:
            current_idx = chunk.index + 1
            pct = (current_idx / total_chunks) * 100
            print(f"-> Processing chunk {current_idx}/{total_chunks} ({pct:.1f}%)...")

            if progress_callback:
                progress_callback(current_idx, total_chunks)

            rolling_context = self.ledger.get_context()
            chunk_text = render_chunk(log_file_path, chunk)

            prompt = PROMPT_TEMPLATE.format(
                rolling_context=rolling_context,
                chunk_index=current_idx,
                total_chunks=total_chunks,
                chunk_text=chunk_text
            )

            response_text = self.backend.generate(prompt, max_new_tokens=150)
            parsed = parse_llm_json(response_text)

            new_context = parsed.get("context_update", rolling_context)
            self.ledger.set_context(new_context)

            for finding in parsed.get("findings", []):
                self.ledger.add_finding(finding, chunk_index=current_idx)

        print("[+] Analysis complete successfully!")
        final_findings = self.ledger.get_all_findings()
        return {
            "total_chunks_processed": total_chunks,
            "findings": final_findings
        }