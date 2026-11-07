# app/parser.py
import json
import re

def parse_llm_json(response_text: str) -> dict:
    # 1. Try direct JSON load
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find the first '{' and last '}'
    brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if brace_match:
        raw_json = brace_match.group(0)
        # Attempt common repair: remove trailing commas before closing braces/brackets
        repaired = re.sub(r',\s*([\]}])', r'\1', raw_json)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # 4. Fallback safe structure if parsing completely fails
    return {
        "findings": [],
        "context_update": "Failed to parse model JSON response.",
        "raw_response": response_text
    }