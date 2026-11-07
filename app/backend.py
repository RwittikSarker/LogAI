# app/backend.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app import config

class LLMBackend:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or config.ACTIVE_MODEL_DIR
        print(f"Loading model from {self.model_dir} (dtype=float32) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        self.model.eval()
        print("Model loaded successfully.")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        
        # Ensure total tokens stay within 2048 budget
        if input_len + max_new_tokens > 2048:
            max_new_tokens = max(50, 2048 - input_len)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][input_len:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response_text