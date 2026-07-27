"""
Stage 3 smoke test: confirm we can load Qwen2.5-0.5B-Instruct on the Apple GPU (MPS)
and generate text, BEFORE building GRPO training on top of it.

Run:  uv run python stage3_capstone/smoke_test.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL} on '{device}' (first run downloads ~1 GB)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
print(f"Loaded. Parameter count: {model.num_parameters() / 1e6:.0f}M")

# Build a chat-formatted prompt using the model's own chat template.
messages = [
    {"role": "system", "content": "You are a helpful assistant. Think briefly, then answer."},
    {"role": "user", "content": "Using the numbers 3, 7, and 8 exactly once each with + - * /, "
                                "write one expression that equals 24."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)

# Decode only the newly generated tokens (skip the prompt we fed in).
generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\n--- Model output ---")
print(generated)
print("\nSmoke test passed: model loads and generates on this machine.")
