"""
Honest before/after eval: base model vs the GRPO-trained model on Countdown.

Reports three distinct tiers (the earlier eval conflated the last two):
  - format         : used the full <think>/<answer> structure
  - valid-numbers  : produced a valid expression using exactly the right numbers (partial or better)
  - exactly-solved : the expression actually hits the target (the real goal)

Run:  uv run python stage3_capstone/eval_compare.py
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown_task import correctness_reward, format_reward, generate_countdown, make_prompt

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED = "stage3_capstone/runs/final"
N = 24
N_NUMBERS = 2   # match the training curriculum


@torch.no_grad()
def eval_model(path, tokenizer, puzzles, device):
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()
    fmt = valid = solved = 0
    for numbers, target in puzzles:
        text = tokenizer.apply_chat_template(
            make_prompt(numbers, target), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        r = correctness_reward([gen], numbers=[numbers], target=[target])[0]
        fmt += 1 if format_reward([gen])[0] > 0 else 0
        valid += 1 if r > 0 else 0        # partial (0.2) or correct (1.0)
        solved += 1 if r >= 1.0 else 0    # strictly hits the target
    del model
    n = len(puzzles)
    return fmt / n, valid / n, solved / n


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    puzzles = [generate_countdown(n_numbers=N_NUMBERS) for _ in range(N)]
    print(f"Evaluating on {N} fixed puzzles (greedy decoding), device={device}\n")
    for name, path in [("BASE (untrained)", BASE), ("GRPO-trained", TRAINED)]:
        fmt, valid, solved = eval_model(path, tokenizer, puzzles, device)
        print(f"{name:>18}:  format={fmt:6.1%}   valid-numbers={valid:6.1%}   exactly-solved={solved:6.1%}")


if __name__ == "__main__":
    main()
