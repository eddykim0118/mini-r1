"""
Stage 3, Step 3: GRPO training on Qwen2.5-0.5B for the Countdown game.

This is where all three stages converge. TRL's GRPOTrainer runs the exact loop you
hand-coded in Stage 2 — sample a GROUP of completions per prompt, score each with our
verifier, compute group-relative advantage, and update with a clipped objective + KL leash.

Smoke test (just prove the loop runs):
    uv run python stage3_capstone/train_grpo.py --max-steps 3 --eval-size 6

A longer real run (kick off and let it cook):
    uv run python stage3_capstone/train_grpo.py --max-steps 300 --eval-size 30
"""

import argparse
import os

# Apple-GPU safety valve: run any MPS-unsupported op on CPU instead of crashing the job.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from countdown_task import (
    correctness_reward,
    format_reward,
    generate_countdown,
    make_prompt,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_dataset(n: int, n_numbers: int) -> Dataset:
    """A dataset of Countdown puzzles. `numbers`/`target` ride along for the reward funcs."""
    rows = {"prompt": [], "numbers": [], "target": []}
    for _ in range(n):
        numbers, target = generate_countdown(n_numbers=n_numbers)
        rows["prompt"].append(make_prompt(numbers, target))
        rows["numbers"].append(numbers)
        rows["target"].append(target)
    return Dataset.from_dict(rows)


@torch.no_grad()
def evaluate(model, tokenizer, puzzles, device, max_new_tokens=160):
    """Greedy-decode one answer per puzzle; report correctness and format rates."""
    model.eval()
    n_correct = n_format = 0
    for numbers, target in puzzles:
        text = tokenizer.apply_chat_template(
            make_prompt(numbers, target), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        n_format += 1 if format_reward([gen])[0] > 0 else 0
        n_correct += 1 if correctness_reward([gen], numbers=[numbers], target=[target])[0] > 0 else 0
    model.train()
    return n_correct / len(puzzles), n_format / len(puzzles)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--eval-size", type=int, default=20)
    p.add_argument("--n-numbers", type=int, default=3)         # puzzle difficulty (2 = curriculum)
    p.add_argument("--num-generations", type=int, default=4)   # the "group" G
    p.add_argument("--batch-size", type=int, default=4)        # must be divisible by G
    p.add_argument("--max-completion", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04)         # KL leash toward reference model
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)

    # Fixed eval puzzles so the before/after comparison is apples-to-apples.
    eval_puzzles = [generate_countdown(n_numbers=args.n_numbers) for _ in range(args.eval_size)]

    print("\nEvaluating BEFORE training...")
    acc0, fmt0 = evaluate(model, tokenizer, eval_puzzles, device)
    print(f"  before: correctness={acc0:.1%}  format={fmt0:.1%}")

    config = GRPOConfig(
        output_dir="stage3_capstone/runs",
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch_size,
        max_completion_length=args.max_completion,
        learning_rate=args.lr,
        beta=args.beta,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        args=config,
        train_dataset=build_dataset(args.n_train, args.n_numbers),
        processing_class=tokenizer,
    )

    print("\nTraining...")
    trainer.train()

    print("\nEvaluating AFTER training...")
    acc1, fmt1 = evaluate(model, tokenizer, eval_puzzles, device)
    print("\n===== BEFORE -> AFTER =====")
    print(f"  correctness: {acc0:.1%} -> {acc1:.1%}")
    print(f"  format:      {fmt0:.1%} -> {fmt1:.1%}")

    save_dir = "stage3_capstone/runs/final"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nSaved trained model to {save_dir}")


if __name__ == "__main__":
    main()
