# mini-r1

A from-scratch journey from **CartPole** to an **R1-style reasoning LLM**, using reinforcement learning.

The goal: understand — by building — how models like DeepSeek-R1 are trained to *reason*, using
**RL with Verifiable Rewards (RLVR)** and the **GRPO** algorithm. We start with the simplest possible
RL agent and climb, one comprehensible step at a time, to teaching a small language model to solve
problems it was never explicitly shown how to solve.

> Built as a learning project on a laptop (Apple M4 Pro). No data-center required.

## The core idea, in one sentence

Let a model try a problem many times, automatically check which attempts were correct, and nudge the
model toward whatever it did on the good attempts — no human labelers, no reward model, just a checker.

## Roadmap

- [ ] **Stage 1 — The engine.** A `REINFORCE` policy-gradient agent, from scratch, that learns to
  balance CartPole. Establishes the four primitives everything else is built on: *policy, action,
  reward, gradient*.
- [ ] **Stage 2 — The bridge.** GRPO explained as "REINFORCE with a group baseline." The conceptual
  step from balancing a pole to training a language model — same gradient, smarter baseline.
- [ ] **Stage 3 — The capstone.** GRPO on a small language model (via HuggingFace TRL) for a task with
  a *verifiable* reward: the **Countdown** number game. Watch accuracy climb — and watch the model
  start writing out its reasoning on its own, because reasoning earns reward.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for Python and dependency management.

```bash
# Install dependencies into an isolated environment (.venv)
uv sync

# Run something inside that environment
uv run python stage1_fundamentals/reinforce_cartpole.py
```

Python 3.12 is pinned (`.python-version`) because PyTorch does not yet ship builds for newer versions.

## Why these choices

- **CartPole first** — RL's "hello world." Runs in seconds on a CPU, so we learn the algorithm without
  waiting on training.
- **GRPO** — the algorithm behind DeepSeek-R1. Simpler than PPO (no separate value network), which
  makes it a friendlier first "real" RL algorithm.
- **Countdown** — correctness is trivially checkable by code, which is exactly what "verifiable reward"
  means. It's also the task the well-known *TinyZero* R1 reproduction used.
