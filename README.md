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

- [x] **Stage 1 — The engine.** A `REINFORCE` policy-gradient agent, from scratch, that learns to
  balance CartPole. Establishes the four primitives everything else is built on: *policy, action,
  reward, gradient*. (It also vividly demonstrates *policy collapse* — the instability that motivates
  Stage 2.)
- [x] **Stage 2 — The bridge.** GRPO explained as "REINFORCE with a group baseline." The conceptual
  step from balancing a pole to training a language model — same gradient, smarter baseline. The
  group-relative advantage + a gradient-clipping leash turn Stage 1's boom-bust into a stable plateau.
- [x] **Stage 3 — The capstone.** GRPO on a small language model (via HuggingFace TRL) for a task with
  a *verifiable* reward: the **Countdown** number game. The full RLVR loop works end-to-end on a laptop
  — and it taught a sharp, honest lesson about reward design (see Results below).

## Results (Stage 3 capstone)

Trained `Qwen2.5-0.5B-Instruct` with GRPO on Countdown (RLVR) on an Apple M4 Pro — no datacenter.
Honest, strict, greedy evaluation on held-out puzzles (N=24, so treat as approximate):

| Run | format (before → after) | exactly-solved (before → after) |
|---|---|---|
| 3-number puzzles, 200 steps | 0% → ~92% | 8% → 4% |
| 2-number curriculum, 300 steps | 0% → 100% | 46% → 29% |

**What worked:** the full RLVR loop runs end-to-end, and GRPO reliably taught the model the
`<think>/<answer>` output format (→ ~100% compliance). Reward provably climbs; the mechanism from
Stages 1–2 scales to a language model.

**What didn't — the interesting part:** GRPO did *not* improve, and consistently *degraded*, actual
problem-solving. The dense, easy **format** reward dominated the sparse **correctness** reward, so the
model optimized formatting. And forcing a 0.5B model to "show its work" makes it condition answers on
its own unreliable reasoning — derailing even trivial puzzles (`21 + 2 = 23` → the trained model
reasoned itself into `(21 - 1) + 1`).

**The lesson:** *you get what you reward, not what you want.* The hard part of RLVR isn't running GRPO —
it's designing a reward and curriculum where the easy path and the intended path are the same path.
A reproducible negative result that captures the central difficulty of the field.

**Future work:** gate or drop the format reward so correctness dominates; SFT warm-start on correct
solutions before RL (as real R1 pipelines do); a larger base model; longer training with a denser
correctness signal.

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
