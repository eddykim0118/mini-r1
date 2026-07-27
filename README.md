# mini-bandits

A from-scratch tour of the **multi-armed bandit** — the cleanest lens on the most fundamental
tradeoff in reinforcement learning: **explore vs. exploit.**

Build four classic algorithms from scratch, simplest to most sophisticated, and benchmark them
head-to-head by **regret** on simulated environments. Pure `numpy` — runs in milliseconds, no GPU.

> Sibling project to [mini-r1](https://github.com/eddykim0118/mini-r1), which covered the
> *policy-optimization* branch of RL (REINFORCE → GRPO → RLVR). This covers the
> *exploration/exploitation* branch that policy gradients skip over.

## The core idea

A row of slot machines ("multi-armed bandit"), each with an unknown payout rate, and limited pulls.
Every step you face the dilemma: **explore** an uncertain machine to *learn*, or **exploit** the
best-so-far to *earn*? We grade algorithms by **regret** — the reward missed by not always playing the
truly-best arm. The goal: make regret grow *logarithmically* (mostly-early mistakes), not *linearly*.

## Roadmap

- [x] **Stage 1 — ε-greedy.** The raw explore/exploit knob: exploit the best arm, but with probability
  ε pick a random arm. Sweeping ε shows the tradeoff (ε=0 gets stuck, ε=0.5 wastes pulls, ε=0.1 is the
  sweet spot) — and that fixed-ε exploration means *linear* regret forever, motivating UCB.
- [x] **Stage 2 — UCB1.** "Optimism under uncertainty" — explore the arm whose *upper confidence bound*
  is highest. Exploration driven by how unsure we are, not by a coin flip. Over a long horizon its
  regret is *sublinear* (the curve flattens) and crosses below ε-greedy's *linear* line.
- [x] **Stage 3 — Thompson Sampling.** The Bayesian approach: keep a Beta belief per arm, sample from
  it, play the winner. Elegant, tuning-free, and what actually ships in industry — beat ε-greedy and
  UCB1 by ~7–9× regret here, with zero knobs.
- [ ] **Stage 4 — LinUCB (contextual).** Decisions that depend on *features* (a user, a situation) —
  the leap from toy to practical.

## Setup

Uses [`uv`](https://docs.astral.sh/uv/). Python 3.12 pinned.

```bash
uv sync
uv run python stage1_epsilon_greedy/epsilon_greedy.py
```

## Why bandits

Bandits are "one-step RL": no state transitions, no episodes — pull an arm, get an immediate reward,
repeat. Stripping away the sequential part isolates the explore/exploit question, which is why bandits
are both the classic entry point to RL theory and the form of RL most widely deployed in production
(A/B testing, recommendation, ad selection).
