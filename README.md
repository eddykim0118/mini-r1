# rl-lab

Reinforcement learning, built from the ground up. Three algorithm families, implemented from scratch,
each one benchmarked and written up honestly — including where it failed.

> Built on a laptop (Apple M4 Pro). No datacenter required.

## Why three projects

Most RL material teaches one algorithm and stops. But "RL" is really three different questions, and you
don't understand the field until you've built all three:

| Question | Family | Project |
|---|---|---|
| How do I make good actions more likely? | **Policy optimization** | [`mini-r1/`](mini-r1/) |
| When do I try something new vs. take the sure thing? | **Exploration / exploitation** | [`mini-bandits/`](mini-bandits/) |
| How good is this situation, and how do I know? | **Value-based methods** | [`mini-dqn/`](mini-dqn/) |

- [x] **`mini-r1`** — REINFORCE → GRPO → RLVR on a real language model. Policy gradients, from CartPole
  to teaching `Qwen2.5-0.5B` to play the Countdown number game.
- [x] **`mini-bandits`** — ε-greedy → UCB1 → Thompson Sampling → LinUCB. The explore/exploit tradeoff,
  graded by *regret*, ending at the contextual bandit that powers real recommenders.
- [ ] **`mini-dqn`** — tabular Q-learning → DQN → replay buffer + target network → Double DQN. The
  value-based branch, and where bootstrapping earns both its power and its instability.

## Results

**`mini-bandits`** — cumulative regret (lower is better), 10-armed Bernoulli:

| Algorithm | Regret | Shape | Takeaway |
|---|---|---|---|
| ε-greedy (ε=0.1) | ~439 | linear | fixed exploration → linear regret forever |
| UCB1 (c=√2) | ~374 | flattens | explore by *uncertainty*, not at random |
| Thompson Sampling | ~50 | flattest | Bayesian, tuning-free, 7–9× better; what ships in industry |
| LinUCB (contextual) | ~33 vs ~3733 context-blind | flat | using context beats ignoring it by >100× |

**`mini-r1`** — GRPO on `Qwen2.5-0.5B-Instruct`, Countdown, strict greedy eval on held-out puzzles
(N=24, so approximate):

| Run | format compliance | exactly solved |
|---|---|---|
| 3-number puzzles, 200 steps | 0% → ~92% | 8% → 4% |
| 2-number curriculum, 300 steps | 0% → 100% | 46% → 29% |

## What these actually taught me

**You get what you reward, not what you want.** (`mini-r1`) The full RLVR loop works end to end, and
GRPO reliably taught the *output format* — up to 100% compliance. It also consistently **degraded**
actual problem-solving. The dense, easy format reward drowned out the sparse correctness reward, so the
model optimized the thing that was easy to score. A reproducible negative result that captures the
central difficulty of the field: the hard part isn't running the algorithm, it's designing a reward
where the easy path and the intended path are the same path.

**The best algorithm is a property of the problem, not the algorithm.** (`mini-bandits`) Thompson
Sampling dominates everything by 7–9× — on a *stationary* problem. Change the world underneath it and
that lead evaporates. Likewise LinUCB's >100× win came not from smarter exploration but from modeling
reward correctly; a perfect explore/exploit strategy still loses if it's asking the wrong question.

**Optimism, uncertainty, and forgetting are the same conversation.** UCB explores by uncertainty,
Thompson by sampling beliefs, and a constant step size decides how much of the past to trust. All three
are answers to "how confident should I be in what I think I know?"

## Layout

Each project is self-contained — its own `pyproject.toml`, its own dependencies, run from its own
directory. `mini-bandits` is pure numpy; `mini-r1` needs torch and transformers. Nothing shared, nothing
to untangle.

```bash
cd mini-bandits && uv sync && uv run python stage1_epsilon_greedy/epsilon_greedy.py
cd mini-r1     && uv sync && uv run python stage1_fundamentals/reinforce_cartpole.py
```

Python 3.12 is pinned (PyTorch doesn't yet ship builds for newer versions). Each project's own README is
the deep dive; this one is the map.

## Why these choices

- **CartPole first, everywhere** — RL's "hello world," and it runs in seconds on a CPU, so the algorithm
  is the thing you wait on, not the hardware. It also lets `mini-dqn` be compared head-to-head against
  `mini-r1`'s policy-gradient agent on identical ground.
- **Regret as the bandit metric** — it measures *decision quality* against the true best arm, not luck.
- **From scratch, then a library** — every algorithm is hand-written first (numpy or bare torch). The
  one exception is `mini-r1`'s capstone, which uses HuggingFace TRL, because by then the point was
  whether the loop scales to a real model, not whether I could reimplement it.
