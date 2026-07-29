# mini-dqn

The third pillar of [`rl-lab`](../README.md): **value-based** reinforcement learning, built from
scratch. Tabular Q-learning to Double DQN.

## The core idea, in one sentence

Instead of learning "what should I do here?" directly, learn "how good is this?" — and let the best
action fall out of the answer.

## Why this is different from the other two projects

[`mini-r1`](../mini-r1/) learned a **policy** directly: nudge up whatever actions preceded good
outcomes. [`mini-bandits`](../mini-bandits/) answered *when to try something new*, but every decision
paid off immediately. Neither one had to deal with a reward that arrives twenty steps after the action
that earned it.

That's the problem here — **credit assignment** — and value functions are the answer to it. The bridge
is smaller than it looks. A bandit's update rule is:

```
Q += alpha * ( reward                  - Q )     # nudge toward what I just saw
```

and Q-learning's is:

```
Q += alpha * ( r + gamma * max Q(s')   - Q )     # nudge toward reward + what comes next
```

Identical machinery. The only change is that the target stops being a plain observed reward and becomes
partly a *guess about the future* — which is called **bootstrapping**, and which is where all of this
gets both its power and its instability.

## Roadmap

- [x] **Stage 1 — Tabular Q-learning** on a from-scratch gridworld (numpy, no gym). The Bellman update
  by hand. Watch value propagate *backwards* from the goal, one square per episode — credit assignment
  made visible.
- [x] **Stage 2 — Naive DQN** on CartPole. Swap the lookup table for a neural network, because you can't
  tabulate continuous states. It learns... and then falls apart. The instability is the lesson.
- [x] **Stage 3 — Replay buffer + target network**, each ablated so it's clear what each one buys.
  Both are patches for problems that bootstrapping created, and seeing them fail first is the
  difference between understanding them and cargo-culting them.
- [x] **Stage 4 — Double DQN.** Taking `max` over noisy estimates systematically picks whichever action
  got *lucky*, so the agent grows persistently over-optimistic. Measuring the gap between predicted Q
  and actual return makes the bias visible; splitting action-*selection* from action-*evaluation*
  closes it.

## Results

| Stage | Setup | Outcome |
|---|---|---|
| 1 | Tabular Q-learning, 5x5 maze | Recovers the **exact** optimal value function (start cell 0.698 = γ⁷) and the optimal 8-step policy, from 109 random steps |
| 2 | Naive DQN, no fixes | **Fails.** Peaks at a 49.8 25-ep average (475 = solved), and spends 175 episodes *below* random |
| 3 | Replay / target ablation | **Replay is load-bearing** — both replay configs hit 500.0, neither non-replay config cleared 50. A target network *alone* is worse than nothing (32.8 vs 49.8) |
| 4 | Double DQN | Cuts converged-phase overestimation **62%** (+22.4 → +8.4) and finishes at a stable 500 vs plain DQN's 302 |

**The lesson worth keeping:** the two DQN fixes are not interchangeable and not independent. Replay does
the learning; the target network doesn't *prevent* collapse — both replay configs catastrophically
forgot — it provides **recoverability**. Replay-only collapsed at episode 320 and flatlined; with a
target network it collapsed and climbed back within 25 episodes.

## Setup

```bash
uv sync
uv run python stage1_gridworld/tabular_q.py
```

Stage 1 is pure numpy; `torch` and `gymnasium` arrived at Stage 2, when there was actually a network to
train. Every stage runs on CPU in a few minutes — no GPU, and `DEVICE = "cpu"` on purpose, since a
4→128→2 network is small enough that MPS transfer overhead costs more than it saves.

## Why these choices

- **Gridworld before CartPole** — with a small table you can *print the value function* and watch it
  fill in. Once a neural network is in the loop, the algorithm and the function approximator fail in
  different ways, and you want to have already seen the algorithm work.
- **CartPole for stages 2–4** — the same task `mini-r1`'s REINFORCE agent solved, which turns
  "policy-gradient vs. value-based" from an abstract distinction into two curves on comparable axes.
- **ε-greedy for exploration** — carried straight over from `mini-bandits` Stage 1 (now decayed over
  training). Thompson Sampling dominated in the bandit setting, but maintaining a posterior over an
  entire value function is a genuinely hard problem, which is why plain ε-greedy is what deep RL
  actually ships.
