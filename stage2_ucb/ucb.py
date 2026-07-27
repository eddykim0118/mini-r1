"""
Stage 2: UCB1 (Upper Confidence Bound).

The fix for epsilon-greedy's fatal flaw. Instead of exploring RANDOMLY at a fixed rate, UCB explores
by UNCERTAINTY: pick the arm with the highest optimistic estimate,

    argmax_i [ Q_i  +  c * sqrt( ln(t) / N_i ) ]
                ^^^        ^^^^^^^^^^^^^^^^^^^^^
             how good      exploration bonus: big when an arm is under-sampled (N_i small),
             it looks       shrinks as we learn about it.

Because uncertainty shrinks with data, exploration auto-tapers off -> regret grows LOGARITHMICALLY
(the curve flattens), unlike epsilon-greedy's LINEAR regret (a straight line forever).

Run:  uv run python stage2_ucb/ucb.py
"""

import matplotlib.pyplot as plt
import numpy as np

K = 10
T = 10000        # long horizon, so ε-greedy's LINEAR regret and UCB's crossover are unmistakable
N_RUNS = 150
SEED = 0


class BernoulliBandit:
    def __init__(self, probs, rng):
        self.probs = np.asarray(probs, dtype=float)
        self.k = len(self.probs)
        self.optimal = self.probs.max()
        self.rng = rng

    def pull(self, arm):
        return 1.0 if self.rng.random() < self.probs[arm] else 0.0


class EpsilonGreedy:  # Stage 1 baseline, for comparison
    def __init__(self, k, epsilon, rng):
        self.epsilon = epsilon
        self.rng = rng
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.values)))
        return int(np.argmax(self.values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class UCB1:
    def __init__(self, k, c, rng):
        self.c = c
        self.rng = rng          # kept for interface symmetry; UCB1 is deterministic
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.t = 0

    def select(self):
        self.t += 1
        # Pull each arm once up front (an unpulled arm has infinite uncertainty).
        for arm in range(len(self.values)):
            if self.counts[arm] == 0:
                return arm
        bonus = self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


def run(bandit, agent, T):
    regret = np.zeros(T)
    cumulative = 0.0
    for t in range(T):
        arm = agent.select()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        cumulative += bandit.optimal - bandit.probs[arm]
        regret[t] = cumulative
    return regret


def main():
    master = np.random.default_rng(SEED)
    true_probs = master.uniform(0.0, 1.0, size=K)
    print(f"True arm payouts: {np.round(true_probs, 3)}")
    print(f"Best arm: #{int(true_probs.argmax())} (payout {true_probs.max():.3f})\n")

    agents = {
        "ε-greedy (ε=0.1)": lambda rng: EpsilonGreedy(K, 0.1, rng),
        "UCB1 (c=√2)": lambda rng: UCB1(K, np.sqrt(2), rng),
    }

    plt.figure(figsize=(9, 5))
    for label, make_agent in agents.items():
        avg_regret = np.zeros(T)
        for run_i in range(N_RUNS):
            rng = np.random.default_rng(SEED + 1 + run_i)
            avg_regret += run(BernoulliBandit(true_probs, rng), make_agent(rng), T)
        avg_regret /= N_RUNS
        plt.plot(avg_regret, label=label)
        print(f"{label:<18} final avg cumulative regret: {avg_regret[-1]:6.1f}")

    plt.xlabel("Step")
    plt.ylabel("Cumulative regret (avg over runs)")
    plt.title(f"UCB1 vs ε-greedy on a {K}-armed Bernoulli bandit")
    plt.legend()
    plt.tight_layout()
    path = "stage2_ucb/regret_ucb_vs_epsilon.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
