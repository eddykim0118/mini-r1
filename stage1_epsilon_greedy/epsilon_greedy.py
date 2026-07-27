"""
Stage 1: the epsilon-greedy multi-armed bandit.

The simplest explore/exploit strategy: usually pull the best-so-far arm (exploit), but with
probability epsilon pull a random arm (explore). One knob dials the whole tradeoff.

We sweep several epsilon values and plot cumulative REGRET (reward missed vs. always playing the
best arm) to see the tradeoff with our own eyes.

Run:  uv run python stage1_epsilon_greedy/epsilon_greedy.py
"""

import matplotlib.pyplot as plt
import numpy as np

K = 10           # number of arms
T = 1000         # steps per run
N_RUNS = 200     # independent runs to average over (smooths the noise)
SEED = 0
EPSILONS = [0.0, 0.01, 0.1, 0.5]


# ---- The environment: a row of slot machines ----
class BernoulliBandit:
    """k arms; arm i pays 1 with hidden probability probs[i], else 0."""

    def __init__(self, probs, rng):
        self.probs = np.asarray(probs, dtype=float)
        self.k = len(self.probs)
        self.optimal = self.probs.max()   # expected reward of the best arm (for regret)
        self.rng = rng

    def pull(self, arm):
        return 1.0 if self.rng.random() < self.probs[arm] else 0.0


# ---- The agent ----
class EpsilonGreedy:
    def __init__(self, k, epsilon, rng):
        self.epsilon = epsilon
        self.rng = rng
        self.counts = np.zeros(k)     # times each arm was pulled
        self.values = np.zeros(k)     # running estimate of each arm's mean reward (Q)

    def select(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.values)))   # explore: random arm
        return int(np.argmax(self.values))                    # exploit: best estimate

    def update(self, arm, reward):
        self.counts[arm] += 1
        # Incremental mean: Q <- Q + (reward - Q)/n. Updates the average without storing history.
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


def run(bandit, agent, T):
    """Play T steps; return the cumulative EXPECTED regret at each step."""
    regret = np.zeros(T)
    cumulative = 0.0
    for t in range(T):
        arm = agent.select()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        # Expected regret this step = best arm's mean - chosen arm's mean. Using the TRUE means
        # (not the noisy realized reward) measures decision quality, not luck.
        cumulative += bandit.optimal - bandit.probs[arm]
        regret[t] = cumulative
    return regret


def main():
    master = np.random.default_rng(SEED)
    true_probs = master.uniform(0.0, 1.0, size=K)   # the fixed "problem": each arm's hidden payout
    print(f"True arm payouts: {np.round(true_probs, 3)}")
    print(f"Best arm: #{int(true_probs.argmax())} (payout {true_probs.max():.3f})\n")

    plt.figure(figsize=(9, 5))
    for eps in EPSILONS:
        avg_regret = np.zeros(T)
        for run_i in range(N_RUNS):
            rng = np.random.default_rng(SEED + 1 + run_i)   # independent stream per run
            bandit = BernoulliBandit(true_probs, rng)
            agent = EpsilonGreedy(K, eps, rng)
            avg_regret += run(bandit, agent, T)
        avg_regret /= N_RUNS
        plt.plot(avg_regret, label=f"ε = {eps}")
        print(f"ε={eps:<4}  final avg cumulative regret: {avg_regret[-1]:6.1f}")

    plt.xlabel("Step")
    plt.ylabel("Cumulative regret (avg over runs)")
    plt.title(f"ε-greedy on a {K}-armed Bernoulli bandit")
    plt.legend()
    plt.tight_layout()
    path = "stage1_epsilon_greedy/regret_curves.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
