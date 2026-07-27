"""
Stage 3: Thompson Sampling (Bayesian bandit).

Keep a BELIEF distribution over each arm's true payout. Each step: draw one random sample from every
arm's belief and play whichever sampled highest; then update that arm's belief with the reward.

For Bernoulli arms the belief is a Beta distribution: just track (wins, losses) per arm, and
Beta(1 + wins, 1 + losses) IS the belief. Exploration is automatic and tuning-free:
  - an uncertain arm has a WIDE belief -> its samples vary a lot -> it occasionally wins the draw (explore)
  - a well-known arm has a NARROW belief -> it only wins the draw when it genuinely is best (exploit)

No epsilon, no c. The uncertainty does the exploring itself.

Run:  uv run python stage3_thompson/thompson.py
"""

import matplotlib.pyplot as plt
import numpy as np

K = 10
T = 10000
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


class EpsilonGreedy:
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
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.t = 0

    def select(self):
        self.t += 1
        for arm in range(len(self.values)):
            if self.counts[arm] == 0:
                return arm
        bonus = self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class ThompsonSampling:
    def __init__(self, k, rng):
        self.rng = rng
        self.alpha = np.ones(k)   # 1 + wins  (Beta parameter)
        self.beta = np.ones(k)    # 1 + losses

    def select(self):
        # Draw a plausible payout for each arm from its current belief, play the best draw.
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward > 0.5:
            self.alpha[arm] += 1  # a win sharpens the belief upward
        else:
            self.beta[arm] += 1   # a loss sharpens it downward


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
        "Thompson Sampling": lambda rng: ThompsonSampling(K, rng),
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
    plt.title(f"Thompson Sampling vs UCB1 vs ε-greedy ({K}-armed Bernoulli)")
    plt.legend()
    plt.tight_layout()
    path = "stage3_thompson/regret_all_three.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
