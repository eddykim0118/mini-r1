"""
Stage 4: LinUCB — the contextual bandit (the leap to practical).

Until now every step faced the SAME arms, so there was one global "best arm." Real problems have
CONTEXT: the best action depends on features of the situation (a user, a page, a patient). Here each
step provides a context vector x, and each arm's expected reward is linear in it: reward ~= theta_a . x.

LinUCB learns each arm's weight vector theta_a online via ridge regression, and adds a UCB-style
optimism bonus about its uncertainty IN THIS CONTEXT'S DIRECTION, then picks the best optimistic arm
for this specific x. Per arm it keeps:
    A = I + sum(x x^T)     (d x d)   -- ridge "design matrix"
    b = sum(reward * x)    (d,)      -- accumulated reward-weighted contexts
    theta_hat = A^-1 b               -- current best-guess weights
    score(x) = theta_hat . x  +  alpha * sqrt(x^T A^-1 x)   (estimate + uncertainty)

To prove context matters we race it against a CONTEXT-BLIND agent (ε-greedy on global arm means):
because expected reward averages to ~0 over random contexts, the blind agent sees every arm as equally
mediocre and picks almost randomly.

Run:  uv run python stage4_linucb/linucb.py
"""

import matplotlib.pyplot as plt
import numpy as np

K = 6          # arms
D = 5          # context dimension (number of features)
T = 3000
N_RUNS = 80
SEED = 0
NOISE = 0.1


class LinearContextualBandit:
    def __init__(self, k, d, rng, noise=NOISE):
        self.k, self.d, self.rng, self.noise = k, d, rng, noise
        self.theta = rng.normal(0, 1, size=(k, d))   # hidden weight vector per arm

    def context(self):
        x = self.rng.normal(0, 1, size=self.d)
        return x / np.linalg.norm(x)                 # unit-norm context vector

    def means(self, x):
        return self.theta @ x                        # expected reward of each arm for this context

    def pull(self, arm, x):
        return self.theta[arm] @ x + self.rng.normal(0, self.noise)


class LinUCB:
    def __init__(self, k, d, alpha):
        self.k, self.alpha = k, alpha
        self.A = [np.eye(d) for _ in range(k)]        # ridge design matrix per arm
        self.b = [np.zeros(d) for _ in range(k)]      # reward-weighted context sum per arm

    def select(self, x):
        scores = np.zeros(self.k)
        for a in range(self.k):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            mean = theta_hat @ x                       # predicted reward for this context
            bonus = self.alpha * np.sqrt(x @ A_inv @ x)  # uncertainty in this context's direction
            scores[a] = mean + bonus
        return int(np.argmax(scores))

    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)                  # rank-1 ridge update
        self.b[arm] += reward * x


class ContextBlind:
    """ε-greedy on each arm's GLOBAL mean reward; ignores the context entirely."""

    def __init__(self, k, epsilon, rng):
        self.epsilon, self.rng = epsilon, rng
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select(self, x):  # x ignored — that's the whole point
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.values)))
        return int(np.argmax(self.values))

    def update(self, arm, x, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


def run(bandit, agent, T):
    regret = np.zeros(T)
    cumulative = 0.0
    for t in range(T):
        x = bandit.context()
        means = bandit.means(x)
        arm = agent.select(x)
        reward = bandit.pull(arm, x)
        agent.update(arm, x, reward)
        cumulative += means.max() - means[arm]   # regret vs the best arm FOR THIS context
        regret[t] = cumulative
    return regret


def main():
    agents = {
        "context-blind (ε=0.1)": lambda rng: ContextBlind(K, 0.1, rng),
        "LinUCB (α=1.0)": lambda rng: LinUCB(K, D, alpha=1.0),
    }

    plt.figure(figsize=(9, 5))
    for label, make_agent in agents.items():
        avg_regret = np.zeros(T)
        for run_i in range(N_RUNS):
            rng = np.random.default_rng(SEED + 1 + run_i)
            avg_regret += run(LinearContextualBandit(K, D, rng), make_agent(rng), T)
        avg_regret /= N_RUNS
        plt.plot(avg_regret, label=label)
        print(f"{label:<24} final avg cumulative regret: {avg_regret[-1]:7.1f}")

    plt.xlabel("Step")
    plt.ylabel("Cumulative regret (avg over runs)")
    plt.title(f"LinUCB vs context-blind ({K} arms, {D}-dim context)")
    plt.legend()
    plt.tight_layout()
    path = "stage4_linucb/regret_linucb.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
