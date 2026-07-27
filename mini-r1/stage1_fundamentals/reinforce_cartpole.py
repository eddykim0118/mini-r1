"""
Stage 1: REINFORCE on CartPole.

The simplest policy-gradient algorithm, built from scratch. The agent learns to
balance a pole by making the actions that were followed by high return more likely.

Run:  uv run python stage1_fundamentals/reinforce_cartpole.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ---- Hyperparameters ----
SEED = 42
GAMMA = 0.99          # discount: how much a future reward is worth vs. one now
LR = 1e-2             # learning rate for the optimizer
HIDDEN = 128          # width of the policy network's hidden layer
NUM_EPISODES = 1000   # how many full attempts to train on
DEVICE = "cpu"        # tiny net -> CPU beats MPS (GPU transfer overhead isn't worth it)


# ---- Part 1: The policy network (the agent's "brain") ----
class PolicyNet(nn.Module):
    """Maps a 4-number state to a score for each of the 2 actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Returns raw scores ("logits"), one per action. Softmax happens in Categorical.
        return self.net(state)


# ---- Part 2: Choosing an action (with exploration) ----
def select_action(policy: PolicyNet, state: np.ndarray):
    """Sample an action from the policy; return it plus its log-probability."""
    state_t = torch.from_numpy(state).float().to(DEVICE)
    logits = policy(state_t)
    dist = Categorical(logits=logits)  # logits -> a proper probability distribution
    action = dist.sample()             # sampling = exploration, not always the top action
    return action.item(), dist.log_prob(action)


# ---- Part 3: Turning rewards into discounted, normalized returns ----
def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...   (computed back-to-front)."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    # Normalize (subtract mean, divide by std). Acts as a baseline: actions with
    # above-average return get pushed up, below-average get pushed down. Big stabilizer.
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ---- Part 4: The training loop ----
def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = gym.make("CartPole-v1")
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    episode_rewards = []  # total reward per episode (for the learning curve)

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        log_probs, rewards = [], []
        done = False

        # --- Play one full episode ---
        while not done:
            action, log_prob = select_action(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)

        # --- Learn from it ---
        returns = compute_returns(rewards, GAMMA)
        log_probs = torch.stack(log_probs)
        # REINFORCE loss: -sum(log_prob * return). Minimizing this raises the
        # probability of actions that were followed by above-average return.
        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()  # clear last step's gradients
        loss.backward()        # compute new gradients (backprop through the episode)
        optimizer.step()       # nudge the weights

        # --- Log progress ---
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        if episode % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:4d} | avg reward (last 50): {avg:6.1f}")

    env.close()
    return episode_rewards


# ---- Part 5: Plot the learning curve ----
def plot(episode_rewards, path="stage1_fundamentals/learning_curve.png"):
    plt.figure(figsize=(9, 5))
    plt.plot(episode_rewards, alpha=0.3, label="per episode")
    window = 50
    if len(episode_rewards) >= window:  # moving average to see the trend through noise
        moving = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(episode_rewards)), moving, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (steps survived)")
    plt.title("REINFORCE on CartPole")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"\nSaved learning curve to {path}")


if __name__ == "__main__":
    rewards = train()
    plot(rewards)
