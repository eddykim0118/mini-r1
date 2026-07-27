"""
Stage 2: GRPO-style policy gradient on CartPole (the bridge).

Takes Stage 1's REINFORCE and adds GRPO's two ideas, plus a simple leash:
  1. Group sampling           - collect a GROUP of episodes per update, not just 1.
  2. Group-relative advantage - weight each episode by (return - group_mean) / group_std.
  3. Gradient clipping        - a stand-in for GRPO's trust region; caps the step size so
                                one update can't blow up the policy.

Same core loss as Stage 1 (log_prob * weight), but a smarter weight and a leash.
The payoff: it reaches ~500 and *holds*, instead of Stage 1's boom-bust.

Run:  uv run python stage2_bridge/grpo_style_cartpole.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ---- Hyperparameters ----
SEED = 42
LR = 1e-2
HIDDEN = 128
GROUP_SIZE = 10        # episodes sampled per update -> the "group" in GRPO
NUM_UPDATES = 150      # 150 updates x 10 episodes = 1500 episodes total
MAX_GRAD_NORM = 1.0    # gradient-clipping leash (stand-in for GRPO's trust region)
DEVICE = "cpu"


# ---- Policy network (identical to Stage 1) ----
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state):
        return self.net(state)


def select_action(policy, state):
    state_t = torch.from_numpy(state).float().to(DEVICE)
    dist = Categorical(logits=policy(state_t))
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def run_episode(policy, env, seed):
    """Play one full episode. Return (summed log-probs, total reward)."""
    state, _ = env.reset(seed=seed)
    log_probs, total_reward = [], 0.0
    done = False
    while not done:
        action, log_prob = select_action(policy, state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        total_reward += reward
    return torch.stack(log_probs).sum(), total_reward


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = gym.make("CartPole-v1")
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    episode_rewards = []  # every individual episode, for a per-episode learning curve

    for update in range(1, NUM_UPDATES + 1):
        group_logps, group_returns = [], []

        # --- 1. Collect a GROUP of episodes under the CURRENT policy ---
        for i in range(GROUP_SIZE):
            seed = SEED + update * GROUP_SIZE + i
            logp_sum, total_reward = run_episode(policy, env, seed)
            group_logps.append(logp_sum)
            group_returns.append(total_reward)
            episode_rewards.append(total_reward)

        # --- 2. Group-relative advantage: grade each episode on the group's own curve ---
        returns_t = torch.tensor(group_returns, dtype=torch.float32, device=DEVICE)
        advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        # Note: if all episodes score the same (e.g. all hit 500), std ~ 0 -> advantages ~ 0
        # -> no update. Once mastered, the agent naturally STOPS thrashing. That's stability.

        # --- 3. GRPO-style loss + the leash ---
        # One advantage per whole episode, broadcast to all its actions -- exactly how GRPO
        # broadcasts a sequence-level reward to every token of a generated response.
        logps_t = torch.stack(group_logps)
        loss = -(advantages * logps_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)  # the leash
        optimizer.step()

        if update % 10 == 0:
            recent = np.mean(episode_rewards[-GROUP_SIZE * 10:])
            print(f"Update {update:4d} | episodes {update*GROUP_SIZE:5d} | avg reward: {recent:6.1f}")

    env.close()
    return episode_rewards


def plot(episode_rewards, path="stage2_bridge/learning_curve.png"):
    plt.figure(figsize=(9, 5))
    plt.plot(episode_rewards, alpha=0.3, label="per episode")
    window = 50
    if len(episode_rewards) >= window:
        moving = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(episode_rewards)), moving, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (steps survived)")
    plt.title("GRPO-style (group baseline + gradient clipping) on CartPole")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"\nSaved learning curve to {path}")


if __name__ == "__main__":
    rewards = train()
    plot(rewards)
