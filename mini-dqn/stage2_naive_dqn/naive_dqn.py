"""
Stage 2: naive DQN on CartPole -- the table becomes a network, and things get unstable.

Stage 1's Q-table worked because the gridworld had 25 states we could visit over and over. CartPole's
state is 4 continuous floats, so there are infinitely many states and the agent will never see the same
one twice. A lookup table is useless; we need something that GENERALIZES between similar states.

So Q becomes a neural network. Same architecture as mini-r1's PolicyNet, but the outputs mean something
different: not "how likely is this action" but "what return do I expect from this action."

This version is deliberately naive -- no replay buffer, no target network. It trains on each transition
once, in the order it happens. Expect it to learn and then fall apart. Three reasons, all worth feeling
before Stage 3 fixes them:

  1. SHARED WEIGHTS. In the table, updating Q[s,a] moved exactly one number. Here every update moves
     shared weights, so learning about one state silently changes the value of EVERY state.
  2. CORRELATED SAMPLES. Gradient descent assumes roughly independent samples; consecutive CartPole
     frames are nearly identical.
  3. A MOVING TARGET. r + gamma*max Q(s') is computed with the same network we're updating, so we chase
     our own tail.

Run:  uv run python stage2_naive_dqn/naive_dqn.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---- Hyperparameters ----
SEED = 0
GAMMA = 0.99
LR = 1e-3
HIDDEN = 128
NUM_EPISODES = 400
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99        # per episode
DEVICE = "cpu"              # tiny net -> CPU beats MPS (GPU transfer overhead isn't worth it)


# ---- Part 1: The Q-network (the table's replacement) ----
class QNet(nn.Module):
    """Maps a 4-number state to an estimated RETURN for each of the 2 actions.

    Identical shape to mini-r1's PolicyNet, but read the outputs differently: PolicyNet produced logits
    (how likely should this action be), QNet produces values (what do I expect to get from it).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---- Part 2: Choosing an action (epsilon-greedy, same as Stage 1) ----
def select_action(q_net: QNet, state: np.ndarray, epsilon: float, rng: np.random.Generator,
                  n_actions: int) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():   # just picking an action, not learning -- don't build a graph
        q_values = q_net(torch.from_numpy(state).float().to(DEVICE))
    return int(q_values.argmax().item())


# ---- Part 3: One learning step (the Stage 1 update, now by gradient descent) ----
def learn(q_net: QNet, optimizer: torch.optim.Optimizer, state: np.ndarray, action: int,
          reward: float, next_state: np.ndarray, terminated: bool) -> float:
    """Stage 1 nudged a table entry toward the TD target. Here we can't assign to a cell, so instead we
    take a gradient step that moves the network's PREDICTION toward that same target."""
    state_t = torch.from_numpy(state).float().to(DEVICE)
    next_state_t = torch.from_numpy(next_state).float().to(DEVICE)

    predicted = q_net(state_t)[action]

    with torch.no_grad():
        # no_grad is essential: the target must be treated as a fixed number. If gradients flowed into
        # it, the network could "win" by dragging the target down to meet its prediction.
        #
        # Note we bootstrap unless TERMINATED. CartPole also stops at 500 steps (truncation), but that
        # is the pole still balanced, not failure -- calling it terminal would teach the agent that
        # surviving leads to zero future value.
        target = reward if terminated else reward + GAMMA * q_net(next_state_t).max().item()
    target_t = torch.tensor(target, dtype=torch.float32, device=DEVICE)

    loss = nn.functional.mse_loss(predicted, target_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ---- Part 4: Training loop ----
def train() -> list[float]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    env = gym.make("CartPole-v1")
    q_net = QNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

    episode_rewards = []
    epsilon = EPSILON_START

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0
        done = False

        while not done:
            action = select_action(q_net, state, epsilon, rng, env.action_space.n)
            next_state, reward, terminated, truncated, _ = env.step(action)
            learn(q_net, optimizer, state, action, float(reward), next_state, terminated)
            state = next_state
            total_reward += float(reward)
            done = terminated or truncated   # the EPISODE ends either way; only `terminated` blocks bootstrapping

        episode_rewards.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 25 == 0:
            avg = np.mean(episode_rewards[-25:])
            print(f"Episode {episode:4d} | eps {epsilon:.2f} | avg reward (last 25): {avg:6.1f}")

    env.close()
    return episode_rewards


# ---- Part 5: Plot ----
def plot(episode_rewards: list[float], path: str = "stage2_naive_dqn/learning_curve.png") -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(episode_rewards, alpha=0.3, label="per episode")
    window = 25
    if len(episode_rewards) >= window:
        moving = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(episode_rewards)), moving, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (steps balanced)")
    plt.title("Naive DQN on CartPole (no replay buffer, no target network)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    rewards = train()
    best = max(rewards)
    best_window = max(np.convolve(rewards, np.ones(25) / 25, mode="valid"))
    print(f"\nbest single episode: {best:.0f} | best 25-ep average: {best_window:.1f}"
          f" | final 25-ep average: {np.mean(rewards[-25:]):.1f}")
    plot(rewards)
