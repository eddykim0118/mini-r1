"""
Stage 3: the two fixes that make DQN work -- ablated, so we learn what each one buys.

Stage 2's naive DQN barely learned. Three problems, all from putting a function approximator inside a
bootstrapping loop: shared weights destroy isolation, consecutive samples are correlated, and the target
moves because it's computed from the network being updated.

Two standard patches, aimed at DIFFERENT problems:

  REPLAY BUFFER  -> attacks correlation. Store transitions; train on 64 random PAST ones each step
                    instead of the single one that just happened. Also keeps re-visiting old
                    experience, so regions the agent stopped visiting don't get forgotten.
  TARGET NETWORK -> attacks the moving target. Compute r + gamma*max Q(s') with a FROZEN copy of the
                    network, refreshed every TARGET_SYNC steps. Between syncs this is ordinary
                    supervised regression against a fixed number.

We run all four combinations on identical seeds and hyperparameters. Adopting both as a package would
teach you nothing about which problem actually mattered.

Run:  uv run python stage3_fixes/dqn_ablation.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---- Hyperparameters (identical to Stage 2 where they overlap, so the comparison is fair) ----
SEED = 0
GAMMA = 0.99
LR = 1e-3
HIDDEN = 128
NUM_EPISODES = 400
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99
DEVICE = "cpu"

BUFFER_CAPACITY = 10_000
BATCH_SIZE = 64
LEARN_START = 500      # collect this many transitions before training, so the first batches aren't
                       # 64 copies of the same opening moment
TARGET_SYNC = 200      # steps between refreshing the frozen target network


# ---- Part 1: The Q-network (unchanged from Stage 2) ----
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---- Part 2: The replay buffer ----
class ReplayBuffer:
    """A fixed-size ring buffer of past transitions, sampled uniformly at random.

    The point is decorrelation: a random batch spans many different moments of many different episodes,
    which is much closer to the independent samples gradient descent assumes than a run of consecutive
    frames. It also means old experience keeps being revisited instead of being forgotten.
    """

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.rng = rng
        self.size = 0
        self.ptr = 0     # where the next write goes; wraps around, overwriting the oldest

    def push(self, state, action, reward, next_state, terminated) -> None:
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.terminated[idx])


# ---- Part 3: Choosing an action (unchanged) ----
def select_action(q_net: QNet, state: np.ndarray, epsilon: float, rng: np.random.Generator,
                  n_actions: int) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():
        q_values = q_net(torch.from_numpy(state).float().to(DEVICE))
    return int(q_values.argmax().item())


# ---- Part 4: One learning step, now on a batch ----
def learn(q_net: QNet, target_net: QNet, optimizer: torch.optim.Optimizer,
          batch: tuple[np.ndarray, ...]) -> None:
    """Same TD update as Stages 1 and 2, vectorized over a batch.

    `target_net` is whichever network computes the target -- the frozen copy if the target-network fix
    is on, otherwise q_net itself (which is exactly Stage 2's tail-chasing behaviour).
    """
    states, actions, rewards, next_states, terminated = batch
    s = torch.from_numpy(states).to(DEVICE)
    a = torch.from_numpy(actions).to(DEVICE)
    r = torch.from_numpy(rewards).to(DEVICE)
    s2 = torch.from_numpy(next_states).to(DEVICE)
    term = torch.from_numpy(terminated).to(DEVICE)

    # gather picks, for each row, the Q of the action actually taken
    predicted = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_net(s2).max(dim=1).values
        # (1 - term) is the vectorized form of Stage 2's `if terminated`: at a true terminal there is
        # no future to bootstrap from, so that term is zeroed out.
        target = r + GAMMA * next_q * (1.0 - term)

    loss = nn.functional.mse_loss(predicted, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---- Part 5: Training, parameterized by which fixes are switched on ----
def train(use_replay: bool, use_target: bool, label: str) -> list[float]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    env = gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    q_net = QNet(obs_dim, n_actions).to(DEVICE)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

    if use_target:
        target_net = QNet(obs_dim, n_actions).to(DEVICE)
        target_net.load_state_dict(q_net.state_dict())
    else:
        target_net = q_net          # no fix: the target is computed from the net we're updating

    buffer = ReplayBuffer(BUFFER_CAPACITY, obs_dim, rng) if use_replay else None

    episode_rewards = []
    epsilon = EPSILON_START
    global_step = 0

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0
        done = False

        while not done:
            action = select_action(q_net, state, epsilon, rng, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)

            if use_replay:
                buffer.push(state, action, reward, next_state, terminated)
                if buffer.size >= LEARN_START:
                    learn(q_net, target_net, optimizer, buffer.sample(BATCH_SIZE))
            else:
                # no buffer: train on just the transition that happened, as a batch of one
                single = (state[None].astype(np.float32), np.array([action]),
                          np.array([reward], dtype=np.float32), next_state[None].astype(np.float32),
                          np.array([float(terminated)], dtype=np.float32))
                learn(q_net, target_net, optimizer, single)

            global_step += 1
            if use_target and global_step % TARGET_SYNC == 0:
                target_net.load_state_dict(q_net.state_dict())   # unfreeze, catch up, re-freeze

            state = next_state
            total_reward += float(reward)
            done = terminated or truncated

        episode_rewards.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            print(f"  [{label}] episode {episode:4d} | avg reward (last 25): "
                  f"{np.mean(episode_rewards[-25:]):6.1f}")

    env.close()
    return episode_rewards


# ---- Part 6: Run the ablation ----
def main() -> None:
    configs = [
        ("neither (= Stage 2)", False, False),
        ("replay only", True, False),
        ("target only", False, True),
        ("both (real DQN)", True, True),
    ]

    results = {}
    for label, use_replay, use_target in configs:
        print(f"\n=== {label} ===")
        results[label] = train(use_replay, use_target, label)

    print("\n=== summary (400 episodes each, identical seeds) ===")
    print(f"{'config':<22} {'best 25-ep avg':>15} {'final 25-ep avg':>17}")
    for label, rewards in results.items():
        best = max(np.convolve(rewards, np.ones(25) / 25, mode="valid"))
        print(f"{label:<22} {best:>15.1f} {np.mean(rewards[-25:]):>17.1f}")

    plt.figure(figsize=(10, 6))
    window = 25
    for label, rewards in results.items():
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(rewards)), moving, label=label)
    plt.axhline(475, ls="--", c="gray", label="solved (475)")
    plt.xlabel("Episode")
    plt.ylabel(f"Total reward ({window}-ep moving avg)")
    plt.title("What each DQN fix actually buys (CartPole)")
    plt.legend()
    plt.tight_layout()
    path = "stage3_fixes/ablation.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
