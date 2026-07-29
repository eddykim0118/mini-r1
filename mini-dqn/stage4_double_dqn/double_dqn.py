"""
Stage 4: Double DQN -- fixing the overestimation baked into `max`.

Stage 3's target was `r + gamma * max_a' Q_target(s', a')`. That `max` has a subtle, systematic flaw.
Q values are noisy estimates, and taking the max of several noisy numbers doesn't pick the best action
-- it picks whichever action's noise happened to point UP. Do that at every step of training and the
target is biased upward, so the agent grows persistently over-optimistic about how good things are.

The fix (Double DQN) is to stop letting one network both choose and grade its own choice:

  DQN:        target = r + gamma * Q_target(s', argmax_a' Q_TARGET(s', a'))   # same net does both
  Double DQN: target = r + gamma * Q_target(s', argmax_a' Q_ONLINE(s', a'))   # online picks, target grades

The online net nominates the action it thinks is best; the frozen target net says what it's worth. For
the estimate to be inflated now, BOTH networks have to be wrong in the same direction at once, which is
much rarer than one network fooling itself.

To make the bias visible we track two numbers per episode:
  - what the agent PREDICTED at the starting state: V(s0) = max_a Q(s0, a)
  - what it ACTUALLY got: the discounted return of that episode
An honest agent's two curves sit on top of each other. An over-optimistic one predicts above what it
earns. (With gamma=0.99, a perfect 500-step episode is worth sum(0.99^t) ~= 99.3, so that's the ceiling.)

Run:  uv run python stage4_double_dqn/double_dqn.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---- Hyperparameters (identical to Stage 3's "both" config) ----
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
LEARN_START = 500
TARGET_SYNC = 200


# ---- Part 1: Network and buffer (unchanged from Stage 3) ----
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


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.rng = rng
        self.size = 0
        self.ptr = 0

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


def select_action(q_net: QNet, state: np.ndarray, epsilon: float, rng: np.random.Generator,
                  n_actions: int) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():
        q_values = q_net(torch.from_numpy(state).float().to(DEVICE))
    return int(q_values.argmax().item())


# ---- Part 2: The learning step, with the one-line difference that defines Double DQN ----
def learn(q_net: QNet, target_net: QNet, optimizer: torch.optim.Optimizer,
          batch: tuple[np.ndarray, ...], double: bool) -> None:
    states, actions, rewards, next_states, terminated = batch
    s = torch.from_numpy(states).to(DEVICE)
    a = torch.from_numpy(actions).to(DEVICE)
    r = torch.from_numpy(rewards).to(DEVICE)
    s2 = torch.from_numpy(next_states).to(DEVICE)
    term = torch.from_numpy(terminated).to(DEVICE)

    predicted = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if double:
            # ONLINE net nominates the action; TARGET net prices it. Splitting the two jobs means an
            # inflated value now requires both networks to err upward on the same action.
            best = q_net(s2).argmax(dim=1)
            next_q = target_net(s2).gather(1, best.unsqueeze(1)).squeeze(1)
        else:
            # one network both picks and grades -> it grades whichever of its own errors points up
            next_q = target_net(s2).max(dim=1).values
        target = r + GAMMA * next_q * (1.0 - term)

    loss = nn.functional.mse_loss(predicted, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---- Part 3: Training, tracking predicted-vs-actual so the bias is measurable ----
def train(double: bool, label: str) -> tuple[list[float], list[float], list[float]]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    env = gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    q_net = QNet(obs_dim, n_actions).to(DEVICE)
    target_net = QNet(obs_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY, obs_dim, rng)

    episode_rewards, predicted_v0, actual_return = [], [], []
    epsilon = EPSILON_START
    global_step = 0

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)

        # What does the agent CLAIM this episode is worth, before it has played it?
        with torch.no_grad():
            v0 = q_net(torch.from_numpy(state).float().to(DEVICE)).max().item()

        total_reward, discounted, discount = 0.0, 0.0, 1.0
        done = False

        while not done:
            action = select_action(q_net, state, epsilon, rng, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.push(state, action, reward, next_state, terminated)
            if buffer.size >= LEARN_START:
                learn(q_net, target_net, optimizer, buffer.sample(BATCH_SIZE), double)

            global_step += 1
            if global_step % TARGET_SYNC == 0:
                target_net.load_state_dict(q_net.state_dict())

            state = next_state
            total_reward += float(reward)
            discounted += discount * float(reward)   # what it ACTUALLY earned, same discounting as Q
            discount *= GAMMA
            done = terminated or truncated

        episode_rewards.append(total_reward)
        predicted_v0.append(v0)
        actual_return.append(discounted)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            print(f"  [{label}] episode {episode:4d} | reward {np.mean(episode_rewards[-25:]):6.1f} "
                  f"| predicted V(s0) {np.mean(predicted_v0[-25:]):6.1f} "
                  f"| actual return {np.mean(actual_return[-25:]):6.1f}")

    env.close()
    return episode_rewards, predicted_v0, actual_return


# ---- Part 4: Compare ----
def moving(x: list[float], window: int = 25) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main() -> None:
    results = {}
    for label, double in [("DQN (max)", False), ("Double DQN", True)]:
        print(f"\n=== {label} ===")
        results[label] = train(double, label)

    # The bias FLIPS SIGN during training: while Q is still climbing from its zero init the agent
    # badly UNDER-predicts, and only once it converges does overestimation appear. Averaging over both
    # regimes cancels them out and reports ~nothing, so measure the two phases separately.
    ceiling = (1.0 - GAMMA ** 500) / (1.0 - GAMMA)   # best return the environment can possibly pay
    print("\n=== summary (400 episodes, identical seeds) ===")
    print(f"ceiling: a perfect 500-step episode is worth {ceiling:.1f} at gamma={GAMMA}\n")
    print(f"{'config':<14} {'best 25-ep':>11} {'final 25-ep':>12} {'bias ep100-200':>15} "
          f"{'bias last 100':>14} {'eps over ceiling':>18}")
    for label, (rewards, pred, actual) in results.items():
        early = np.mean(pred[100:200]) - np.mean(actual[100:200])
        late = np.mean(pred[-100:]) - np.mean(actual[-100:])
        over = 100.0 * np.mean(np.array(pred) > ceiling)
        print(f"{label:<14} {max(moving(rewards)):>11.1f} {np.mean(rewards[-25:]):>12.1f} "
              f"{early:>+15.1f} {late:>+14.1f} {over:>17.0f}%")

    _, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, (rewards, _, _) in results.items():
        axes[0].plot(range(24, len(rewards)), moving(rewards), label=label)
    axes[0].axhline(475, ls="--", c="gray", label="solved (475)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward (25-ep moving avg)")
    axes[0].set_title("Performance")
    axes[0].legend()

    for label, (_, pred, actual) in results.items():
        line, = axes[1].plot(range(24, len(pred)), moving(pred), label=f"{label}: predicted V(s0)")
        axes[1].plot(range(24, len(actual)), moving(actual), ls="--", color=line.get_color(),
                     label=f"{label}: actual return")
    axes[1].axhline(1 / (1 - GAMMA) * (1 - GAMMA ** 500), ls=":", c="gray",
                    label="max possible (~99.3)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Discounted value")
    axes[1].set_title("Predicted vs actual — the gap IS the overestimation")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = "stage4_double_dqn/overestimation.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
