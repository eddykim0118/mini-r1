"""
Stage 1: tabular Q-learning on a gridworld.

The agent starts knowing nothing and has to find a goal in a small maze. The only reward in the whole
world is +1 for reaching the goal; every other step pays exactly 0.

The point of this stage is to WATCH the value function fill in. Since the agent's entire knowledge is
just 25 states x 4 actions = 100 numbers, we can print all of it. What you should see: after one
episode only the cells near the goal are worth anything, and with more episodes that value seeps
backwards through the maze, one cell per visit, until it reaches the start. That backward seepage is
how a delayed reward gets credited to the actions that earned it.

Run:  uv run python stage1_gridworld/tabular_q.py
"""

import matplotlib.pyplot as plt
import numpy as np

# ---- Hyperparameters ----
SEED = 0
SIZE = 5              # the grid is SIZE x SIZE
GAMMA = 0.95          # discount: a reward 6 steps away is worth 0.95^6 ~= 0.74, so SHORT paths win
ALPHA = 0.1           # step size. Constant (not 1/n) -> the agent keeps adapting. See mini-bandits/docs/05
EPSILON_START = 1.0   # start by exploring exclusively: an all-zero table has no signal to exploit yet
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99  # per episode
NUM_EPISODES = 500
MAX_STEPS = 200       # give up on an episode after this, so early random walks can't run forever
SNAPSHOTS = (1, 5, 25, 100, NUM_EPISODES)   # episodes at which we print the value function


# ---- Part 1: The world (a tiny MDP) ----
class GridWorld:
    """A SIZE x SIZE maze. Deterministic moves; walking into a wall or edge leaves you where you are.

    Layout (S = start, G = goal, # = wall):
        S . . . .
        . # # # .
        . . . . .
        . # # # .
        . . . . G
    """

    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))   # up, down, left, right
    ARROWS = ("^", "v", "<", ">")

    def __init__(self, size: int = SIZE):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.walls = {(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3)}
        self.pos = self.start

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return len(self.MOVES)

    def index(self, pos: tuple[int, int]) -> int:
        """Flatten a (row, col) into a single state number, so Q can be a 2-D array."""
        return pos[0] * self.size + pos[1]

    def reset(self) -> int:
        self.pos = self.start
        return self.index(self.pos)

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take an action; return (next_state, reward, done)."""
        dr, dc = self.MOVES[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        # Blocked by the edge or a wall? Then the move simply doesn't happen.
        if 0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.walls:
            self.pos = (r, c)
        done = self.pos == self.goal
        reward = 1.0 if done else 0.0   # the ONLY reward in the entire world
        return self.index(self.pos), reward, done


# ---- Part 2: Choosing an action (epsilon-greedy, straight from mini-bandits) ----
def select_action(Q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))          # explore
    q = Q[state]
    # Break ties at random. This matters more than it looks: early on the whole row is 0.0, and
    # np.argmax would always return action 0, so the agent would march into the same wall forever.
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))


# ---- Part 3: Looking inside the agent's head ----
def format_values(Q: np.ndarray, env: GridWorld) -> str:
    """V(s) = max_a Q(s,a) -- 'how good is this cell?' -- laid out as the grid itself."""
    V = Q.max(axis=1).reshape(env.size, env.size)
    lines = []
    for r in range(env.size):
        cells = []
        for c in range(env.size):
            cells.append("  ##  " if (r, c) in env.walls else f"{V[r, c]:6.3f}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def format_policy(Q: np.ndarray, env: GridWorld) -> str:
    """The greedy policy: in each cell, which way does the agent think it should go?"""
    lines = []
    for r in range(env.size):
        cells = []
        for c in range(env.size):
            if (r, c) in env.walls:
                cells.append("#")
            elif (r, c) == env.goal:
                cells.append("G")
            else:
                cells.append(env.ARROWS[int(np.argmax(Q[env.index((r, c))]))])
        lines.append(" ".join(cells))
    return "\n".join(lines)


# ---- Part 4: Q-learning ----
def train() -> tuple[np.ndarray, list[int], GridWorld]:
    rng = np.random.default_rng(SEED)
    env = GridWorld()
    Q = np.zeros((env.n_states, env.n_actions))   # the agent's entire knowledge: 100 numbers
    steps_per_episode = []
    epsilon = EPSILON_START

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        steps = MAX_STEPS
        for t in range(MAX_STEPS):
            action = select_action(Q, state, epsilon, rng)
            next_state, reward, done = env.step(action)

            # The TD target: what we now think this action was worth.
            #   reward         -- what we actually just got (0 almost everywhere)
            #   GAMMA * max Q  -- plus the discounted value of the best thing available next.
            # That second term is BOOTSTRAPPING: an estimate built from our own other estimates. It's
            # what lets a zero-reward step still teach us something, and it's why value spreads
            # backwards from the goal. At a terminal state there IS no "next", so the target is just
            # the reward -- bootstrapping past the end of an episode is a classic silent bug.
            td_target = reward if done else reward + GAMMA * Q[next_state].max()
            Q[state, action] += ALPHA * (td_target - Q[state, action])

            state = next_state
            if done:
                steps = t + 1
                break

        steps_per_episode.append(steps)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode in SNAPSHOTS:
            print(f"\n--- after episode {episode} (epsilon={epsilon:.2f}, {steps} steps) ---")
            print(format_values(Q, env))

    return Q, steps_per_episode, env


# ---- Part 5: Plot ----
def plot(steps_per_episode: list[int], Q: np.ndarray, env: GridWorld,
         path: str = "stage1_gridworld/learning_curve.png") -> None:
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(steps_per_episode, alpha=0.3, label="per episode")
    window = 25
    if len(steps_per_episode) >= window:
        moving = np.convolve(steps_per_episode, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window - 1, len(steps_per_episode)), moving, label=f"{window}-ep moving avg")
    optimal = 2 * (env.size - 1)   # shortest possible path: right/down along the open corridors
    axes[0].axhline(optimal, ls="--", c="gray", label=f"optimal ({optimal} steps)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Steps to reach the goal")
    axes[0].set_title("Learning to escape the maze")
    axes[0].legend()

    V = Q.max(axis=1).reshape(env.size, env.size).copy()
    for r, c in env.walls:
        V[r, c] = np.nan          # leave walls blank rather than pretending they have a value
    im = axes[1].imshow(V, cmap="viridis")
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) not in env.walls:
                axes[1].text(c, r, f"{V[r, c]:.2f}", ha="center", va="center", color="w", fontsize=8)
    axes[1].set_title("Learned value V(s) = max_a Q(s,a)")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    Q, steps, env = train()
    print("\n=== final greedy policy ===")
    print(format_policy(Q, env))
    print(f"\nfirst episode: {steps[0]} steps | last 25 episodes avg: {np.mean(steps[-25:]):.1f} steps")
    plot(steps, Q, env)
