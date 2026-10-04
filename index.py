# AI Smart Mouse: Q-learning maze solver.
# Phase 1 trains the mouse headless (fast, no rendering), phase 2 shows what it learned.
#
# Usage:  python index.py [--seed N] [--no-window]
import sys
import numpy as np
import pygame

unitSize = 100
maze = np.array([
    ['S', '#', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '#', '#', '#', '.'],
    ['.', '#', '#', '.', '.', '#', '.'],
    ['.', '.', '.', '#', '.', '#', '#'],
    ['.', '#', '.', '.', '.', '.', 'C']
])

maze_rows, maze_cols = maze.shape
total_states = maze_rows * maze_cols  # 35 states, one per cell
num_actions = 4                       # up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (d_row, d_col) for UP, DOWN, LEFT, RIGHT

start_row, start_col = np.argwhere(maze == 'S')[0]
cheese_row, cheese_col = np.argwhere(maze == 'C')[0]
start_state = start_row * maze_cols + start_col
cheese_state = cheese_row * maze_cols + cheese_col

# --- Q-Learning Parameters ---
learning_rate = 0.1        # alpha: how quickly new information overrides old
discount_factor = 0.95     # gamma: how much future rewards matter
max_epsilon = 1.0          # start fully random
min_epsilon = 0.01         # never stop exploring completely
epsilon_decay_rate = 0.005 # per episode; epsilon is ~0.02 by episode 1000

num_episodes = 2000
max_steps = 200            # an episode that takes longer than this counts as a failure

# --- Rewards ---
REWARD_CHEESE = 100
REWARD_STEP = -1           # every move costs a little, so shorter paths win
REWARD_BUMP = -5           # tried to walk into a wall or off the map

# Q-table: rows = states, columns = actions
q_table = np.zeros((total_states, num_actions))


def state_to_cell(state):
    return int(state // maze_cols), int(state % maze_cols)


def cell_center(row, col):
    return pygame.Vector2(col * unitSize + unitSize / 2, row * unitSize + unitSize / 2)


def step(state, action):
    """Moves one cell. Returns (new_state, reward, done). Blocked moves leave the mouse in place."""
    row, col = state_to_cell(state)
    d_row, d_col = ACTIONS[action]
    new_row, new_col = row + d_row, col + d_col

    if not (0 <= new_row < maze_rows and 0 <= new_col < maze_cols) or maze[new_row, new_col] == '#':
        return state, REWARD_BUMP, False

    new_state = new_row * maze_cols + new_col
    if new_state == cheese_state:
        return new_state, REWARD_CHEESE, True
    return new_state, REWARD_STEP, False


def greedy_action(state):
    """Best known action; ties (e.g. an untouched all-zero row) are broken randomly."""
    row = q_table[state, :]
    return np.random.choice(np.flatnonzero(row == row.max()))


def choose_action(state, epsilon):
    """Epsilon-greedy: explore with probability epsilon, otherwise exploit the Q-table."""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_actions)
    return greedy_action(state)


def update_q_table(state, action, reward, new_state, done):
    """Q-learning (Bellman) update. Returns how much the Q-value changed."""
    old_value = q_table[state, action]
    # A terminal state has no future, so don't bootstrap from it.
    max_future_q = 0.0 if done else np.max(q_table[new_state, :])
    new_value = old_value + learning_rate * (reward + discount_factor * max_future_q - old_value)
    q_table[state, action] = new_value
    return abs(new_value - old_value)


def train():
    print(f"Training for {num_episodes} episodes...")
    print(f"{'episode':>8} {'avg reward':>11} {'avg steps':>10} {'success':>8} {'epsilon':>8} {'avg |dQ|':>9}")

    rewards, steps_taken, successes, q_changes = [], [], [], []
    for episode in range(1, num_episodes + 1):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        state, total_reward, done, steps = start_state, 0, False, 0

        while not done and steps < max_steps:
            action = choose_action(state, epsilon)
            new_state, reward, done = step(state, action)
            q_changes.append(update_q_table(state, action, reward, new_state, done))
            state = new_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_taken.append(steps)
        successes.append(done)

        if episode % 100 == 0:
            print(f"{episode:>8} {np.mean(rewards):>11.1f} {np.mean(steps_taken):>10.1f} "
                  f"{np.mean(successes):>8.0%} {epsilon:>8.3f} {np.mean(q_changes):>9.4f}")
            rewards, steps_taken, successes, q_changes = [], [], [], []


def run_greedy():
    """Follows the learned policy from the start. Returns (list of states visited, reached_cheese)."""
    state, path = start_state, [start_state]
    for _ in range(max_steps):
        state, _, done = step(state, greedy_action(state))
        path.append(state)
        if done:
            return path, True
    return path, False


def run_random(num_steps=30):
    """A mouse that has learned nothing, for the 'before training' comparison."""
    state, path = start_state, [start_state]
    for _ in range(num_steps):
        state, _, done = step(state, np.random.choice(num_actions))
        path.append(state)
        if done:
            break
    return path


def format_path(path):
    return " -> ".join(str(state_to_cell(s)) for s in path)


# ---------- Visualization ----------

def draw_map(screen):
    colors = {'S': "green", 'C': "yellow", '#': "black", '.': "white"}
    for row in range(maze_rows):
        for col in range(maze_cols):
            rect = (col * unitSize, row * unitSize, unitSize, unitSize)
            pygame.draw.rect(screen, colors[maze[row, col]], rect)
            pygame.draw.rect(screen, "gray80", rect, 1)


def draw_policy_arrows(screen):
    """Draws the best known action in every open cell the mouse has learned something about."""
    for row in range(maze_rows):
        for col in range(maze_cols):
            state = row * maze_cols + col
            if maze[row, col] in ('#', 'C') or not q_table[state].any():
                continue
            d_row, d_col = ACTIONS[int(np.argmax(q_table[state]))]
            direction = pygame.Vector2(d_col, d_row)
            side = pygame.Vector2(-direction.y, direction.x)
            center = cell_center(row, col)
            tip = center + direction * 30
            base = center - direction * 10
            pygame.draw.polygon(screen, "firebrick", [tip, base + side * 15, base - side * 15])


def visualize(before_path, after_path, reached):
    pygame.init()
    screen = pygame.display.set_mode((maze_cols * unitSize, maze_rows * unitSize))
    pygame.display.set_caption("AI Smart Mouse")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    after_label = "After training: learned path" if reached else "After training: FAILED to reach the cheese"
    segments = [("Before training: random moves", before_path, 0.12, False),
                (after_label, after_path, 0.3, True)]

    def restart():
        return 0, 0, 0.0, 0.0   # segment, step index, progress within step, pause timer

    seg, idx, progress, pause = restart()
    running = True
    while running:
        dt = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    seg, idx, progress, pause = restart()

        label, path, step_time, show_policy = segments[seg]
        finished = idx >= len(path) - 1

        if finished:
            pause += dt
            if pause > 1.0 and seg < len(segments) - 1:
                seg, idx, progress, pause = seg + 1, 0, 0.0, 0.0
        else:
            progress += dt / step_time
            while progress >= 1 and idx < len(path) - 1:
                progress -= 1
                idx += 1
            finished = idx >= len(path) - 1

        screen.fill("white")
        draw_map(screen)
        if show_policy:
            draw_policy_arrows(screen)

        if finished:
            row, col = state_to_cell(path[-1])
            pos = cell_center(row, col)
        else:
            a_row, a_col = state_to_cell(path[idx])
            b_row, b_col = state_to_cell(path[idx + 1])
            pos = cell_center(a_row, a_col).lerp(cell_center(b_row, b_col), progress)
        pygame.draw.rect(screen, "gray40", pygame.Rect(0, 0, 50, 50).move(pos.x - 25, pos.y - 25), border_radius=10)

        hint = label + ("   (R = replay, Esc = quit)" if finished and seg == len(segments) - 1 else "")
        text = font.render(hint, True, "white", "black")
        screen.blit(text, (8, maze_rows * unitSize - text.get_height() - 8))
        pygame.display.flip()

    pygame.quit()


def main():
    if "--seed" in sys.argv:
        np.random.seed(int(sys.argv[sys.argv.index("--seed") + 1]))

    before_path = run_random()
    train()

    path, reached = run_greedy()
    print()
    if reached:
        print(f"The mouse reached the cheese in {len(path) - 1} steps (optimal is 10):")
    else:
        print("The mouse did NOT reach the cheese. Path it followed:")
    print(format_path(path))

    if "--no-window" not in sys.argv:
        visualize(before_path, path, reached)


if __name__ == "__main__":
    main()
