# AI Smart Mouse: Q-learning maze solver.
# Watch the mouse learn episode by episode, then watch it run the learned path.
#
# Usage:  python index.py [--seed N]
# Window keys: UP/DOWN = faster/slower, Esc = quit
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


speed = 20  # mouse steps per second in the window


def draw(screen, font, state, text):
    """Draws one frame and waits one tick. Returns False if the user quit."""
    global speed
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            speed = min(speed * 2, 1000)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
            speed = max(speed // 2, 1)

    colors = {'S': "green", 'C': "yellow", '#': "black", '.': "white"}
    for row in range(maze_rows):
        for col in range(maze_cols):
            rect = (col * unitSize, row * unitSize, unitSize, unitSize)
            pygame.draw.rect(screen, colors[maze[row, col]], rect)
            pygame.draw.rect(screen, "gray80", rect, 1)
    pos = cell_center(*state_to_cell(state))
    pygame.draw.rect(screen, "gray40", pygame.Rect(0, 0, 50, 50).move(pos.x - 25, pos.y - 25), border_radius=10)
    label = font.render(f"{text}   speed {speed}/s", True, "white", "black")
    screen.blit(label, (8, maze_rows * unitSize - label.get_height() - 8))
    pygame.display.flip()
    pygame.time.Clock().tick(speed)
    return True


def train(screen=None, font=None):
    """Trains for num_episodes. With a screen, shows every step; without one, runs instantly.
    Returns False if the user closed the window."""
    for episode in range(1, num_episodes + 1):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        state, total_reward, done, steps = start_state, 0, False, 0

        while not done and steps < max_steps:
            action = choose_action(state, epsilon)
            new_state, reward, done = step(state, action)
            update_q_table(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward
            steps += 1
            if screen and not draw(screen, font, state, f"Episode {episode}/{num_episodes}  step {steps}  "
                                                        f"reward {total_reward:.0f}  epsilon {epsilon:.2f}"):
                return False

        print(f"Episode {episode}: {'cheese' if done else 'timeout'} in {steps} steps, "
              f"reward {total_reward:.0f}, epsilon {epsilon:.3f}")
    return True


def run_greedy(screen=None, font=None):
    """Follows the learned policy from the start. Returns (list of states visited, reached_cheese)."""
    state, path = start_state, [start_state]
    for _ in range(max_steps):
        state, _, done = step(state, greedy_action(state))
        path.append(state)
        if screen and not draw(screen, font, state, "Learned path (no random moves)"):
            break
        if done:
            return path, True
    return path, False


def main():
    if "--seed" in sys.argv:
        np.random.seed(int(sys.argv[sys.argv.index("--seed") + 1]))

    pygame.init()
    screen = pygame.display.set_mode((maze_cols * unitSize, maze_rows * unitSize))
    pygame.display.set_caption("AI Smart Mouse")
    font = pygame.font.Font(None, 28)

    # To skip the animation and jump straight to all 2000 episodes, comment out the next line
    # and uncomment the one after it.
    if not train(screen, font):
        return
    # train()

    path, reached = run_greedy(screen, font)
    print("\n" + ("Reached the cheese" if reached else "Did NOT reach the cheese") + f" in {len(path) - 1} steps (optimal is 10):")
    print(" -> ".join(str(state_to_cell(s)) for s in path))

    while draw(screen, font, path[-1], "Done. Esc to quit"):
        pass
    pygame.quit()


if __name__ == "__main__":
    main()
