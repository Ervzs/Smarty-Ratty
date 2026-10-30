# AI Smart Mouse: Q-learning maze solver.
# Watch the mouse learn, episode after episode, until you press Esc or close the window.
import numpy as np
import pygame

unitSize = 100
maze = np.array([
    ['S', '#', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '#', '#', '#', '.'],
    ['.', '#', '#', '.', '.', '#', '.'],
    ['.', '.', '.', '#', '.', '#', '.'],
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
epsilon_decay_rate = 0.05  # per episode; epsilon is ~0.09 by episode 50

episode_time_limit = 10    # seconds; an episode that takes longer counts as a timeout
FPS = 10                   # frames per second; the mouse makes one move per frame

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
    """Q-learning (Bellman) update."""
    old_value = q_table[state, action]
    # A terminal state has no future, so don't bootstrap from it.
    max_future_q = 0.0 if done else np.max(q_table[new_state, :])
    new_value = old_value + learning_rate * (reward + discount_factor * max_future_q - old_value)
    q_table[state, action] = new_value


clock = pygame.time.Clock()


def draw(screen, font, state, text):
    """Draws one frame. Returns False if the user quit."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            return False

    colors = {'S': "green", 'C': "yellow", '#': "black", '.': "white"}
    for row in range(maze_rows):
        for col in range(maze_cols):
            rect = (col * unitSize, row * unitSize, unitSize, unitSize)
            pygame.draw.rect(screen, colors[maze[row, col]], rect)
            pygame.draw.rect(screen, "gray80", rect, 1)
    pos = cell_center(*state_to_cell(state))
    pygame.draw.rect(screen, "gray40", pygame.Rect(0, 0, 50, 50).move(pos.x - 25, pos.y - 25), border_radius=10)
    label = font.render(text, True, "white", "black")
    screen.blit(label, (8, maze_rows * unitSize - label.get_height() - 8))
    pygame.display.flip()
    clock.tick(FPS)
    return True


def train(screen, font):
    """Trains until the user quits. Returns the episode and epsilon at that moment."""
    episode = 0
    while True:
        episode += 1
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        state, total_reward, done, steps = start_state, 0, False, 0
        start_time = pygame.time.get_ticks()

        while not done and pygame.time.get_ticks() - start_time < episode_time_limit * 1000:
            action = choose_action(state, epsilon)
            new_state, reward, done = step(state, action)
            update_q_table(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward
            steps += 1
            if not draw(screen, font, state, f"Episode {episode}  step {steps}  "
                                             f"reward {total_reward:.0f}  epsilon {epsilon:.2f}"):
                return episode, epsilon

        print(f"Episode {episode}: {'cheese' if done else 'timeout'} in {steps} steps, "
              f"reward {total_reward:.0f}, epsilon {epsilon:.3f}")


def main():
    pygame.init()
    screen = pygame.display.set_mode((maze_cols * unitSize, maze_rows * unitSize))
    pygame.display.set_caption("AI Smart Mouse")
    font = pygame.font.Font(None, 28)

    episode, epsilon = train(screen, font)
    pygame.quit()

    print(f"\nStopped during episode {episode}")
    print(f"Epsilon (chance of a random move): {epsilon:.3f}")
    print(f"Learning rate: {learning_rate}   Discount factor: {discount_factor}")
    print(f"Rewards: cheese {REWARD_CHEESE}, step {REWARD_STEP}, bump {REWARD_BUMP}")


if __name__ == "__main__":
    main()
