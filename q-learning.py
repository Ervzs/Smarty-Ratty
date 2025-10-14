from index import get_maze, get_wall_rects, get_maze_dimensions, get_state_from_pos, get_pos_from_state
import numpy as np


maze_rows, maze_cols = get_maze_dimensions()

alpha = 0.9
gamma = 0.95
epsilon = 1.0 #1 means its actions are random and 0 means it knows its actions
epsilon_decay = 0.9995 #Will be used to be multiplied to epsilon
min_epsilon = 0.05
num_episodes = 10000 # How many times the agent will play the game
max_steps = 100

total_states = maze_rows * maze_cols  # 35 states
num_actions = 4  # up, down, left, right

# Initialize Q-table
Q = np.zeros((total_states, num_actions))

print(f"Q-Learning setup complete:")
print(f"Maze dimensions: {maze_rows}x{maze_cols}")
print(f"Total states: {total_states}")
print(f"Q-table shape: {Q.shape}")

# TODO: Add Q-Learning functions here
# Example usage:
# current_state = get_state_from_pos(mousePos)
# action = choose_action(current_state, Q, epsilon)
# new_mousePos = get_pos_from_state(new_state)



