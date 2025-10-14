from index import get_maze, get_wall_rects, get_maze_dimensions, get_state_from_pos, get_pos_from_state
import numpy as np


maze_rows, maze_cols = get_maze_dimensions()

total_states = maze_rows * maze_cols  # 35 states
num_actions = 4  # up, down, left, right

# --- Q-Learning Parameters ---

# 1. Learning Rate (alpha)
# How quickly the AI accepts new information.
# A high value means it learns fast but can be unstable.
# A low value means it's more stubborn but learns more steadily.
learning_rate = 0.1 

# 2. Discount Factor (gamma)
# How much the AI values future rewards.
# A high value (close to 1) means it's patient and aims for the big long-term reward (the cheese).
# A low value (close to 0) means it's short-sighted and only cares about immediate rewards.
discount_factor = 0.95

# 3. Exploration vs. Exploitation (epsilon)
# The chance of the AI taking a random action instead of the "best" one it knows.
# This is crucial for discovering new, better paths.
# It starts high (lots of exploring) and gets lower over time.
epsilon = 1.0             # Starting exploration rate
max_epsilon = 1.0         # Max value
min_epsilon = 0.01        # Minimum value so it never stops exploring completely
epsilon_decay_rate = 0.0005 # How fast it stops exploring

# --- Q-Table Setup ---
total_states = maze_rows * maze_cols  # 35 states
num_actions = 4  # up, down, left, right

# Initialize Q-table
# Rows = states, Columns = actions
q_table = np.zeros((total_states, num_actions))

def choose_action(state, epsilon):
    """Decides whether to explore (random action) or exploit (best known action)."""
    
    # Generate a random number between 0 and 1
    exploration_tradeoff = np.random.uniform(0, 1)

    if exploration_tradeoff < epsilon:
        # EXPLORE: Take a random action
        action = np.random.choice(num_actions)
    else:
        # EXPLOIT: Take the best action for the current state from the Q-table
        action = np.argmax(q_table[state, :])
        
    return action

def update_q_table(state, action, reward, new_state):
    """Updates the Q-value for the state-action pair using the Bellman equation."""
    
    # The Q-Learning formula in code
    old_value = q_table[state, action]
    max_future_q = np.max(q_table[new_state, :]) # The best Q-value for the state we ended up in
    
    # The core formula
    new_value = old_value + learning_rate * (reward + discount_factor * max_future_q - old_value)
    
    q_table[state, action] = new_value



