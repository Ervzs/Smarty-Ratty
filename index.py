# Example file showing a basic pygame "game loop"
import pygame
import numpy as np



unitSize = 100
maze = np.array([
    ['S', '#', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '#', '#', '#', '.'],
    ['.', '#', '#', '.', '.', '#', '.'],
    ['.', '.', '.', '#', '.', '#', '#'],
    ['.', '#', '.', '.', '.', '.', 'C']
])

maze_rows, maze_cols = maze.shape

# Discretization functions
def get_state_from_pos(mousePos):
    """Converts pixel coordinates to a single state number."""
    col = int(mousePos.x // unitSize)
    row = int(mousePos.y // unitSize)
    # Clamp to valid bounds
    col = max(0, min(col, maze_cols - 1))
    row = max(0, min(row, maze_rows - 1))
    return row * maze_cols + col

def get_pos_from_state(state):
    """Converts state number back to pixel coordinates."""
    row = state // maze_cols
    col = state % maze_cols
    # Return center of the grid cell
    x = col * unitSize + unitSize // 2
    y = row * unitSize + unitSize // 2
    return pygame.Vector2(x, y)

def get_wall_rects():
    wall_rects = []
    for row_idx, row in enumerate(maze):
        for col_idx, cell in enumerate(row):
            if cell == '#':
                x = col_idx * unitSize
                y = row_idx * unitSize
                wall_rects.append(pygame.Rect(x, y, unitSize, unitSize))
    return wall_rects


#To visualize the cells (Optional)
def drawGrid():
    x = 0
    y = 0
    screenWidth = screen.get_width()
    screenHeight = screen.get_height()
    #For Vertical Lines
    while (x < screenWidth):
        pygame.draw.line(screen, "black", (x, 0), (x, screenHeight))
        x += unitSize

    #For Horizontal Lines
    while (y < screenHeight):
        pygame.draw.line(screen, "black", (0, y), (screenWidth, y))
        y += unitSize

# SETUP FOR Q-LEARNING

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

# pygame setup
def main():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((700, 500))
    
    running = True
    global epsilon

    
    # Timer for each episode (20 seconds = 20000 milliseconds)
    current_episode = 0
    episode_time_limit = 20000  # 20 seconds in milliseconds
    episode_start_time = pygame.time.get_ticks()  # Track when episode started

    # Reset position for each episode
    start_pos_coords = np.argwhere(maze == 'S')[0]
    start_row, start_col = start_pos_coords
    mousePos = pygame.Vector2(start_col * unitSize + unitSize // 2, start_row * unitSize + unitSize // 2)
    
    cheese_pos_coords = np.argwhere(maze == 'C')[0]
    cheese_row, cheese_col = cheese_pos_coords
    cheesePos = pygame.Vector2(cheese_col * unitSize + unitSize // 2, cheese_row * unitSize + unitSize // 2)

    # Track current state for comparison
    previous_state = get_state_from_pos(mousePos)

    # Needed to not make the mouse pass through walls
    wall_rects = get_wall_rects()
    episode_done = False

    #Added visuals for the np maze 
    def drawMap():
        for row_idx, row in enumerate(maze):
            for col_idx, cell in enumerate(row):
                x = col_idx * unitSize
                y = row_idx * unitSize
                match cell:
                    case 'S':
                        pygame.draw.rect(screen, "green", (x, y, unitSize, unitSize))
                    case 'C':
                        pygame.draw.rect(screen, "yellow", (x, y, unitSize, unitSize))
                    case '#':
                        pygame.draw.rect(screen, "black", (x, y, unitSize, unitSize))
                    case '.':
                        pygame.draw.rect(screen, "white", (x, y, unitSize, unitSize))

    
    while running:
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check if 20 seconds have passed for this episode
        current_time = pygame.time.get_ticks()
        time_elapsed = current_time - episode_start_time
        
        # If 20 seconds passed, force episode reset
        if time_elapsed >= episode_time_limit:
            episode_done = True
            print(f"Episode {current_episode}: Time limit reached (20 seconds)")

        screen.fill("white")

        drawMap()
        pygame.draw.rect(screen, "gray", (mousePos.x, mousePos.y, 50, 50))

        # Get current state and check if it changed
        current_state = get_state_from_pos(mousePos)
        if current_state != previous_state:
            row = current_state // maze_cols
            col = current_state % maze_cols
            cell_type = maze[row][col]
            print(f"State changed: {previous_state} -> {current_state} (Row: {row}, Col: {col}, Cell: '{cell_type}')")
            previous_state = current_state

        if not episode_done:
            # First get the current state
            state = get_state_from_pos(mousePos)
            # Then choose an action
            action = choose_action(state, epsilon)
            # Perform the action which replaced the movements keys (AI moves 3  pixels per second)
            old_pos = mousePos.copy() # This will check if the AI move

            # ----- MOVEMENT LOGIC -----
            # We check boundaries BEFORE moving to prevent going outside the screen
            # Each action moves 3 pixels in the respective direction

            if action == 0: #UP
                next_rect = pygame.Rect(mousePos.x, mousePos.y - 3, 50, 50) 
                if next_rect.top >= 0.1 and next_rect.collidelist(wall_rects) == -1:
                    mousePos.y -= 3
            elif action == 1: #DOWN
                next_rect = pygame.Rect(mousePos.x, mousePos.y + 3, 50, 50) 
                if next_rect.bottom <= screen.get_height() and next_rect.collidelist(wall_rects) == -1:
                    mousePos.y += 3
            elif action == 2: #LEFT
                next_rect = pygame.Rect(mousePos.x - 3, mousePos.y, 50, 50) 
                if next_rect.left >= 0 and next_rect.collidelist(wall_rects) == -1:
                    mousePos.x -= 3
            elif action == 3: #RIGHT
                next_rect = pygame.Rect(mousePos.x + 3, mousePos.y, 50, 50) 
                if next_rect.right <= screen.get_width() and next_rect.collidelist(wall_rects) == -1:
                    mousePos.x += 3

            # After moving we need to track the new state
            new_state = get_state_from_pos(mousePos)


            # ===== REWARD SYSTEM =====
            # Reward system that will improve the AI

            # Calculate Manhattan distance to cheese (used for distance-based rewards)
            # Manhattan distance = |x1 - x2| + |y1 - y2|
            current_distance = abs(mousePos.x - cheesePos.x) + abs(mousePos.y - cheesePos.y)

            # Dead states
            dead_end_states = [
                (2, 6),
                (2, 3),
                (4, 0)
            ]

            # Close to the goal states
            closer_states = [
                (3,1),
                (4,4),
                (2,0)
            ]

            # Convert current state to (row, col) for checking
            current_row = new_state // maze_cols
            current_col = new_state % maze_cols

            reward = 0

            # 1. GOAL REACHED: Highest reward for reaching the cheese
            if new_state == get_state_from_pos(cheesePos):
                reward += 100
                episode_done = True
                print(f"Episode {current_episode}: Cheese found! 🧀")

            # 2. COLLISION PENALTY: AI tried to move but couldn't (hit wall or boundary)   
            elif old_pos == mousePos : # meaning AI didnt move. Needs consequence.
                reward -= 10

            # 3. DEAD END PENALTY: AI entered a known dead-end state
            elif (current_row, current_col) in dead_end_states:
                reward -= 50
                # Heavy penalty to discourage exploring dead ends


            # 4. Closer to the goal reward AI
            elif (current_row, current_col) in closer_states:
                reward += 10
            else:
                reward -= 0.1

            # Store current distance for next iteration
            main.previous_distance = current_distance

            # 5. REVISITING STATES: Track how many times we've visited this state
            # Initialize state visit counter if it doesn't exist
            if not hasattr(main, 'state_visit_count'):
                main.state_visit_count = {}
            
            # Increment visit count for this state
            
            if new_state not in main.state_visit_count:
                main.state_visit_count[new_state] = 1
            else:
                main.state_visit_count[new_state] += 1
            
            # Penalize revisiting the same state too many times (looping behavior)
            if main.state_visit_count[new_state] > 2:
                reward -= 5  # Penalty for revisiting a state more than 2 times
            
            # 6. TIME IN SAME STATE: Track if AI is stuck in one state too long
            if not hasattr(main, 'state_time_tracker'):
                main.state_time_tracker = {'state': new_state, 'frames': 0}
            
            # If we're in the same state as before, increment frame counter
            if main.state_time_tracker['state'] == new_state:
                main.state_time_tracker['frames'] += 1
            else:
                # Reset counter when we move to a new state
                main.state_time_tracker = {'state': new_state, 'frames': 0}
            

            # If stuck in same state for ~10 seconds (600 frames at 60 FPS)
            if main.state_time_tracker['frames'] > 600:
                reward -= 20  # Heavy penalty for being stuck
                print(f"Episode {current_episode}: Stuck too long, restarting...")
                # Immediately reset agent position and trackers
                mousePos = pygame.Vector2(start_col * unitSize + unitSize // 2, start_row * unitSize + unitSize // 2)
                main.state_time_tracker = {'state': get_state_from_pos(mousePos), 'frames': 0}
                main.state_visit_count = {}
                # Optionally reset previous_distance if you use it
                main.previous_distance = abs(mousePos.x - cheesePos.x) + abs(mousePos.y - cheesePos.y)
                continue  # Skip the rest of this frame and start fresh
            
            #UPDATE Q-TABLE
            update_q_table(state, action, reward, new_state)

                
        # Reset for next episode (happens when goal is reached OR 20 seconds pass)
        # Update epsilon after each episode to make AI more confident over time
        if episode_done:
            current_episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * current_episode)
            
            # Reset position
            mousePos = pygame.Vector2(start_col * unitSize + unitSize // 2, start_row * unitSize + unitSize // 2)
            episode_done = False

            # Reset episode timer for new episode
            episode_start_time = pygame.time.get_ticks()
            
            # Reset state trackers
            main.state_time_tracker = {'state': get_state_from_pos(mousePos), 'frames': 0}
            main.state_visit_count = {}

            # Print progress every 2 episodes
            if current_episode % 2 == 0:
                print(f"Episode: {current_episode}, Epsilon: {epsilon:.4f}")

        # flip() the display to put your work on screen
        pygame.display.flip()
        dt = clock.tick(60) / 1000

    pygame.quit()

if __name__ == "__main__":
    main()

