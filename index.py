# Example file showing a basic pygame "game loop"
import pygame
import numpy as np

# Global variables for maze setup
screen = pygame.display.set_mode((700, 500))

unitSize = 100
mousePos = pygame.Vector2(unitSize / 4, unitSize / 4)

maze = np.array([
    ['S', '#', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '#', '#', '#', '.'],
    ['.', '#', '#', '.', '.', '#', '.'],
    ['.', '.', '.', '#', '.', '#', '#'],
    ['.', '#', '.', '.', '.', '.', 'C']
])

maze_rows, maze_cols = maze.shape

# Discretization functions
def get_state_from_pos(mouse_pos):
    """Converts pixel coordinates to a single state number."""
    col = int(mouse_pos.x // unitSize)
    row = int(mouse_pos.y // unitSize)
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

# Export functions for other modules

def get_wall_rects():
    wall_rects = []
    for row_idx, row in enumerate(maze):
        for col_idx, cell in enumerate(row):
            if cell == '#':
                x = col_idx * unitSize
                y = row_idx * unitSize
                wall_rects.append(pygame.Rect(x, y, unitSize, unitSize))
    return wall_rects

def get_maze_dimensions():
    return maze_rows, maze_cols

#To visualize
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

# pygame setup
def main():
    pygame.init()
    clock = pygame.time.Clock()
    running = True
    dt = 0
    
    # Get maze dimensions for state calculations
    total_states = maze_rows * maze_cols  # This will be 35 states (5 rows × 7 cols)

    # Needed to not make the mouse pass through walls
    wall_rects = get_wall_rects()

    # Track current state for comparison
    previous_state = get_state_from_pos(mousePos)
    
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        drawMap()

        # Get current state and check if it changed
        current_state = get_state_from_pos(mousePos)
        if current_state != previous_state:
            row = current_state // maze_cols
            col = current_state % maze_cols
            cell_type = maze[row][col]
            print(f"State changed: {previous_state} -> {current_state} (Row: {row}, Col: {col}, Cell: '{cell_type}')")
            previous_state = current_state

        # RENDER YOUR GAME HERE
        pygame.draw.rect(screen, "gray", (mousePos.x, mousePos.y, 50, 50))
        

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            next_rect = pygame.Rect(mousePos.x, mousePos.y - 300 * dt, 50, 50)
            collision_check = next_rect.collidelist(wall_rects)
            if (collision_check == -1) and (next_rect.top >= 0):
                mousePos.y -= 300 * dt
        if keys[pygame.K_s]:
            next_rect = pygame.Rect(mousePos.x, mousePos.y + 300 * dt, 50, 50)
            collision_check = next_rect.collidelist(wall_rects)
            if (collision_check == -1) and (next_rect.bottom <= screen.get_height()):
                mousePos.y += 300 * dt
        if keys[pygame.K_a]:
            next_rect = pygame.Rect(mousePos.x - 300 * dt, mousePos.y, 50, 50)
            collision_check = next_rect.collidelist(wall_rects)
            if (collision_check == -1) and (next_rect.left >= 0):
                mousePos.x -= 300 * dt
            
        if keys[pygame.K_d]:
            next_rect = pygame.Rect(mousePos.x + 300 * dt, mousePos.y, 50, 50)
            collision_check = next_rect.collidelist(wall_rects)
            if (collision_check == -1) and (next_rect.right <= screen.get_width()):
                mousePos.x += 300 * dt

        # flip() the display to put your work on screen
        pygame.display.flip()

        dt = clock.tick(60) / 1000

    pygame.quit()

if __name__ == "__main__":
    main()

