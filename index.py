# Example file showing a basic pygame "game loop"
import pygame
import numpy as np

# pygame setup
pygame.init()
screen = pygame.display.set_mode((700, 500))
clock = pygame.time.Clock()
running = True
dt = 0
unitSize = 100
mousePos = pygame.Vector2(unitSize / 4, unitSize / 4)

maze = np.array([
    ['S', '#', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '#', '#', '#', '.'],
    ['.', '#', '#', '.', '.', '#', '.'],
    ['.', '.', '.', '#', '.', '#', '#'],
    ['.', '#', '.', '.', '.', '.', 'C']
])

wall_rects = []
for row_idx, row in enumerate(maze):
    for col_idx, cell in enumerate(row):
        if cell == '#':
            x = col_idx * unitSize
            y = row_idx * unitSize
            wall_rects.append(pygame.Rect(x, y, unitSize, unitSize))


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
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")
    drawMap()

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

