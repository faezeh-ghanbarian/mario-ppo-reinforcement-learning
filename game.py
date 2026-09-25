import pygame
import sys

import config

pygame.init()

# -------------------------
# Window
# -------------------------
WIDTH = config.WIDTH
HEIGHT = config.HEIGHT

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mario PPO Challenge")

clock = pygame.time.Clock()

# -------------------------
# Player
# -------------------------
START_X = config.START_X
START_Y = config.START_Y

player_x = float(START_X)
player_y = float(START_Y)

PLAYER_WIDTH = config.PLAYER_WIDTH
PLAYER_HEIGHT = config.PLAYER_HEIGHT

velocity_x = 0
velocity_y = 0

MOVE_SPEED = config.MOVE_SPEED
JUMP_FORWARD_SPEED = config.JUMP_FORWARD_SPEED
JUMP_SPEED = config.JUMP_SPEED
GRAVITY = config.GRAVITY

on_ground = False

# -------------------------
# Level
# -------------------------
GROUND_Y = config.GROUND_Y

PIT_START = config.PIT_START
PIT_END = config.PIT_END

left_ground = pygame.Rect(
    0,
    GROUND_Y,
    PIT_START,
    HEIGHT - GROUND_Y
)

right_ground = pygame.Rect(
    PIT_END,
    GROUND_Y,
    WIDTH - PIT_END,
    HEIGHT - GROUND_Y
)

platforms = [left_ground, right_ground]

# -------------------------
# Enemy
# -------------------------
ENEMY_WIDTH = config.ENEMY_WIDTH
ENEMY_HEIGHT = config.ENEMY_HEIGHT

enemy_x = float(config.ENEMY_START_X)
enemy_y = GROUND_Y - ENEMY_HEIGHT

enemy_speed = float(config.ENEMY_SPEED)

enemy_min_x = config.ENEMY_MIN_X
enemy_max_x = config.ENEMY_MAX_X

# -------------------------
# Hazard / Spike
# -------------------------
SPIKE_X = config.SPIKE_X
SPIKE_WIDTH = config.SPIKE_WIDTH
SPIKE_HEIGHT = config.SPIKE_HEIGHT

spike_rect = pygame.Rect(
    SPIKE_X,
    GROUND_Y - SPIKE_HEIGHT,
    SPIKE_WIDTH,
    SPIKE_HEIGHT
)

# -------------------------
# Goal / Flag
# -------------------------
FLAG_X = config.FLAG_X
FLAG_WIDTH = config.FLAG_WIDTH
FLAG_HEIGHT = config.FLAG_HEIGHT

flag_rect = pygame.Rect(
    FLAG_X,
    GROUND_Y - FLAG_HEIGHT,
    FLAG_WIDTH,
    FLAG_HEIGHT
)

# -------------------------
# Reset function
# -------------------------
def reset_player():
    global player_x, player_y
    global velocity_x, velocity_y
    global on_ground

    player_x = float(START_X)
    player_y = float(START_Y)

    velocity_x = 0
    velocity_y = 0

    on_ground = False


# -------------------------
# Game loop
# -------------------------
running = True

while running:

    # -------------------------
    # Events
    # -------------------------
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE and on_ground:
                velocity_y = JUMP_SPEED
                velocity_x = JUMP_FORWARD_SPEED
                on_ground = False

    # -------------------------
    # Keyboard movement
    # -------------------------
    keys = pygame.key.get_pressed()

    if on_ground:

        velocity_x = 0

        if keys[pygame.K_LEFT]:
            velocity_x = -MOVE_SPEED

        if keys[pygame.K_RIGHT]:
            velocity_x = MOVE_SPEED

    else:

        if keys[pygame.K_RIGHT]:
            velocity_x = JUMP_FORWARD_SPEED

        if keys[pygame.K_LEFT]:
            velocity_x = -JUMP_FORWARD_SPEED

    # -------------------------
    # Previous position
    # -------------------------
    previous_bottom = player_y + PLAYER_HEIGHT

    # -------------------------
    # Physics
    # -------------------------
    velocity_y += GRAVITY

    player_x += velocity_x
    player_y += velocity_y

    player_rect = pygame.Rect(
        int(player_x),
        int(player_y),
        PLAYER_WIDTH,
        PLAYER_HEIGHT
    )

    # -------------------------
    # Ground collision
    # -------------------------
    on_ground = False

    if velocity_y >= 0:

        for platform in platforms:

            horizontal_overlap = (
                player_rect.right > platform.left
                and player_rect.left < platform.right
            )

            crossed_platform_top = (
                previous_bottom <= platform.top
                and player_rect.bottom >= platform.top
            )

            if horizontal_overlap and crossed_platform_top:

                player_y = platform.top - PLAYER_HEIGHT
                velocity_y = 0
                on_ground = True

                break

    # -------------------------
    # Screen boundaries
    # -------------------------
    if player_x < 0:
        player_x = 0

    if player_x + PLAYER_WIDTH > WIDTH:
        player_x = WIDTH - PLAYER_WIDTH

    # -------------------------
    # Pit death
    # -------------------------
    if player_y > HEIGHT:

        print("Player fell into the pit! Resetting...")
        reset_player()

    # -------------------------
    # Move enemy
    # -------------------------
    enemy_x += enemy_speed

    if enemy_x <= enemy_min_x or enemy_x >= enemy_max_x:
        enemy_speed *= -1

    enemy_rect = pygame.Rect(
        int(enemy_x),
        int(enemy_y),
        ENEMY_WIDTH,
        ENEMY_HEIGHT
    )

    # -------------------------
    # Player rectangle again
    # -------------------------
    player_rect = pygame.Rect(
        int(player_x),
        int(player_y),
        PLAYER_WIDTH,
        PLAYER_HEIGHT
    )

    # -------------------------
    # Enemy collision
    # -------------------------
    if player_rect.colliderect(enemy_rect):

        print("Player hit the enemy! Resetting...")
        reset_player()

    # -------------------------
    # Spike collision
    # -------------------------
    if player_rect.colliderect(spike_rect):

        print("Player hit the spike! Resetting...")
        reset_player()

    # -------------------------
    # Goal collision
    # -------------------------
    if player_rect.colliderect(flag_rect):

        print("LEVEL COMPLETE!")
        reset_player()

    # -------------------------
    # Draw
    # -------------------------
    screen.fill((150, 200, 255))

    # Ground
    pygame.draw.rect(
        screen,
        (50, 180, 70),
        left_ground
    )

    pygame.draw.rect(
        screen,
        (50, 180, 70),
        right_ground
    )

    # Enemy
    pygame.draw.rect(
        screen,
        (120, 70, 30),
        enemy_rect
    )

    # Spike
    spike_points = [
        (SPIKE_X, GROUND_Y),
        (SPIKE_X + SPIKE_WIDTH // 2, GROUND_Y - SPIKE_HEIGHT),
        (SPIKE_X + SPIKE_WIDTH, GROUND_Y)
    ]

    pygame.draw.polygon(
        screen,
        (80, 80, 80),
        spike_points
    )

    # Flag pole
    pygame.draw.rect(
        screen,
        (40, 40, 40),
        flag_rect
    )

    # Flag cloth
    pygame.draw.polygon(
        screen,
        (255, 215, 0),
        [
            (FLAG_X, GROUND_Y - FLAG_HEIGHT),
            (FLAG_X + 45, GROUND_Y - FLAG_HEIGHT + 15),
            (FLAG_X, GROUND_Y - FLAG_HEIGHT + 30)
        ]
    )

    # Player
    pygame.draw.rect(
        screen,
        (220, 60, 60),
        (
            int(player_x),
            int(player_y),
            PLAYER_WIDTH,
            PLAYER_HEIGHT
        )
    )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()