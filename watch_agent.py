import time
import pygame

from stable_baselines3 import PPO

from env import MarioEnv
import config


# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = "models/ppo_mario_v3_20k"

model = PPO.load(MODEL_PATH)


# -------------------------
# Create RL environment
# -------------------------
env = MarioEnv()

observation, info = env.reset()


# -------------------------
# Pygame setup
# -------------------------
pygame.init()

screen = pygame.display.set_mode(
    (config.WIDTH, config.HEIGHT)
)

pygame.display.set_caption(
    "PPO Mario Agent Demo"
)

clock = pygame.time.Clock()


# -------------------------
# Level rectangles
# -------------------------
left_ground = pygame.Rect(
    0,
    config.GROUND_Y,
    config.PIT_START,
    config.HEIGHT - config.GROUND_Y
)

right_ground = pygame.Rect(
    config.PIT_END,
    config.GROUND_Y,
    config.WIDTH - config.PIT_END,
    config.HEIGHT - config.GROUND_Y
)

spike_rect = pygame.Rect(
    config.SPIKE_X,
    config.GROUND_Y - config.SPIKE_HEIGHT,
    config.SPIKE_WIDTH,
    config.SPIKE_HEIGHT
)

flag_rect = pygame.Rect(
    config.FLAG_X,
    config.GROUND_Y - config.FLAG_HEIGHT,
    config.FLAG_WIDTH,
    config.FLAG_HEIGHT
)


# -------------------------
# Demo loop
# -------------------------
running = True

while running:

    # -------------------------
    # Window events
    # -------------------------
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


    # -------------------------
    # PPO chooses action
    # -------------------------
    action, _ = model.predict(
        observation,
        deterministic=True
    )


    # -------------------------
    # Environment step
    # -------------------------
    observation, reward, terminated, truncated, info = env.step(
        action
    )


    # -------------------------
    # Draw background
    # -------------------------
    screen.fill(
        (150, 200, 255)
    )


    # -------------------------
    # Draw ground
    # -------------------------
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


    # -------------------------
    # Draw enemy
    # -------------------------
    enemy_rect = pygame.Rect(
        int(env.enemy_x),
        int(env.enemy_y),
        config.ENEMY_WIDTH,
        config.ENEMY_HEIGHT
    )

    pygame.draw.rect(
        screen,
        (120, 70, 30),
        enemy_rect
    )


    # -------------------------
    # Draw spike
    # -------------------------
    spike_points = [
        (
            config.SPIKE_X,
            config.GROUND_Y
        ),
        (
            config.SPIKE_X + config.SPIKE_WIDTH // 2,
            config.GROUND_Y - config.SPIKE_HEIGHT
        ),
        (
            config.SPIKE_X + config.SPIKE_WIDTH,
            config.GROUND_Y
        )
    ]

    pygame.draw.polygon(
        screen,
        (80, 80, 80),
        spike_points
    )


    # -------------------------
    # Draw flag
    # -------------------------
    pygame.draw.rect(
        screen,
        (40, 40, 40),
        flag_rect
    )

    pygame.draw.polygon(
        screen,
        (255, 215, 0),
        [
            (
                config.FLAG_X,
                config.GROUND_Y - config.FLAG_HEIGHT
            ),
            (
                config.FLAG_X + 45,
                config.GROUND_Y - config.FLAG_HEIGHT + 15
            ),
            (
                config.FLAG_X,
                config.GROUND_Y - config.FLAG_HEIGHT + 30
            )
        ]
    )


    # -------------------------
    # Draw player
    # -------------------------
    player_rect = pygame.Rect(
        int(env.player_x),
        int(env.player_y),
        config.PLAYER_WIDTH,
        config.PLAYER_HEIGHT
    )

    pygame.draw.rect(
        screen,
        (220, 60, 60),
        player_rect
    )


    # -------------------------
    # Status text
    # -------------------------
    font = pygame.font.Font(
        None,
        28
    )

    status_text = (
        f"X: {env.player_x:.0f}   "
        f"Action: {int(action)}   "
        f"Reward: {reward:.2f}"
    )

    text_surface = font.render(
        status_text,
        True,
        (20, 20, 20)
    )

    screen.blit(
        text_surface,
        (20, 20)
    )


    # -------------------------
    # Update screen
    # -------------------------
    pygame.display.flip()

    clock.tick(30)


    # -------------------------
    # Episode finished
    # -------------------------
    if terminated or truncated:

        if info.get(
            "reached_goal",
            False
        ):

            print()
            print(
                "SUCCESS - PPO AGENT REACHED THE GOAL!"
            )

        else:

            print()
            print(
                "Episode finished."
            )

            print(
                "Reason:",
                info.get("death_reason")
            )

        # Keep final frame visible briefly
        time.sleep(2)

        running = False


# -------------------------
# Cleanup
# -------------------------
env.close()

pygame.quit()