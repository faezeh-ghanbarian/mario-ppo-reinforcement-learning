import random
import numpy as np

from env import MarioEnv


# -------------------------
# Evaluation settings
# -------------------------
NUM_EPISODES = 50
MAX_STEPS = 500
SEED = 42


# -------------------------
# Reproducibility
# -------------------------
random.seed(SEED)
np.random.seed(SEED)


# -------------------------
# Create environment
# -------------------------
env = MarioEnv()


# -------------------------
# Store results
# -------------------------
episode_rewards = []
episode_max_x = []

successful_episodes = 0
pit_passes = 0
deaths = 0


print()
print("=" * 60)
print("RANDOM AGENT BASELINE EVALUATION")
print("=" * 60)
print(f"Episodes : {NUM_EPISODES}")
print(f"Max steps: {MAX_STEPS}")
print(f"Seed     : {SEED}")
print()


# -------------------------
# Run evaluation episodes
# -------------------------
for episode in range(NUM_EPISODES):

    episode_seed = SEED + episode

    # Reset environment with a fixed seed
    observation, info = env.reset(
        seed=episode_seed
    )

    # Seed the action space as well.
    # This makes action_space.sample()
    # reproducible across repeated runs.
    env.action_space.seed(
        episode_seed
    )

    total_reward = 0.0
    max_x = 0.0

    passed_pit = False
    reached_goal = False
    death_reason = None


    for step in range(MAX_STEPS):

        # Random baseline:
        # choose an action uniformly from the action space
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        x_position = float(
            info.get("x_position", 0.0)
        )

        max_x = max(
            max_x,
            x_position
        )


        passed_pit = (
            passed_pit
            or info.get("passed_pit", False)
        )

        reached_goal = (
            reached_goal
            or info.get("reached_goal", False)
        )


        if info.get("death_reason") is not None:
            death_reason = info["death_reason"]


        if terminated or truncated:
            break


    # -------------------------
    # Save episode statistics
    # -------------------------
    episode_rewards.append(
        total_reward
    )

    episode_max_x.append(
        max_x
    )


    if reached_goal:
        successful_episodes += 1


    if passed_pit:
        pit_passes += 1


    if death_reason is not None:
        deaths += 1


    # -------------------------
    # Print episode result
    # -------------------------
    print(
        f"Episode {episode + 1:02d}/{NUM_EPISODES} | "
        f"Seed: {episode_seed:3d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Max X: {max_x:6.1f} | "
        f"Pit: {str(passed_pit):5s} | "
        f"Goal: {str(reached_goal):5s} | "
        f"Death: {death_reason}"
    )


# -------------------------
# Calculate summary
# -------------------------
success_rate = (
    successful_episodes / NUM_EPISODES
) * 100.0


pit_pass_rate = (
    pit_passes / NUM_EPISODES
) * 100.0


average_reward = float(
    np.mean(episode_rewards)
)


std_reward = float(
    np.std(episode_rewards)
)


average_max_x = float(
    np.mean(episode_max_x)
)


std_max_x = float(
    np.std(episode_max_x)
)


# -------------------------
# Print final results
# -------------------------
print()
print("=" * 60)
print("RANDOM BASELINE RESULTS")
print("=" * 60)

print(f"Total episodes      : {NUM_EPISODES}")
print(f"Base seed           : {SEED}")

print()

print(f"Successful episodes : {successful_episodes}")
print(f"Success rate        : {success_rate:.2f}%")

print()

print(f"Pit passes          : {pit_passes}")
print(f"Pit pass rate       : {pit_pass_rate:.2f}%")

print()

print(f"Deaths              : {deaths}")

print()

print(f"Average reward      : {average_reward:.2f}")
print(f"Reward std          : {std_reward:.2f}")

print()

print(f"Average maximum X   : {average_max_x:.2f}")
print(f"Maximum X std       : {std_max_x:.2f}")

print("=" * 60)


# -------------------------
# Close environment
# -------------------------
env.close()