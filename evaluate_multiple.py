from stable_baselines3 import PPO

import os
import csv

import config
from env import MarioEnv


# -------------------------
# Create folders
# -------------------------
os.makedirs("results", exist_ok=True)


# -------------------------
# Create environment
# -------------------------
env = MarioEnv()


# -------------------------
# Load trained model
# -------------------------
model = PPO.load(
    config.MODEL_PATH,
    env=env
)


# -------------------------
# Statistics
# -------------------------
success_count = 0
pit_pass_count = 0
death_count = 0

total_rewards = []
max_x_positions = []

episode_results = []


print()
print("=" * 60)
print("MULTIPLE EPISODE EVALUATION")
print("=" * 60)
print()


# -------------------------
# Run evaluation episodes
# -------------------------
for episode in range(1, config.NUM_EVAL_EPISODES + 1):

    observation, info = env.reset(
        seed=config.SEED + episode
    )

    total_reward = 0.0
    max_x = 0.0

    passed_pit = False
    reached_goal = False
    death_reason = None


    for step in range(config.EVAL_MAX_STEPS):

        action, _ = model.predict(
            observation,
            deterministic=True
        )


        observation, reward, terminated, truncated, info = env.step(
            action
        )


        total_reward += reward

        x_position = info["x_position"]

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


    total_rewards.append(total_reward)
    max_x_positions.append(max_x)


    if passed_pit:
        pit_pass_count += 1


    if reached_goal:
        success_count += 1


    if death_reason is not None:
        death_count += 1


    if reached_goal:
        result = "SUCCESS"

    elif passed_pit:
        result = "PASSED PIT"

    else:
        result = "FAILED"


    episode_results.append(
        [
            episode,
            total_reward,
            max_x,
            passed_pit,
            reached_goal,
            death_reason,
            result
        ]
    )


    print(
        f"Episode {episode:02d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Max X: {max_x:6.1f} | "
        f"Result: {result}"
    )


# -------------------------
# Final statistics
# -------------------------
success_rate = (
    success_count / config.NUM_EVAL_EPISODES
) * 100


pit_pass_rate = (
    pit_pass_count / config.NUM_EVAL_EPISODES
) * 100


average_reward = (
    sum(total_rewards) /
    config.NUM_EVAL_EPISODES
)


average_max_x = (
    sum(max_x_positions) /
    config.NUM_EVAL_EPISODES
)


# -------------------------
# Save results CSV
# -------------------------
with open(
    "results/evaluation_results.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Episode",
            "Reward",
            "Max_X",
            "Passed_Pit",
            "Reached_Goal",
            "Death_Reason",
            "Result"
        ]
    )

    writer.writerows(
        episode_results
    )


# -------------------------
# Print summary
# -------------------------
print()
print("=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(
    f"Episodes            : {config.NUM_EVAL_EPISODES}"
)

print(
    f"Successful episodes : {success_count}"
)

print(
    f"Success rate        : {success_rate:.2f}%"
)

print()

print(
    f"Pit passes          : {pit_pass_count}"
)

print(
    f"Pit pass rate       : {pit_pass_rate:.2f}%"
)

print()

print(
    f"Deaths              : {death_count}"
)

print()

print(
    f"Average reward      : {average_reward:.2f}"
)

print(
    f"Average maximum X   : {average_max_x:.2f}"
)

print()

print(
    "Results saved to results/evaluation_results.csv"
)

print("=" * 60)


env.close()