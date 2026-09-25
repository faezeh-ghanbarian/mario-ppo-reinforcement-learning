import os

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Paths
# -------------------------
LOG_PATH = "logs/training_v3_20k.monitor.csv"
OUTPUT_DIR = "results"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "learning_curve_v3.png"
)


# -------------------------
# Create results folder
# -------------------------
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# -------------------------
# Read Stable-Baselines3 monitor log
# -------------------------
data = pd.read_csv(
    LOG_PATH,
    skiprows=1
)


# -------------------------
# Calculate cumulative timesteps
# -------------------------
data["timesteps"] = data["l"].cumsum()


# -------------------------
# Moving average
# -------------------------
WINDOW = 10

data["reward_moving_average"] = (
    data["r"]
    .rolling(
        window=WINDOW,
        min_periods=1
    )
    .mean()
)


# -------------------------
# Print basic information
# -------------------------
print()
print("=" * 60)
print("TRAINING LOG SUMMARY")
print("=" * 60)

print(
    "Episodes:",
    len(data)
)

print(
    "Total timesteps:",
    int(data["timesteps"].iloc[-1])
)

print(
    "Final moving-average reward:",
    round(
        data["reward_moving_average"].iloc[-1],
        2
    )
)

print("=" * 60)
print()


# -------------------------
# Plot learning curve
# -------------------------
plt.figure(
    figsize=(10, 6)
)

# Raw episode rewards
plt.plot(
    data["timesteps"],
    data["r"],
    alpha=0.35,
    label="Episode reward"
)

# Smoothed reward
plt.plot(
    data["timesteps"],
    data["reward_moving_average"],
    linewidth=2,
    label=f"{WINDOW}-episode moving average"
)

plt.xlabel(
    "Training Timesteps"
)

plt.ylabel(
    "Episode Reward"
)

plt.title(
    "PPO Learning Curve - Mario Platformer"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# -------------------------
# Save figure
# -------------------------
plt.savefig(
    OUTPUT_PATH,
    dpi=200
)

print(
    "Learning curve saved to:"
)

print(
    OUTPUT_PATH
)


# -------------------------
# Show plot
# -------------------------
plt.show()