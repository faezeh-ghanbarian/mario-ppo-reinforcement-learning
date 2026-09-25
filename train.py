import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import config
from env import MarioEnv


# -------------------------
# Create folders
# -------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# -------------------------
# Create environment
# -------------------------
env = MarioEnv()

env = Monitor(
    env,
    filename=config.TRAIN_LOG_PATH
)


# -------------------------
# Create PPO agent
# -------------------------
model = PPO(
    policy="MlpPolicy",
    env=env,

    learning_rate=config.LEARNING_RATE,

    n_steps=config.N_STEPS,
    batch_size=config.BATCH_SIZE,

    gamma=config.GAMMA,
    gae_lambda=config.GAE_LAMBDA,

    clip_range=config.CLIP_RANGE,
    ent_coef=config.ENT_COEF,

    verbose=1,
    seed=config.SEED
)


# -------------------------
# Train
# -------------------------
print()
print(
    f"Starting PPO training for {config.TOTAL_TIMESTEPS} timesteps..."
)
print()

start_time = time.perf_counter()

model.learn(
    total_timesteps=config.TOTAL_TIMESTEPS,
    progress_bar=False
)

end_time = time.perf_counter()

training_time_seconds = end_time - start_time
training_time_minutes = training_time_seconds / 60.0


# -------------------------
# Save model
# -------------------------
model.save(
    config.MODEL_PATH
)


# -------------------------
# Finish
# -------------------------
print()

print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Training time: {training_time_seconds:.2f} seconds "
    f"({training_time_minutes:.2f} minutes)"
)

print(
    f"Model saved to {config.MODEL_PATH}.zip"
)

print(
    f"Training log saved to {config.TRAIN_LOG_PATH}.monitor.csv"
)

print("=" * 60)


env.close()