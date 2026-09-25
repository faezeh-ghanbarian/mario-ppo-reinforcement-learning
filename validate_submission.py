import os
import numpy as np
import torch

from env import MarioEnv
from stable_baselines3 import PPO
from config import MODEL_PATH, SEED


def check_environment():
    print("1. Checking environment...")

    env = MarioEnv()

    obs1, _ = env.reset(seed=SEED)
    obs2, _ = env.reset(seed=SEED)

    if np.array_equal(obs1, obs2):
        print("✓ Deterministic reset passed")
    else:
        print("✗ Deterministic reset failed")

    env.close()


def check_model():
    print("\n2. Checking model loading...")

    model_path = MODEL_PATH

    if not os.path.exists(model_path + ".zip"):
        print("✗ Model file not found:", model_path)
        return None

    model = PPO.load(model_path)

    print("✓ Model loaded successfully")
    return model


def check_evaluation(model):
    print("\n3. Checking evaluation...")

    env = MarioEnv()
    obs, _ = env.reset(seed=SEED)

    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    env.close()

    print("✓ Evaluation completed")
    print("Total reward:", total_reward)


if __name__ == "__main__":

    print("=== Submission Validation ===\n")

    check_environment()

    model = check_model()

    if model:
        check_evaluation(model)

    print("\n=== Validation Finished ===")