from stable_baselines3 import PPO

import config
from env import MarioEnv


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
# Reset environment
# -------------------------
observation, info = env.reset(
    seed=config.SEED
)

total_reward = 0.0

passed_pit = False
reached_goal = False
death_reason = None

max_x = 0.0


# -------------------------
# Run one evaluation episode
# -------------------------
print()
print("Starting evaluation...")
print()


for step in range(config.EVAL_MAX_STEPS):

    # PPO chooses an action
    action, _ = model.predict(
        observation,
        deterministic=True
    )


    # Execute action
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


    print(
        f"Step: {step:3d} | "
        f"Action: {int(action)} | "
        f"Reward: {reward:7.2f} | "
        f"X: {x_position:6.1f} | "
        f"Pit: {passed_pit}"
    )


    if terminated or truncated:

        print()
        print("=" * 50)
        print("EPISODE FINISHED")
        print("=" * 50)

        print(f"Total reward : {total_reward:.2f}")
        print(f"Final X      : {x_position:.1f}")
        print(f"Maximum X    : {max_x:.1f}")
        print(f"Passed pit   : {passed_pit}")
        print(f"Reached goal : {reached_goal}")
        print(f"Death reason : {death_reason}")

        print()

        if reached_goal:
            print("Result        : SUCCESS - GOAL REACHED!")

        elif passed_pit:
            print("Result        : Passed pit, but failed later.")

        else:
            print("Result        : Failed before crossing pit.")

        print("=" * 50)

        break


else:

    print()
    print("=" * 50)
    print("MAXIMUM EVALUATION STEPS REACHED")
    print("=" * 50)

    print(f"Total reward : {total_reward:.2f}")
    print(f"Maximum X    : {max_x:.1f}")
    print(f"Passed pit   : {passed_pit}")
    print(f"Reached goal : {reached_goal}")
    print(f"Death reason : {death_reason}")

    print("=" * 50)


# -------------------------
# Close environment
# -------------------------
env.close()