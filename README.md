# PPO Mario Platformer

A custom Super Mario Bros-style platformer built for Deep Reinforcement Learning.

This project implements a custom game environment and trains a Proximal Policy Optimization (PPO) agent using Stable-Baselines3.

The goal is to train an agent that can navigate a platformer level, avoid obstacles, cross gaps, and reach the final goal.

---

# Project Structure

Main files:

- `game.py`: Game engine and physics
- `env.py`: Reinforcement learning environment
- `config.py`: Environment and training configuration
- `train.py`: PPO training script
- `evaluate.py`: Single-episode evaluation
- `evaluate_multiple.py`: 50-episode evaluation
- `random_test.py`: Reproducible random-agent baseline
- `watch_agent.py`: Visual gameplay demo
- `plot_learning_curve.py`: Training-curve generation

Folders:

- `models/`: Trained PPO weights
- `logs/`: Training logs
- `results/`: Evaluation results and plots

Other files:

- `requirements.txt`: Python dependencies
- `WRITEUP.md`: Short technical report
- `WRITEUP.pdf`: PDF version of the technical report

---

# Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# Training the Agent

To train the PPO agent from scratch:

```bash
python train.py
```

The trained model will be saved as:

```text
models/ppo_mario_v3_20k.zip
```

Training logs will be stored in:

```text
logs/
```

---

# Evaluation

## Single-Episode Evaluation

To evaluate the trained PPO agent on one episode:

```bash
python evaluate.py
```

Example successful output:

```text
Passed pit   : True
Reached goal : True
Result       : SUCCESS - GOAL REACHED!
```

## Multiple-Episode Evaluation

To evaluate the trained agent over 50 episodes:

```bash
python evaluate_multiple.py
```

The trained PPO policy was evaluated on 50 episodes.

| Metric | Value |
|---|---:|
| Episodes | 50 |
| Successful Episodes | 50 |
| Success Rate | 100.00% |
| Pit Passes | 50 |
| Pit Pass Rate | 100.00% |
| Deaths | 0 |
| Average Reward | 376.80 |
| Average Maximum X Position | 723.00 |

The PPO agent completed all 50 evaluation episodes and consistently crossed the pit and reached the goal.

---

# Random Baseline

A random-action policy was evaluated over 50 episodes as a baseline for comparison.

The baseline uses a fixed base seed of `42`. Each evaluation episode uses its own deterministic seed derived from this base seed, and the action space is also explicitly seeded so that the random-action experiment is reproducible.

Run:

```bash
python random_test.py
```

Final reproducible random-baseline results:

| Metric | Value |
|---|---:|
| Episodes | 50 |
| Base Seed | 42 |
| Successful Episodes | 0 |
| Success Rate | 0.00% |
| Pit Passes | 7 |
| Pit Pass Rate | 14.00% |
| Deaths | 50 |
| Average Reward | -37.16 |
| Reward Std | 35.03 |
| Average Maximum X | 417.24 |
| Maximum X Std | 49.84 |

The trained PPO policy substantially outperformed the random-action baseline. PPO achieved a 100% level-completion rate, while the random policy completed none of the 50 episodes.

---

# Gameplay Demo

To visualize the trained PPO agent playing the level:

```bash
python watch_agent.py
```

The visual demo shows the trained agent navigating the level, crossing the pit, avoiding hazards, and reaching the final goal.

The demo terminates with:

```text
SUCCESS - PPO AGENT REACHED THE GOAL!
```

---

# Reinforcement Learning Formulation

The game is formulated as a Markov Decision Process (MDP).

The environment follows a Gymnasium-compatible interaction interface:

```text
reset() -> observation

step(action) -> observation, reward, terminated, truncated, info
```

## Observation Space

The agent uses an 8-dimensional normalized feature-based observation vector.

The observation contains:

1. Player horizontal position (normalized)
2. Player vertical position (normalized)
3. Horizontal velocity (normalized)
4. Vertical velocity (normalized)
5. Whether the player is on the ground
6. Distance to the pit (normalized)
7. Relative distance to the enemy (normalized)
8. Distance to the goal (normalized)

A feature-based representation was selected instead of raw pixels to improve sample efficiency and reduce computational requirements. This allows the Core PPO agent to learn within a relatively small training budget.

## Action Space

The agent uses a discrete action space with five possible actions:

| Action | Meaning |
|---:|---|
| 0 | Nothing |
| 1 | Move left |
| 2 | Move right |
| 3 | Jump |
| 4 | Jump-right |

## Reward Function

The reward function uses reward shaping to provide intermediate learning signals during training.

Positive rewards include:

- Forward movement reward based on horizontal progress
- New-territory exploration reward when reaching a new maximum position
- Crossing the pit reward (+75)
- Reaching the goal flag reward (+150)

Negative rewards include:

- Small time penalty at each step
- Penalty for making no progress
- Falling into the pit (-100)
- Enemy collision (-100)
- Spike collision (-100)

This reward design provides gradual feedback during exploration and helps PPO learn useful navigation behaviors before it is able to reach the final goal.

---

# PPO Training Method

The agent was trained using Proximal Policy Optimization (PPO) implemented with Stable-Baselines3.

PPO was selected as a practical and stable policy-optimization method for the Core challenge while allowing the project to focus on environment design, RL formulation, evaluation, and reproducibility.

## Training Configuration

| Parameter | Value |
|---|---:|
| Algorithm | PPO |
| Library | Stable-Baselines3 |
| Policy | MlpPolicy |
| Total Timesteps | 20,000 |
| Learning Rate | 3e-4 |
| Rollout Steps (`n_steps`) | 1024 |
| Batch Size | 64 |
| Discount Factor (`gamma`) | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.01 |
| Random Seed | 42 |

## Training Procedure

The environment was wrapped using the Gymnasium-compatible interface and the Stable-Baselines3 `Monitor` wrapper.

During training:

1. The environment is reset at the beginning of each episode.
2. The PPO policy interacts with the environment using discrete actions.
3. The environment returns the next observation, reward, and termination information.
4. PPO collects trajectories and updates its policy.
5. Training statistics are written to the monitor log.
6. The trained policy is saved for later evaluation.

## Network Architecture

The agent uses Stable-Baselines3's `MlpPolicy` because the observation is a compact feature vector rather than a raw image.

Feature-based observations were chosen to reduce computational requirements and improve sample efficiency. A pixel-based CNN policy would be a useful extension for studying the trade-off between engineered state representations and more general visual observations.

---

# Results

The trained PPO agent was evaluated over 50 episodes and compared with a reproducible random-action baseline.

## PPO Evaluation Results

| Metric | Value |
|---|---:|
| Evaluation Episodes | 50 |
| Successful Episodes | 50 |
| Success Rate | 100.00% |
| Pit Pass Rate | 100.00% |
| Deaths | 0 |
| Average Reward | 376.80 |
| Average Maximum X Position | 723.00 |

The PPO agent consistently completed the current handcrafted level during evaluation.

## PPO vs. Random Baseline

| Metric | PPO Agent | Random Agent |
|---|---:|---:|
| Episodes | 50 | 50 |
| Success Rate | 100.00% | 0.00% |
| Pit Pass Rate | 100.00% | 14.00% |
| Deaths | 0 | 50 |
| Average Reward | 376.80 | -37.16 |
| Average Maximum X | 723.00 | 417.24 |

The large performance gap indicates that PPO learned meaningful navigation behavior rather than succeeding through random exploration.

However, the PPO evaluation is performed on the same handcrafted level used during development and training. Therefore, the result demonstrates mastery of the current level but does not establish generalization to unseen level layouts.

---

# Learning Curve

Training progress was monitored using episode rewards recorded by the Stable-Baselines3 `Monitor` wrapper.

The plotted learning curve contains raw episode rewards together with a 10-episode moving average to make the overall training trend easier to inspect.

![PPO Learning Curve](results/learning_curve_v3.png)

The learning curve can be regenerated with:

```bash
python plot_learning_curve.py
```

---

# Hardware and Compute Budget

The project was developed and trained on a consumer laptop environment.

The implementation was designed to remain computationally lightweight by using:

- A custom lightweight platformer environment
- Feature-based observations instead of raw pixels
- An MLP policy
- A target training budget of 20,000 timesteps
- Approximate training time: 41 seconds for the final PPO training run

Stable-Baselines3 completed 20,480 timesteps because PPO collects complete rollout batches before each update.

The final training run reported approximately 40.90 seconds of wall-clock training time.

The focus was on producing a small, reproducible RL system and demonstrating measurable improvement over a random baseline rather than relying on large-scale compute.
---

# Limitations and Future Work

Although the trained PPO agent performs consistently on the current level, several important limitations remain.

## Current Limitations

- Training and evaluation currently use a single handcrafted level.
- The observation space contains engineered features rather than raw visual input.
- Enemy and environment behavior are largely deterministic.
- The training budget is relatively small.
- The current results demonstrate performance on the training environment but do not establish generalization to unseen levels.

## Future Work

With additional time, the most useful next experiments would include:

- Training on multiple or procedurally generated levels and evaluating on held-out levels
- Measuring the generalization gap between training and unseen levels
- Performing reward-shaping ablations to quantify the contribution of individual reward components
- Comparing feature-based observations with raw-pixel observations using CNN policies
- Introducing stochastic enemy behavior or action/observation noise to evaluate robustness
- Exploring curriculum learning to improve exploration and sample efficiency
- Implementing PPO from scratch to study components such as GAE, clipping, entropy regularization, and advantage normalization

Generalization would be the highest-priority extension because successful performance on a single deterministic level does not show whether the learned behavior transfers to new platform layouts.

---

# Reproducibility

The project provides separate commands for training, evaluation, visualization, and result generation.

Install dependencies:

```bash
pip install -r requirements.txt
```

Train from scratch:

```bash
python train.py
```

Evaluate one episode:

```bash
python evaluate.py
```

Evaluate multiple PPO episodes:

```bash
python evaluate_multiple.py
```

Run the reproducible random baseline:

```bash
python random_test.py
```

Generate the learning curve:

```bash
python plot_learning_curve.py
```

Run the visual demo:

```bash
python watch_agent.py
```

The PPO training script uses a fixed random seed of `42`.

The random baseline also uses a fixed base seed of `42`, and each episode explicitly seeds both the environment and the discrete action space. Repeated executions therefore produce the same reported baseline statistics.

The trained weights, training logs, evaluation output, and learning-curve artifact are included with the project.

---

# Disclosure

## Libraries and Tools

This project uses the following libraries:

- Stable-Baselines3 for the PPO implementation
- Gymnasium for the reinforcement learning environment interface
- Pygame for game development and visualization
- NumPy for numerical operations
- Pandas for training-log analysis
- Matplotlib for visualization
- PyTorch through the Stable-Baselines3 learning stack

## Validation

## Validation

To verify the submission:

```bash
python validate_submission.py

The validation script checks:

1. deterministic environment reset with the configured seed
2. saved PPO checkpoint loading
3. evaluation execution
4. successful reproduction of the expected evaluation behavior

A successful validation run confirms that the saved model artifact can be loaded and evaluated without retraining.

## AI Assistant Usage

AI assistants were used as a support tool during development for:

- Discussing implementation approaches and design alternatives
- Debugging assistance and identifying potential issues
- Reviewing code organization and documentation clarity
- Supporting analysis and presentation of experimental results

All implementation decisions, experimental design choices, code modifications, and final validation were performed and reviewed by the author. The author is responsible for understanding, explaining, and maintaining the submitted work.
