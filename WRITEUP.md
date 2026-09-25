# PPO Mario Platformer

## Graduate Research Assistant Code Challenge

**Author:** Faezeh Ghanbarian

---

## 1. Introduction and Problem Formulation

This project implements a lightweight Super Mario Bros-style platformer and formulates it as a reinforcement learning problem. The environment was developed specifically for this challenge rather than using an existing Mario reinforcement learning environment.

The main objective was to build a complete and reproducible reinforcement learning pipeline under a limited computational budget. The project therefore focuses on three components: a custom platformer environment, a clear Markov Decision Process (MDP) formulation, and a PPO agent that can learn meaningful behavior and outperform a random-action baseline.

The level contains solid ground, a pit, an enemy, a spike hazard, and a final goal flag. The agent must learn to move through the level, cross the pit, avoid terminal hazards, and reach the goal.

Rather than optimizing for a state-of-the-art game-playing agent, I prioritized a compact environment that could be trained and evaluated quickly and whose behavior could be inspected and explained.

---

## 2. Environment and MDP Design

The custom game is exposed to the agent through a Gymnasium-compatible interface:

```text
reset() -> observation

step(action) -> observation, reward, terminated, truncated, info
```

An episode terminates when the player reaches the goal or dies because of a terminal hazard. Episodes can also be truncated when the maximum allowed number of environment steps is reached.

### 2.1 Observation Space

The agent receives an 8-dimensional normalized feature vector:

1. Player horizontal position
2. Player vertical position
3. Horizontal velocity
4. Vertical velocity
5. Whether the player is on the ground
6. Distance to the pit
7. Relative distance to the enemy
8. Distance to the goal

I selected a feature-based observation representation rather than raw pixels.

This was an engineering trade-off motivated by the computational constraints of the challenge. Pixel-based learning would require a CNN, substantially more interaction data, and a larger training budget. The compact feature vector allows PPO to focus on learning navigation behavior while keeping training feasible on consumer hardware.

The limitation of this choice is that the policy receives engineered information about the environment. Therefore, successful performance does not demonstrate visual perception or general visual reasoning.

### 2.2 Action Space

The environment uses a discrete action space with five actions:

| Action | Meaning |
|---:|---|
| 0 | Nothing |
| 1 | Move left |
| 2 | Move right |
| 3 | Jump |
| 4 | Jump-right |

This action set is intentionally compact. It is expressive enough to complete the level while keeping the exploration problem manageable.

### 2.3 Reward Function

A purely sparse reward based only on reaching the final flag would provide very little learning signal during early exploration. I therefore used reward shaping to provide intermediate feedback.

Positive signals include:

- Reward for horizontal forward progress
- Reward for entering previously unreached territory
- +75 for successfully crossing the pit
- +150 for reaching the final goal

Negative signals include:

- A small time penalty
- A penalty for making no progress
- -100 for falling into the pit
- -100 for colliding with the enemy
- -100 for colliding with the spike hazard

The purpose of this design is to create a progression of useful learning signals. The agent can first discover forward movement, then learn to cross the pit, and finally learn to complete the level.

However, reward shaping also introduces human assumptions about desirable behavior. An important follow-up experiment would therefore be to compare this design with sparse and less-shaped alternatives.

---

## 3. PPO Method and Training

The policy was trained using Proximal Policy Optimization (PPO) from Stable-Baselines3.

PPO was chosen for the Core implementation because it provides a well-established policy-optimization baseline while allowing the project to focus on environment engineering, MDP design, evaluation, and reproducibility.

Because the observation consists of eight numerical features, I used the Stable-Baselines3 `MlpPolicy`.

### 3.1 Hyperparameters

The main training configuration was:

| Parameter | Value |
|---|---:|
| Algorithm | PPO |
| Policy | MlpPolicy |
| Training Timesteps | 20,000 |
| Learning Rate | 3e-4 |
| Rollout Steps (`n_steps`) | 1024 |
| Batch Size | 64 |
| Discount Factor (`gamma`) | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.01 |
| Random Seed | 42 |

The environment was wrapped with the Stable-Baselines3 `Monitor` wrapper so that episode rewards and lengths could be recorded during training.

The training budget was intentionally kept small. The combination of a lightweight custom environment and feature-based observations allowed meaningful behavior to emerge within approximately 20,000 timesteps.

---

## 4. Experimental Results

The trained PPO policy and a random-action baseline were each evaluated over 50 episodes.

### 4.1 PPO Evaluation

| Metric | Result |
|---|---:|
| Evaluation Episodes | 50 |
| Successful Episodes | 50 |
| Success Rate | 100.00% |
| Pit Passes | 50 |
| Pit Pass Rate | 100.00% |
| Deaths | 0 |
| Average Reward | 376.80 |
| Average Maximum X Position | 723.00 |

During this evaluation, the deterministic PPO policy successfully completed all 50 episodes.

The agent consistently learned the sequence of behaviors required by the current level: forward navigation, crossing the pit, avoiding terminal hazards, and reaching the goal.

### 4.2 Random-Action Baseline

To determine whether this performance represented meaningful learned behavior, I also evaluated a random-action policy.

For reproducibility, the random baseline uses a fixed base seed of 42. Each episode uses a deterministic seed derived from the base seed, and the environment action space is explicitly seeded before random actions are sampled.

The final reproducible baseline results are:

| Metric | Random Agent |
|---|---:|
| Evaluation Episodes | 50 |
| Base Seed | 42 |
| Successful Episodes | 0 |
| Success Rate | 0.00% |
| Pit Passes | 7 |
| Pit Pass Rate | 14.00% |
| Deaths | 50 |
| Average Reward | -37.16 |
| Reward Std | 35.03 |
| Average Maximum X Position | 417.24 |
| Maximum X Std | 49.84 |

The trained PPO agent achieved a 100.00% success rate and an average reward of 376.80, whereas the random-action policy completed none of the 50 episodes and obtained an average reward of -37.16.

The random policy crossed the pit in only 14.00% of episodes, compared with 100.00% for PPO.

This substantial performance difference provides evidence that the PPO agent learned useful navigation behavior rather than completing the task through random action selection.

At the same time, the 100% completion rate should be interpreted carefully. The evaluation is performed on the same handcrafted environment used during development and training. It therefore demonstrates mastery of this level, not generalization to unseen platform layouts.

### 4.3 PPO vs. Random Baseline

| Metric | PPO Agent | Random Agent |
|---|---:|---:|
| Evaluation Episodes | 50 | 50 |
| Success Rate | 100.00% | 0.00% |
| Pit Pass Rate | 100.00% | 14.00% |
| Deaths | 0 | 50 |
| Average Reward | 376.80 | -37.16 |
| Average Maximum X Position | 723.00 | 417.24 |

### 4.4 Learning Curve

Episode rewards were recorded during PPO training. To visualize the learning process, I plotted both raw episode rewards and a 10-episode moving average.

![PPO Learning Curve](results/learning_curve_v3.png)

The learning curve provides a view of how episode-level performance changed during training rather than relying only on the final evaluation result.

The figure can be regenerated using:

```bash
python plot_learning_curve.py
```

---

## 5. Engineering and Reproducibility

The implementation was designed to remain lightweight and easy to reproduce.

The project contains separate scripts for:

- Training the PPO agent
- Evaluating a single episode
- Evaluating multiple PPO episodes
- Running the random baseline
- Generating the learning curve
- Watching the trained agent play

Dependencies are provided in `requirements.txt`.

The PPO training configuration uses a fixed random seed of 42.

The random-action baseline also uses a fixed base seed of 42. Each evaluation episode explicitly seeds both the environment and the action space, making repeated baseline evaluations reproducible.

The trained PPO weights, monitor logs, evaluation results, and learning-curve artifact are included with the submission.

The main training command is:

```bash
python train.py
```

The single-episode evaluation can be run with:

```bash
python evaluate.py
```

The 50-episode PPO evaluation can be run with:

```bash
python evaluate_multiple.py
```

The reproducible random baseline can be run with:

```bash
python random_test.py
```

The visual demonstration can be run using:

```bash
python watch_agent.py
```

This separation between training, quantitative evaluation, baseline comparison, and visualization makes it easier to reproduce and inspect the reported behavior.

---

## 6. Limitations and Next Steps

The strongest limitation of the current system is that the policy is trained and evaluated on one handcrafted level.

A 100% completion rate on this environment therefore does not establish that the policy has learned a general platform-playing strategy. It may have learned behavior that is highly specialized to the geometry and dynamics of this level.

The feature-based observation representation is another important limitation. It improves sample efficiency but provides the policy with engineered state information. A raw-pixel agent would present a more difficult but more general perception problem.

Given another two weeks, my first research extension would be **generalization**.

I would generate multiple training levels by varying pit positions, hazard positions, enemy locations, and goal distances. I would then evaluate the trained policy on held-out level configurations that were never observed during training.

The primary quantity of interest would be the generalization gap between performance on training-level distributions and unseen levels.

Additional experiments would include:

- **Reward-shaping ablation:** compare sparse, progress-shaped, and fully shaped rewards.
- **Observation study:** compare the current feature vector with pixel-based CNN observations.
- **Robustness:** introduce stochastic enemy behavior, action noise, or observation noise.
- **Curriculum learning:** gradually increase level difficulty during training.
- **PPO from scratch:** implement GAE, clipped policy loss, value loss, entropy regularization, and advantage normalization directly and compare the implementation with the Stable-Baselines3 baseline.

These experiments would help determine whether the current result reflects task-specific optimization or behavior that transfers across related environments.

---

## 7. Disclosure

The implementation uses:

- Stable-Baselines3 for PPO
- Gymnasium for the reinforcement learning environment interface
- Pygame for the custom game and visualization
- NumPy for numerical operations
- Pandas for log analysis
- Matplotlib for visualization
- PyTorch through the reinforcement learning stack

AI assistants were used during development for implementation discussions, debugging support, documentation organization, and reviewing the presentation of the experimental results.

The final implementation and experimental decisions were reviewed by the author, who is responsible for understanding and explaining the submitted code and results.

---

## Conclusion

This project demonstrates a complete reinforcement learning pipeline built around a custom platformer environment.

Within a limited training budget, PPO learned a policy that consistently completed the current level and substantially outperformed a reproducible random-action baseline.

Across 50 evaluation episodes, PPO achieved a 100.00% success rate with an average reward of 376.80. In comparison, the seeded random baseline achieved a 0.00% success rate and an average reward of -37.16.

The result establishes that the current MDP and reward formulation are sufficient for PPO to learn the task. The more interesting next question is whether the learned behavior can generalize beyond the single environment on which it was trained. Generalization across unseen level configurations would therefore be the primary direction for extending this Core implementation into a deeper research study.
The final PPO training run took approximately 41 seconds of wall-clock time (40.90 seconds measured), with Stable-Baselines3 completing 20,480 timesteps because PPO collects complete rollout batches before each update.