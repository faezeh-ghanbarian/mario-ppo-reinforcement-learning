import gymnasium as gym
from gymnasium import spaces
import numpy as np

import config


class MarioEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # =====================================================
        # LEVEL
        # =====================================================
        self.WIDTH = config.WIDTH
        self.HEIGHT = config.HEIGHT
        self.GROUND_Y = config.GROUND_Y

        self.PIT_START = config.PIT_START
        self.PIT_END = config.PIT_END

        # =====================================================
        # PLAYER
        # =====================================================
        self.START_X = config.START_X
        self.START_Y = config.START_Y

        self.PLAYER_WIDTH = config.PLAYER_WIDTH
        self.PLAYER_HEIGHT = config.PLAYER_HEIGHT

        self.MOVE_SPEED = config.MOVE_SPEED
        self.JUMP_FORWARD_SPEED = config.JUMP_FORWARD_SPEED
        self.JUMP_SPEED = config.JUMP_SPEED
        self.GRAVITY = config.GRAVITY

        self.player_x = float(self.START_X)
        self.player_y = float(self.START_Y)

        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.on_ground = False

        # =====================================================
        # ENEMY
        # =====================================================
        self.enemy_x = float(config.ENEMY_START_X)

        self.enemy_y = (
            self.GROUND_Y
            - config.ENEMY_HEIGHT
        )

        self.enemy_speed = float(config.ENEMY_SPEED)

        self.enemy_min_x = config.ENEMY_MIN_X
        self.enemy_max_x = config.ENEMY_MAX_X

        # =====================================================
        # SPIKE
        # =====================================================
        self.SPIKE_X = config.SPIKE_X
        self.SPIKE_WIDTH = config.SPIKE_WIDTH
        self.SPIKE_HEIGHT = config.SPIKE_HEIGHT

        # =====================================================
        # GOAL
        # =====================================================
        self.FLAG_X = config.FLAG_X

        # =====================================================
        # ACTION SPACE
        # =====================================================
        # 0 = nothing
        # 1 = left
        # 2 = right
        # 3 = jump
        # 4 = jump-right

        self.action_space = spaces.Discrete(5)

        # =====================================================
        # NORMALIZED OBSERVATION SPACE
        # =====================================================
        #
        # 0 player_x / WIDTH
        # 1 player_y / HEIGHT
        # 2 velocity_x / 20
        # 3 velocity_y / 30
        # 4 on_ground
        # 5 distance_to_pit / WIDTH
        # 6 enemy_relative_x / WIDTH
        # 7 distance_to_goal / WIDTH
        #

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,
                    -2.0,
                    -1.0,
                    -1.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0
                ],
                dtype=np.float32
            ),
            high=np.array(
                [
                    1.0,
                    3.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0
                ],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # =====================================================
        # EPISODE TRACKING
        # =====================================================
        self.max_steps = config.MAX_STEPS
        self.current_step = 0

        self.passed_pit = False
        self.max_x_reached = float(self.START_X)

        # Detect agents that stop making progress
        self.steps_without_progress = 0
        self.MAX_STUCK_STEPS = 100

    # =========================================================
    # OBSERVATION
    # =========================================================
    def _get_observation(self):

        distance_to_pit = (
            self.PIT_START - self.player_x
        )

        enemy_relative_x = (
            self.enemy_x - self.player_x
        )

        distance_to_goal = (
            self.FLAG_X - self.player_x
        )

        observation = np.array(
            [
                self.player_x / self.WIDTH,
                self.player_y / self.HEIGHT,
                self.velocity_x / 20.0,
                self.velocity_y / 30.0,
                float(self.on_ground),
                distance_to_pit / self.WIDTH,
                enemy_relative_x / self.WIDTH,
                distance_to_goal / self.WIDTH
            ],
            dtype=np.float32
        )

        return observation

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.player_x = float(self.START_X)
        self.player_y = float(self.START_Y)

        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.on_ground = False

        self.enemy_x = float(config.ENEMY_START_X)
        self.enemy_speed = float(config.ENEMY_SPEED)

        self.current_step = 0

        self.passed_pit = False

        self.max_x_reached = float(
            self.START_X
        )

        self.steps_without_progress = 0

        observation = self._get_observation()

        return observation, {}

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):

        self.current_step += 1

        old_x = self.player_x

        # =====================================================
        # ACTIONS
        # =====================================================

        self.velocity_x = 0.0

        # 0 = nothing
        if action == 0:
            pass

        # 1 = left
        elif action == 1:

            self.velocity_x = (
                -self.MOVE_SPEED
            )

        # 2 = right
        elif action == 2:

            self.velocity_x = (
                self.MOVE_SPEED
            )

        # 3 = vertical jump
        elif action == 3:

            if self.on_ground:
                self.velocity_y = (
                    self.JUMP_SPEED
                )

        # 4 = jump-right
        elif action == 4:

            # Horizontal movement continues
            # while airborne.
            self.velocity_x = (
                self.JUMP_FORWARD_SPEED
            )

            # A new jump can only start
            # from the ground.
            if self.on_ground:

                self.velocity_y = (
                    self.JUMP_SPEED
                )

        # =====================================================
        # PHYSICS
        # =====================================================

        previous_bottom = (
            self.player_y
            + self.PLAYER_HEIGHT
        )

        self.velocity_y += (
            self.GRAVITY
        )

        self.player_x += (
            self.velocity_x
        )

        self.player_y += (
            self.velocity_y
        )

        # Horizontal boundaries
        self.player_x = max(
            0.0,
            min(
                self.player_x,
                self.WIDTH
                - self.PLAYER_WIDTH
            )
        )

        # =====================================================
        # GROUND COLLISION
        # =====================================================

        player_left = self.player_x

        player_right = (
            self.player_x
            + self.PLAYER_WIDTH
        )

        overlaps_left_ground = (
            player_right > 0
            and
            player_left < self.PIT_START
        )

        overlaps_right_ground = (
            player_right > self.PIT_END
            and
            player_left < self.WIDTH
        )

        has_ground = (
            overlaps_left_ground
            or overlaps_right_ground
        )

        self.on_ground = False

        if (
            has_ground
            and self.velocity_y >= 0
            and previous_bottom <= self.GROUND_Y
            and self.player_y + self.PLAYER_HEIGHT >= self.GROUND_Y
        ):

            self.player_y = (
                self.GROUND_Y
                - self.PLAYER_HEIGHT
            )

            self.velocity_y = 0.0
            self.on_ground = True

        # =====================================================
        # ENEMY MOVEMENT
        # =====================================================

        self.enemy_x += (
            self.enemy_speed
        )

        if self.enemy_x <= self.enemy_min_x:

            self.enemy_x = float(
                self.enemy_min_x
            )

            self.enemy_speed = abs(
                self.enemy_speed
            )

        elif self.enemy_x >= self.enemy_max_x:

            self.enemy_x = float(
                self.enemy_max_x
            )

            self.enemy_speed = -abs(
                self.enemy_speed
            )

        # =====================================================
        # REWARD
        # =====================================================

        reward = 0.0

        terminated = False
        truncated = False

        death_reason = None
        reached_goal = False

        # -----------------------------------------------------
        # 1. Forward progress
        # -----------------------------------------------------

        progress = (
            self.player_x
            - old_x
        )

        # Moving right is useful.
        reward += progress * 0.15

        # -----------------------------------------------------
        # 2. Reward new territory
        # -----------------------------------------------------

        if self.player_x > self.max_x_reached:

            new_progress = (
                self.player_x
                - self.max_x_reached
            )

            reward += (
                new_progress * 0.10
            )

            self.max_x_reached = (
                self.player_x
            )

            self.steps_without_progress = 0

        else:

            self.steps_without_progress += 1

        # -----------------------------------------------------
        # 3. Time penalty
        # -----------------------------------------------------

        reward -= 0.02

        # -----------------------------------------------------
        # 4. Extra penalty for standing still
        # -----------------------------------------------------

        if abs(progress) < 0.01:

            reward -= 0.05

        # -----------------------------------------------------
        # 5. Penalize moving left
        # -----------------------------------------------------

        if progress < 0:

            reward += (
                progress * 0.10
            )

        # =====================================================
        # PIT DEATH
        # =====================================================

        if self.player_y > self.HEIGHT:

            reward -= 100.0

            terminated = True
            death_reason = "pit"

        # =====================================================
        # ENEMY COLLISION
        # =====================================================

        enemy_hit = (
            self.player_x
            < self.enemy_x + config.ENEMY_WIDTH
            and
            self.player_x + self.PLAYER_WIDTH
            > self.enemy_x
            and
            self.player_y
            < self.enemy_y + config.ENEMY_HEIGHT
            and
            self.player_y + self.PLAYER_HEIGHT
            > self.enemy_y
        )

        if enemy_hit and not terminated:

            reward -= 100.0

            terminated = True
            death_reason = "enemy"

        # =====================================================
        # SPIKE COLLISION
        # =====================================================

        spike_top = (
            self.GROUND_Y
            - self.SPIKE_HEIGHT
        )

        spike_hit = (
            self.player_x
            < self.SPIKE_X + self.SPIKE_WIDTH
            and
            self.player_x + self.PLAYER_WIDTH
            > self.SPIKE_X
            and
            self.player_y + self.PLAYER_HEIGHT
            > spike_top
        )

        if spike_hit and not terminated:

            reward -= 100.0

            terminated = True
            death_reason = "spike"

        # =====================================================
        # PIT CROSSING
        # =====================================================

        if (
            not self.passed_pit
            and self.player_x >= self.PIT_END
            and not terminated
        ):

            # Strong one-time reward:
            # crossing the pit is an important
            # intermediate milestone.
            reward += 75.0

            self.passed_pit = True

        # =====================================================
        # GOAL
        # =====================================================

        if (
            self.player_x + self.PLAYER_WIDTH
            >= self.FLAG_X
            and not terminated
        ):

            reward += 150.0

            reached_goal = True
            terminated = True

        # =====================================================
        # STUCK AGENT
        # =====================================================

        if (
            self.steps_without_progress
            >= self.MAX_STUCK_STEPS
            and not terminated
        ):

            # Discourage policies that simply
            # stop before the pit.
            reward -= 20.0

            truncated = True

            death_reason = "stuck"

        # =====================================================
        # TIME LIMIT
        # =====================================================

        if (
            self.current_step >= self.max_steps
            and not terminated
        ):

            truncated = True

        # =====================================================
        # INFO
        # =====================================================

        observation = self._get_observation()

        info = {
            "x_position": self.player_x,
            "max_x": self.max_x_reached,
            "passed_pit": self.passed_pit,
            "reached_goal": reached_goal,
            "death_reason": death_reason,
            "steps_without_progress": self.steps_without_progress
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )