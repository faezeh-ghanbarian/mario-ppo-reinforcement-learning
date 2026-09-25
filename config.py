# =========================
# Window
# =========================

WIDTH = 800
HEIGHT = 450


# =========================
# Ground
# =========================

GROUND_Y = 380


# =========================
# Player
# =========================

START_X = 100
START_Y = 300

PLAYER_WIDTH = 40
PLAYER_HEIGHT = 50

MOVE_SPEED = 4

JUMP_FORWARD_SPEED = 7
JUMP_SPEED = -15
GRAVITY = 0.6


# =========================
# Pit
# =========================

PIT_START = 350
PIT_END = 470


# =========================
# Enemy
# =========================

ENEMY_WIDTH = 40
ENEMY_HEIGHT = 40

ENEMY_START_X = 560

ENEMY_MIN_X = 500
ENEMY_MAX_X = 650

ENEMY_SPEED = 2


# =========================
# Spike
# =========================

SPIKE_X = 710
SPIKE_WIDTH = 35
SPIKE_HEIGHT = 25


# =========================
# Goal
# =========================

FLAG_X = 760
FLAG_WIDTH = 20
FLAG_HEIGHT = 100


# =========================
# Environment
# =========================

MAX_STEPS = 1000


# =========================
# Reproducibility
# =========================

SEED = 42


# =========================
# PPO Hyperparameters
# =========================

TOTAL_TIMESTEPS = 20_000

LEARNING_RATE = 3e-4

N_STEPS = 1024
BATCH_SIZE = 64

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_RANGE = 0.2
ENT_COEF = 0.01


# =========================
# Paths
# =========================

MODEL_PATH = "models/ppo_mario_v3_20k"

TRAIN_LOG_PATH = "logs/training_v3_20k"


# =========================
# Evaluation
# =========================

NUM_EVAL_EPISODES = 50
EVAL_MAX_STEPS = 500