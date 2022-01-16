###########################################################################
############################## DQN CONSTANTS ##############################
###########################################################################
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SELECTED_MODEL = "Base"
NUM_EPISODES = 3000

# Reference hyperparameters: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf


# For Breakout
FRAME_SIZE = (84, 84)
FRAME_DIFF = False

BATCH_SIZE = 32
MEMORY_SIZE = BATCH_SIZE*1000
NUM_STACKED_FRAMES = 4

STEPS_PER_POLICY_UPD = 4

STEPS_PER_TARGET_UPD = 10000
GAMMA = 0.99

LR = 0.00025
MOMENTUM = 0.01
SQR_MOMENTUM = 0.95
MIN_SQR_GRAD = 0.01

EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000

STEPS_BEFORE_START_LEARNING = 10000


"""
# For CartPole
FRAME_SIZE = (84, 84)
FRAME_DIFF = False

BATCH_SIZE = 32
MEMORY_SIZE = BATCH_SIZE*1000
NUM_STACKED_FRAMES = 4

STEPS_PER_POLICY_UPD = 4

STEPS_PER_TARGET_UPD = 100
GAMMA = 0.9

LR = 1e-3
MOMENTUM = 0.01#0.95
SQR_MOMENTUM = 0.95
MIN_SQR_GRAD = 0.01

EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 500

STEPS_BEFORE_START_LEARNING = 10000
"""