###########################################################################
############################## DQN CONSTANTS ##############################
###########################################################################
import torch

# Reference hyperparameters: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

SELECTED_MODEL = "Big"

NUM_EPISODES = 3000
FRAME_SIZE = (84, 84)
FRAME_DIFF = False

BATCH_SIZE = 32
MEMORY_SIZE = BATCH_SIZE*1000
NUM_STACKED_FRAMES = 4

EPOCHS_PER_TARGET_UPD = 10 #5 #C=10000 steps
GAMMA = 0.99

STEPS_PER_POLICY_UPD = 4

LR = 0.00025
MOMENTUM = 0.01 #C=0.95
SQR_MOMENTUM = 0.95
MIN_SQR_GRAD = 0.01

EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000    #200 #C=1000000

STEPS_BEFORE_START_LEARNING = 50000


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")