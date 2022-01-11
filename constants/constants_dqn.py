###########################################################################
############################## DQN CONSTANTS ##############################
###########################################################################
import torch

SELECTED_MODEL = "Base"

FRAME_SIZE = (84, 84)
FRAME_DIFF = False

NUM_EPOCHS = 200
BATCH_SIZE = 32
MEMORY_SIZE = BATCH_SIZE*1000
NUM_STACKED_FRAMES = 4
GAMMA = 0.99
LR = 0.00025
MOMENTUM = 0.95
EPOCHS_PER_TARGET_UPD = 1
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")