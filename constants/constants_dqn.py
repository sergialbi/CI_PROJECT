###########################################################################
############################## DQN CONSTANTS ##############################
###########################################################################
import torch

NUM_EPOCHS = 100
BATCH_SIZE = 128
MEMORY_SIZE = BATCH_SIZE*10
EPOCHS_PER_TARGET_UPD=1
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")