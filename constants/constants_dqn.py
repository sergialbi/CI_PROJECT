###########################################################################
############################## DQN CONSTANTS ##############################
###########################################################################
import torch

NUM_EPOCHS = 100
FRAME_SIZE = (64, 64)
BATCH_SIZE = 32
MEMORY_SIZE = BATCH_SIZE*1000
GAMMA = 0.9
EPOCHS_PER_TARGET_UPD = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")