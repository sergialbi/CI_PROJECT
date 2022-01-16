###########################################################################
############################## PPO CONSTANTS ##############################
###########################################################################

NUM_ENVS = 6
FRAMES_STACKED = 4
FRAMES_RESIZE = (84, 84)
NOOP_ACTIONS = 30
REWARD_SCALE = 1

LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
EPOCHS = 10
GRADIENT_CLIPPING = 0.5
MAX_KL_DIVERG = 0.03

BUFFER_SIZE = 128
BATCH_SIZE = 256

TRAIN_ITERATIONS = 10000
ITERATION_STEPS = BUFFER_SIZE
TEST_EPISODES = 100

# Best constants used with Cart Pole (the others remain the same). Replace them to train

"""
NUM_ENVS = 6
REWARD_SCALE = 1

LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
EPOCHS = 10
GRADIENT_CLIPPING = 0.5
MAX_KL_DIVERG = 0.03

BUFFER_SIZE = 128
BATCH_SIZE = 256

TRAIN_EPISODES = 10000
ITERATION_STEPS = BUFFER_SIZE
TEST_EPISODES = 100
"""

# Best constants used with Bipedal Walker (the others remain the same). Replace them to train

"""
NUM_ENVS = 6
REWARD_SCALE = 1

LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
EPOCHS = 10
GRADIENT_CLIPPING = 0.5
MAX_KL_DIVERG = 0.05

BUFFER_SIZE = 512
BATCH_SIZE = 512

TRAIN_EPISODES = 10000
ITERATION_STEPS = BUFFER_SIZE
TEST_EPISODES = 100
"""