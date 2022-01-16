###########################################################################
############################## PPO CONSTANTS ##############################
###########################################################################

NUM_ENVS = 8
REWARD_SCALE = 1

LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
EPOCHS = 10
GRADIENT_CLIPPING = 0.5
MAX_KL_DIVERG = 0.03

BUFFER_SIZE = 64
BATCH_SIZE = 32

TRAIN_ITERATIONS = 10000
ITERATION_STEPS = BUFFER_SIZE
TEST_EPISODES = 100

# Best constants used with Cart Pole. Replace them to train

"""
NUM_ENVS = 8
REWARD_SCALE = 1

LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
EPOCHS = 10
GRADIENT_CLIPPING = 0.5
MAX_KL_DIVERG = 0.03

BUFFER_SIZE = 64
BATCH_SIZE = 32

TRAIN_ITERATIONS = 10000
ITERATION_STEPS = BUFFER_SIZE
TEST_EPISODES = 100
"""

# Best constants used with Bipedal Walker. Replace them to train

"""
NUM_ENVS = 8
REWARD_SCALE = 1

LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
EPOCHS = 10
GRADIENT_CLIPPING = 0.5
MAX_KL_DIVERG = 0.03

BUFFER_SIZE = 512
BATCH_SIZE = 512

TRAIN_ITERATIONS = 10000
ITERATION_STEPS = BUFFER_SIZE
TEST_EPISODES = 100
"""

# Old breakout constants
FRAMES_STACKED = 4
FRAMES_RESIZE = (84, 84)
NOOP_ACTIONS = 30