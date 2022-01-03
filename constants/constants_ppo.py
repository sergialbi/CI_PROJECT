###########################################################################
############################## PPO CONSTANTS ##############################
###########################################################################

NUM_ENVS = 2
FRAMES_STACKED = 4
FRAMES_RESIZE = (100, 100)

LEARNING_RATE = 1e-4
BUFFER_SIZE = 300
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
EPOCHS = 5
GRADIENT_CLIPPING = 1000000

ITERATIONS = 30
BATCH_SIZE = 50
ITERATION_STEPS = 100
