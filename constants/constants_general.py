###############################################################################
############################## GENERAL CONSTANTS ##############################
###############################################################################
import os

# ----------- NAMES ----------- #
WALKER = "BipedalWalker-v3"    # Requires: pip install box2d
BREAKOUT = "BreakoutDeterministic-v4"    # Requires: pip install gym[atari,accept-rom-license]==0.21.0 (https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym)
CARTPOLE = "CartPole-v1"

# ----------- PATHS ----------- #

PATH_RESULTS = os.path.join(".", "results")

PATH_RESULTS_PPO = os.path.join(PATH_RESULTS, "ppo")
PATH_RESULTS_PPO_WALKER = os.path.join(PATH_RESULTS_PPO, WALKER)
PATH_RESULTS_PPO_BREAKOUT = os.path.join(PATH_RESULTS_PPO, BREAKOUT)
PATH_RESULTS_PPO_CARTPOLE = os.path.join(PATH_RESULTS_PPO, CARTPOLE)

PATH_RESULTS_GENETIC = os.path.join(PATH_RESULTS, "genetic")
PATH_RESULTS_GENETIC_WALKER = os.path.join(PATH_RESULTS_GENETIC, WALKER)
PATH_RESULTS_GENETIC_BREAKOUT = os.path.join(PATH_RESULTS_GENETIC, BREAKOUT)
PATH_RESULTS_GENETIC_CARTPOLE = os.path.join(PATH_RESULTS_GENETIC, CARTPOLE)

PATH_RESULTS_DQN = os.path.join(PATH_RESULTS, "dqn")
PATH_RESULTS_DQN_WALKER = os.path.join(PATH_RESULTS_DQN, WALKER)
PATH_RESULTS_DQN_BREAKOUT = os.path.join(PATH_RESULTS_DQN, BREAKOUT)
PATH_RESULTS_DQN_CARTPOLE = os.path.join(PATH_RESULTS_DQN, CARTPOLE)


PATH_ALGORITHMS = "./algorithms"




