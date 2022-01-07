###############################################################################
############################## GENERAL CONSTANTS ##############################
###############################################################################

# ----------- NAMES ----------- #
WALKER = "BipedalWalker"    # Requires: pip install box2d
BREAKOUT = "Breakout-v0"    # Requires: pip install gym[atari,accept-rom-license]==0.21.0 (https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym)


# ----------- PATHS ----------- #

PATH_RESULTS = "./results"

PATH_RESULTS_DQN = f"{PATH_RESULTS}/dqn"
PATH_RESULTS_PPO = f"{PATH_RESULTS}/ppo/"
PATH_RESULTS_GENETIC = f"{PATH_RESULTS}/genetic"
<<<<<<< Updated upstream

PATH_RESULTS_GENETIC_WALKER = f'{PATH_RESULTS_GENETIC}/{WALKER}'
PATH_RESULTS_GENETIC_BREAKOUT = f'{PATH_RESULTS_GENETIC}/{BREAKOUT}'
=======
PATH_RESULTS_GENETIC_WALKER = f'{PATH_RESULTS_GENETIC}/{WALKER}/'
PATH_RESULTS_GENETIC_BREAKOUT = f'{PATH_RESULTS_GENETIC}/{BREAKOUT}/'
>>>>>>> Stashed changes

PATH_RESULTS_PPO_WALKER = f'{PATH_RESULTS_PPO}{WALKER}'
PATH_RESULTS_PPO_BREAKOUT = f'{PATH_RESULTS_PPO}{BREAKOUT}'

PATH_ALGORITHMS = "./algorithms"




