from constants.constants_general import *
from constants.constants_genetic import *
from algorithms.genetic.bipedalwalker import walker_main
from algorithms.genetic.cartpole import cartpole_main
#from algorithms.genetic.breakout import breakout_main

def run_genetic(game):
    if game == WALKER:
        walker_main()
    
    elif game == BREAKOUT:
        #breakout_main()
        print("Not implemented")

    elif game == CARTPOLE:
        cartpole_main()


    





