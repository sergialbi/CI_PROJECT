from constants.constants_general import *
from algorithms.genetic.genetic_main import run_genetic
from algorithms.dqn.dqn import run_dqn


if __name__ == '__main__':
    # Select the algorithm to run
    val = -1
    while val not in [0,1,2,3]:
        val = int(input("0: create folders \
                        \n1: run Deep Q Learning algorithm\
                        \n2: run PPO algorithm\
                        \n3: run Genetic algorithm \n"))
        if val not in [0,1,2,3]:
            print("Not a valid option\n")
    
    
    if val == 0:
        pass
    else:
        # Select the game
        g = "none"
        while g != "w" and g != "b":
            g = input("Select game: w (Bipedal Walker) - b (Breakout)\n")
            if g != "w" and g!= "b":
                print("Not a valid option\n")
        
        if g == "w":
            game_name = WALKER
        elif g == "b":
            game_name = BREAKOUT
        
        if val == 1:
            run_dqn(game_name)
        elif val == 2:
            pass
        elif val == 3:
            run_genetic(game_name)
        