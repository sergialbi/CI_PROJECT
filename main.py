from constants.constants_general import *
from algorithms.genetic.genetic_main import run_genetic
from algorithms.dqn.dqn import run_dqn
from algorithms.PPO.PPO_run import train_PPO, test_PPO

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
        while g != "w" and g != "b" and g != "c":
            if val == 1:
                g = input("b (Breakout) - c (CartPole)\n")
            elif val == 2:
                g = input("Select game: w (Bipedal Walker) -  c (CartPole)\n")
            elif val == 3:
                g = input("Select game: w (Bipedal Walker) - c (CartPole)\n")
            if g != "w" and g != "b" and g != "c":
                print("Not a valid option\n")
        
        if g == "w":
            game_name = WALKER
        elif g == "b":
            game_name = BREAKOUT
        elif g == "c":
            game_name = CARTPOLE
        
        if val == 1:
            run_dqn(game_name)

        elif val == 2:
            m = None
            while m != "t" and m != "p":
                m = input("Select mode: t (training) - p (test)\n")
                if m != "t" and m != "p":
                    print("Not a valid option\n")

            if m == "t":
                train_PPO(game_name)
            elif m == "p":
                test_PPO(game_name, render=False)

        elif val == 3:
            run_genetic(game_name)

        