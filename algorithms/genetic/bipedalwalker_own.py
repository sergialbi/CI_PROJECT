from algorithms.genetic.ga.population import Population
from algorithms.genetic.ga.individual import Individual, run_individual
from constants.constants_genetic import * 
import gym

def walker_main():
    env = gym.make('BipedalWalker-v3')
    env.seed(256)
    env.reset()

    #population = [Individual() for _ in range(POPULATION_SIZE)]
    results = []

    for i in range(1000):
        env.render()

    #get action_space
    '''for _ in range(GENERATIONS):
        for i in range(len(population)):
            result = run_individual(population[i])
            results.append(result)
        #ranking of better results 
        #crossOver
        #mutation'''
    env.close()


class Walker_Individual(Individual):
    def __init__(self):
        pass



   