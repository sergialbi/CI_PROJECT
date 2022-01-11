import random
import numpy as np
import gym
import neural_network
import time
import math

env_name = "Breakout-ram-v0"

env = gym.make(env_name)


print(env.action_space)  # ovo ispisuje koje akcije mozemo napraviti, discrete (4) znaci da ima 4 akcije
print(env.observation_space)  # ovo ispisuje što vidimo i što ćemo gurat u neuronsku (mozda u izmjenjenom obliku)
print("action size: ", env.action_space.n)
print("observation space sample: ")
print(env.observation_space.sample())
print("observation space je dimenzije: ", env.observation_space.sample().dtype)
print("observation space size:", env.observation_space.sample().size)

# Choosing crossover function
crossover_choice = input("Choose crossover function: 1 for random, 2 for arithmetic, 3 for BLX-alpha\n")
if crossover_choice != "1" and crossover_choice != "2" and crossover_choice != "3":
    crossover_choice = 3
    print("Crossover function is BLX-alpha (default)")

# Choosing probability of mutation
try:
    mutation_probability = float(
        input("Define probability for mutation to occur, must be between 0 and 1 (use '.' not ',')\n"))
    if float(mutation_probability) < 0 or float(mutation_probability) > 1:
        raise Exception
except:
    mutation_probability = 0.001
    print("Mutation probability set to 0.001 (default)")

# Choosing deviation of mutation
try:
    mutation_deviation = float(input("Define deviation for the mutation\n"))
    if mutation_deviation <= 0:
        raise Exception
except:
    mutation_deviation = 1.5
    print("Mutation deviation set to 1.5 (default)")

# Choosing population size
try:
    population_size = int(input("Define population size of a single generation\n"))
    if population_size <= 1:
        raise Exception
except:
    population_size = 20
    print("Population size set to 20 (default)")

# Choosing number of elite specimen
try:
    num_bestIndividuals = int(input("Define number of units for elitism\n"))
    if num_bestIndividuals < 1 or num_bestIndividuals > population_size:
        raise Exception
except:
    num_bestIndividuals = 4
    print("Number of elite units set to 4 (default)")

# Choosing number of generations
try:
    num_generations = int(input("Define number of generations\n"))
    if num_generations < 1:
        raise Exception
except:
    num_generations = 1000
    print("Number of generations set to 1000 (default)")

parameters = {
    "mutation_probability": mutation_probability,
    "mutation_deviation": mutation_deviation
}

currentCrossover = "crossover_blx_alpha"
if crossover_choice == "1":
    currentCrossover = "crossover_random"
elif crossover_choice == "2":
    currentCrossover = "crossover_arithmetic"
elif crossover_choice == "3":
    currentCrossover = "crossover_blx_alpha"

print("Population size:", population_size, file=open("results.txt", "a"))
print("Number of generations:", num_generations, file=open("results.txt", "a"))
print("Number of best individuals:", num_bestIndividuals, file=open("results.txt", "a"))
print("Mutation probability:", mutation_probability, file=open("results.txt", "a"))
print("Mutation deviation:", mutation_deviation, file=open("results.txt", "a"))
print("Number of inputs:", neural_network.no_inputs, file=open("results.txt", "a"))
print("Number of outputs:", neural_network.no_outputs, file=open("results.txt", "a"))
print("Number of layers:", 2, file=open("results.txt", "a"))
print("Size of first layer:", neural_network.first_layer_size, file=open("results.txt", "a"))
print("Size of second layer:", neural_network.second_layer_size, file=open("results.txt", "a"))
print("Current activation function:", neural_network.leakyReLu.__name__, file=open("results.txt", "a"))
print("Current crossover algorithm used: " + currentCrossover, file=open("results.txt", "a"))
print("_____________________________________________________________", file=open("results.txt", "a"))

class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n

    def get_action(self, state, brain):
        action = brain.think(state)
        return action


agent = Agent(env)
state = env.reset()

population = [neural_network.NeuralNet(**parameters) for _ in range(population_size)]


#max average koji se pojavio pri testiranju
maxAverage = 0
# max fitnes koji se pojavio pri testiranju
maxFitness = 0
# bestIndividual = neural_network.NeuralNet(**parameters)

# za X generacija:
for _ in range(num_generations):
    print("GENERATION " + str(_ + 1))
    total = 0
    # prevState = np.zeros(128)
    for i in range(population_size):
        state = env.reset()
        done = False
        totalReward = 0
        counter = 0
        while not done:
            action = agent.get_action(state, population[i])
            state, reward, done, info = env.step(action)
            totalReward += reward
            # preskace jedinke koje zapnu i ne rade nista
            counter += 1
            if counter > 250 and totalReward < 1:
                totalReward = -1
                break
        population[i].fitness = totalReward

        print("total reward for organism" + str(i + 1) + ": " + str(totalReward))

    n = math.floor(population_size/3)
    final_list = []

    population.sort(key=lambda x: x.fitness, reverse=True)

    final_list = population[:n]
    for i in range(n):
        total += final_list[i].fitness

    average = total/n

    if average > maxAverage:
        maxAverage = average

    sortedPopulation = population

    print("BEST IN GENERATION " + str(_ + 1) + " FITNESS: " + str(sortedPopulation[0].fitness), file=open("results.txt", "a"))
    print("AVERAGE IN GENERATION " + str(_ + 1) + " FITNESS: ", average, file=open("results.txt", "a"))
    print("", file=open("results.txt", "a"))

    # spremanje najbolje jedinke
    if population[0].fitness > maxFitness:
        maxFitness = population[0].fitness
        bestIndividual = population[0]

    newPopulation = []
    # spremi odredeni broj najboljih jedinki u novu populaciju  - ELITIZAM
    for i in range(num_bestIndividuals):
        newPopulation.append(sortedPopulation[i])
        newPopulation[i].fitness = 0
    # preostali dio nove populacije popuni crossoverom korsiteci najboljih 33% trenutacne populacije - SELEKCIJA
    s = population_size - num_bestIndividuals
    for i in range(s):
        parent1 = random.choice(sortedPopulation[:math.floor(population_size / 3)])
        parent2 = random.choice(sortedPopulation[:math.floor(population_size / 3)])
        if crossover_choice == "1":
            child = parent1.crossover_random(parent2)
        elif crossover_choice == "2":
            child = parent1.crossover_arithmetic(parent2)
        elif crossover_choice == "3":
            child = parent1.crossover_blx_alpha(parent2)
        else:
            child = parent1.crossover_blx_alpha(parent2)
        newPopulation.append(child)

    # mutacija nove populacije (osim elitnih)

    for i in range(population_size - num_bestIndividuals):
        newPopulation[num_bestIndividuals + i].mutate()

    population = newPopulation
done = False
while not done:
    action = agent.get_action(state, bestIndividual)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.2)

print("MAX FITNESS WHICH OCCURRED:", maxFitness, file=open("results.txt", "a"))
print("MAX AVERAGE WHICH OCCURRED:", maxAverage, file=open("results.txt", "a"))
print("_____________________________________________________________", file=open("results.txt", "a"))
print("_____________________________________________________________", file=open("results.txt", "a"))
print("", file=open("results.txt", "a"))

print(bestIndividual.chromosomes)
t = time.localtime()
print(time.strftime("%H:%M:%S", t))