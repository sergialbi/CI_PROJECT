import numpy as np
import torch
import torch.distributions as tdist
import gym
import copy
import json
import algorithms.genetic.genetic_utils as genetic_utils
from collections import OrderedDict
from algorithms.genetic.ga.individual import Individual
from constants.constants_genetic import *
from constants.constants_general import *
from environments.Breakout import Breakout
from environments.Breakout import Breakout
breakout = Breakout(4, (100,100))


class Individual:
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            # sizes of the layers
            input_layer, hidden_layer, output_layer = 3, 33600, 4
            self.model = create_model(input_layer, hidden_layer, output_layer)
        
        self.fitness = 0.0
        self.nn_weights = None

    def compute_fitness(self, render=False):
        self.fitness, self.nn_weights = run_individual(self.model, render=render)

    def update_model(self):
        # Update model weights and biases
        self.model.load_state_dict(parameter_list_to_state_dict(self.model.state_dict(), self.nn_weights))


def create_model(input_layer, hidden_layer, output_layer):
    return torch.nn.Sequential(
        torch.nn.Linear(input_layer, hidden_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer, output_layer),
        torch.nn.Sigmoid()
    )


def parameter_list_to_state_dict(model_dict, model_parameters):
    """
    Parameters of the model to state dict
    :param model_dict: OrderedDict, model schema
    :param model_parameters: List of model parameters
    :return:
    """
    shapes = [x.shape for x in model_dict.values()]
    shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

    partial_split = model_parameters.split(shapes_prod)
    model_values = []
    for i in range(len(shapes)):
        model_values.append(partial_split[i].view(shapes[i]))

    return OrderedDict((key, value) for (key, value) in zip(model_dict.keys(), model_values))


def create_model(input_layer, hidden_layer, output_layer):
    return torch.nn.Sequential(
        torch.nn.Linear(input_layer, hidden_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer, output_layer),
        torch.nn.Sigmoid()
    )


def run_individual(model, num_episodes=EPISODES, render=False):
    """
    Calculate fitness for a NN model
    :param model: the NN
    :param num_episodes: number of episodes to play
    :param render: bolean to decide to render or not the game
    :return: fitness and the parameters of the model
    """
    obs = breakout.start()
    fitness = 0
    for _ in range(num_episodes):
        obs = torch.from_numpy(obs).float()
        print(obs.shape)
        action = model(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = breakout.step(action)
        fitness += reward
        if done:
            break
    weights_and_biases = state_dict_to_parameter_list(model)
    return fitness, weights_and_biases


def state_dict_to_parameter_list(model):
    """
    Concatenate the model parameters
    :param model:  NN
    :return: torch tensor with the model weights and biases
    """
    param = model.state_dict().values()
    param = [x.flatten() for x in param]
    return torch.cat(param, 0)


def parameter_list_to_state_dict(model_dict, model_parameters):
    """
    Parameters of the model to state dict
    :param model_dict: OrderedDict, model schema
    :param model_parameters: List of model parameters
    :return:
    """
    shapes = [x.shape for x in model_dict.values()]
    shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

    partial_split = model_parameters.split(shapes_prod)
    model_values = []
    for i in range(len(shapes)):
        model_values.append(partial_split[i].view(shapes[i]))

    return OrderedDict((key, value) for (key, value) in zip(model_dict.keys(), model_values))


def crossover(parent1_wb, parent2_wb , p=CROSSOVER_RATE):
    """
    Perform a single point crossover operation
    :param parent1_wb: weights and biases for one of the parents
    :param parent2_wb: weights and biases for the other parent
    """
    position = np.random.randint(0, parent1_wb.shape[0])
    child1_wb = parent1_wb.clone()
    child2_wb = parent2_wb.clone()
    if np.random.rand() < p:
        tmp = child1_wb[:position].clone()
        child1_wb[:position] = child2_wb[:position]
        child2_wb[:position] = tmp
    return child1_wb, child2_wb

def mutation(parent_wb, p=MUTATION_RATE):
    """
    Mutate parent using normal distribution
    :param parent_wb: weights and biases for one of the parents
    :param p: Mutation rate
    return: the mutated child
    """
    child_wb = parent_wb.clone()
    if np.random.rand() < p:
        position = np.random.randint(0, parent_wb.shape[0])
        n = tdist.Normal(child_wb.mean(), child_wb.std())
        child_wb[position] = 5 * n.sample() + np.random.randint(-20, 20)
    return child_wb


def compute_stats(population):
    """
    Given a list of individuals, perform its fitness statistics
    :param population: list of individuals
    return: a triplet of the mean, minimum and maximum fitness of the population
    """
    population_fitness = list(map(lambda individual: individual.fitness, population))
    mean = np.mean(population_fitness)
    min = np.min(population_fitness)
    max = np.max(population_fitness)
    return mean, min, max


def selection(population):
    """
    Perferom a selection of parents from the current population (two best individuals)
    :param population: list of individuals
    return: The two best individuals of population
    """
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    parent1, parent2 = sorted_population[0], sorted_population[1]
    return parent1, parent2


def generation(old_population, new_population, render=False):
    """
    Generate new population
    :param old_population: list of old individuals
    :param new_population: list of new individuals (passed by reference
    """
    parent1, parent2 = selection(old_population)

    for i in range(0, len(old_population) - 1, 2):

        ### STEP 1: CROSSOVER ###
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.nn_weights, child2.nn_weights = crossover(parent1.nn_weights, parent2.nn_weights)

        ### STEP 2: MUTATION ###
        child1.nn_weights = mutation(child1.nn_weights)
        child2.nn_weights = mutation(child2.nn_weights)

        ### STEP 3: UPDATE THE CURRENT MODEL ###
        child1.update_model()
        child2.update_model()

        child1.compute_fitness(render=render)
        child2.compute_fitness(render=render)

        ### STEP 4: UPDATE POPULATION ###
        new_population[i] = child1
        new_population[i + 1] = child2


def breakout_main():
    path = f'{PATH_RESULTS_GENETIC_BREAKOUT}POPULATION={POPULATION_SIZE}_MAX-GEN={GENERATIONS}_MUT-RATE_{MUTATION_RATE}'

    old_population = [Individual() for _ in range(POPULATION_SIZE)]
    new_population = [None] * POPULATION_SIZE
    results = {}
    max_fitness = -500
    for g in range(GENERATIONS):
        [individual.compute_fitness() for individual in old_population]
        generation(old_population, new_population)
        mean, min, max = compute_stats(new_population)
        old_population = copy.deepcopy(new_population)
        results[str(g)] = {'mean': mean, 'min': min, 'max': max}
        stats = f"Mean: {mean}\tmin: {min}\tmax: {max}\n"
        with open(path + '.log', "a") as f:
            f.write(stats)
        print("Generation: ", g, stats)
        if max > max_fitness:
            max_fitness = max
            best_model = sorted(new_population, key=lambda individual: individual.fitness, reverse=True)[0]
            torch.save(best_model.model.state_dict(), path + '_dict.pt')
            torch.save(best_model.model, path + '.pt')

    with open(path + '.log', "a") as f:
            f.write(stats)

    with open(path + '.json', "a") as f:
        json.dump(results, f)

    genetic_utils.plotStatistics(results, BREAKOUT)
    breakout.end()