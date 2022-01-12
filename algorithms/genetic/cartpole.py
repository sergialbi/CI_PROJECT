import numpy as np
import json
import torch
import torch.distributions as torch_distrib
import copy
from collections import OrderedDict

import algorithms.genetic.genetic_utils as genetic_utils
from environments.CartPole import CartPole
from constants.constants_genetic import *
from constants.constants_general import *

CartPole = CartPole(1)
input_layer = 4
hidden_layer = 2
output_layer = 1

class CartPoleIndividual:
    """
    Class that represent an inidividual in the genetic algorithm.
    """
    def __init__(self):
        """
        Initializes the individual with the predefined layer sizes, 0 fitness and no weights
        """
        self.model = self.initialize_model(input_layer, hidden_layer, output_layer)
        self.fitness = 0.0
        self.nn_weights = None

    def compute_fitness(self, render=False):
        """
        Runs the individual and obtains its fitness
        """
        self.fitness, self.nn_weights = run_individual_model(self.model, render=render)

    def update_model(self):
        """
        Updates the weights of the model
        """
        self.model.load_state_dict(parameters_to_state_dict(self.model.state_dict(), self.nn_weights))


    def initialize_model(self, input_layer, hidden_layer, output_layer):
        """
        Creates a torch nerual network with the predefined sizes
        @param input_layer: size of the input layer
        @param hidden_layer: size of the hidden layer
        @param output_layer: size of the output layer
        returns: the torch neural network
        """
        torch_model = torch.nn.Sequential(
                        torch.nn.Linear(input_layer, hidden_layer),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_layer, output_layer),
                        torch.nn.Sigmoid()
                        )
        return torch_model


def run_individual_model(game_model, num_episodes=EPISODES, render=False):
    """
    Compute fitness value for a model (the NN of an individual)
    @param model: the Neural Network
    @param num_episodes: number of episodes to play
    @param render: boolean to decide wether to render or not the game
    returns: fitness and the parameters of the model
    """
    game_state = CartPole.start()
    final_fitness = 0
    # For each episode to play
    for episode in range(num_episodes):
        game_state = torch.from_numpy(game_state).float() # Obtain the state of the game
        action = game_model(game_state).detach().numpy() # Let the NN decide which action to use (detach it from the network)
        
        # Apply the action to the game and obtain the new state, the reward and the done flag (wether the game has ended)
        game_state, reward, done= CartPole.step(round(action.item()))

        # Add the reward to the final fitness and check if the game has ended
        final_fitness += reward
        if done:
            break
    weights = obtain_weights_from_model(game_model)
    return final_fitness, weights


def obtain_weights_from_model(model):
    """
    Obtain and concatenate the netwrok's parameters (weights and also biases)
    @param model:  Neural Network
    returns: torch tensor with the model parameters
    """
    param = [value.flatten() for value in model.state_dict().values()]
    return torch.cat(param, 0)


def parameters_to_state_dict(model_dictionary, model_params):
    """
    Parameters of the model to state dictionary
    @param model_dictionary: OrderedDict, model of the network
    @param model_param: the parameters of the neural network
    returns: an state dictionary containing the new model of the network
    """
    model_shapes = [value.shape for value in model_dictionary.values()]
    list_shapes_product = [torch.tensor(shape).numpy().prod() for shape in model_shapes]

    split = model_params.split(list_shapes_product)
    model_vals = []
    for i in range(len(model_shapes)):
        model_vals.append(split[i].view(model_shapes[i]))

    state_dictionary = OrderedDict((key, value) for (key, value) in zip(model_dictionary.keys(), model_vals))
    return state_dictionary


def crossover(parent1_wb, parent2_wb, p_cross=CROSSOVER_RATE):
    """
    Performs single point crossover operation
    @param parent1_wb: weights and biases for one of the parents
    @param parent2_wb: weights and biases for the other parent
    """

    crossover_point = np.random.randint(0, parent1_wb.shape[0])
    child1_wb = parent1_wb.clone()
    child2_wb = parent2_wb.clone()
    if np.random.rand() < p_cross:
        aux = child1_wb[:crossover_point].clone()
        child1_wb[:crossover_point] = child2_wb[:crossover_point]
        child2_wb[:crossover_point] = aux
   
    return child1_wb, child2_wb


def mutation(individual_wb, p=MUTATION_RATE):
    """
    Mutate individual at one random point according to the probability passed as parameter and a normal distribution
    @param parent_wb: weights and biases for one of the parents
    @param p: Mutation rate
    returns: the mutated child
    """
    mutated_individual = individual_wb.clone()
    if np.random.rand() < p:
        mutation_point = np.random.randint(0, individual_wb.shape[0])
        m = torch_distrib.Normal(mutated_individual.mean(), mutated_individual.std())
        mutated_individual[mutation_point] = 5 * m.sample() + np.random.randint(-25, 25)

    return mutated_individual


def compute_stats(population):
    """
    Given a list of individuals, perform its fitness statistics
    @param population: list of individuals
    returns: a triplet of the mean, minimum and maximum fitness of the population
    """
    pop_fitness = np.asarray([individual.fitness for individual in population])
    mean = np.mean(pop_fitness)
    min = np.min(pop_fitness)
    max = np.max(pop_fitness)
    return mean, min, max


def selection(population, select_type = "roulette"):
    """
    Performs a selection of the parents from the current population
    @param population: list of individuals
    @param select_type: type of selection, either by ranking the two best individuals or by tournament wheel
    returns: The two parents according to the selection type
    """
    parent1, parent2 = None, None

    if select_type == "ranked":
        sorted_individuals = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        parent1, parent2 = sorted_individuals[0], sorted_individuals[1]

    elif select_type == "roulette":
        all_fitness = np.sum([individual.fitness for individual in population])
        probs = [individual.fitness / all_fitness for individual in population]
        ind1 = np.random.choice(len(population), p=probs)
        ind2 = np.random.choice(len(population), p=probs)
        parent1, parent2 = population[ind1], population[ind2]
    else:
        print("No such parent selection available")  
    
    return parent1, parent2



def generation(old_population, new_population, render=False, cross_value=CROSSOVER_RATE, mut_value = MUTATION_RATE):
    """
    Generate new population from old population
    @param old_population: list of old individuals
    @param new_population: list of new individuals (passed by reference)
    @param render: boolen to decide wether to render ot not
    @param corss_value: the crossover rate probability value
    @param mut_value: the mutation rate probability value
    """
   

    # For each individual (by pairs)
    for i in range(0, len(old_population) - 1, 2):

        ### STEP 0: SELECT THE PARENTS
        parent1, parent2 = selection(old_population)

        ### STEP 1: CROSSOVER ###
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.nn_weights, child2.nn_weights = crossover(parent1.nn_weights, parent2.nn_weights, cross_value)

        ### STEP 2: MUTATION ###
        child1.nn_weights, child2.nn_weights = mutation(child1.nn_weights, mut_value), mutation(child2.nn_weights, mut_value)

        ### STEP 3: UPDATE THE CURRENT MODEL ###
        child1.update_model()
        child2.update_model()

        child1.compute_fitness(render=render)
        child2.compute_fitness(render=render)

        ### STEP 4: UPDATE POPULATION ###
        always_replace = False
        is_better = always_replace or (child1.fitness + child2.fitness > parent1.fitness + parent2.fitness)
        new_population[i] = child1 if is_better else parent1
        new_population[i+1] = child2 if is_better else parent2


def cartpole_main():
    """
    Creates the population, makes the generations and selects the best model. Finally it prints the results and saves the model.
    """
    # Define some list of parameters to test them
    crossover_list = [0.9]
    mutation_list = [0.4]
    population_list = [50] # must be even
    generation_list = [25]

    # For every defined combination
    for p in population_list:
        for g in generation_list:
            for c in crossover_list:
                for m in mutation_list:
                    results_path = os.path.join(PATH_RESULTS_GENETIC_CARTPOLE,f'POPULATION={p}_MAX-GEN={g}_CROS_RATE={c}_MUT-RATE_{m}')  
                    old_population = [CartPoleIndividual() for _ in range(p)]
                    new_population = [None] * p
                    json_results = {}
                    max_old_fitness = -200
                    for gen in range(g):
                        [individual.compute_fitness() for individual in old_population]
                        generation(old_population, new_population, cross_value=c, mut_value=m)
                        best_model = sorted(new_population, key=lambda individual: individual.fitness, reverse=True)[0]
                        mean, min, max = compute_stats(new_population)

                        old_population = copy.deepcopy(new_population)
                        # Save best model
                        if max > max_old_fitness:
                            max_old_fitness = max
                            torch.save(best_model.model.state_dict(), results_path + "_dict.pt")
                            torch.save(best_model.model, results_path + "_model.pt")
                        
                        stats = {"mean":mean, "min": min,  "max": max}
                        stats_log = f"Mean: {mean} min: {min} max: {max}\n"
                        with open(results_path + '.log', "a") as f:
                            f.write(stats_log)
                            f.close()
                        print("Generation: ", gen, stats)
                        json_results[str(gen)] = stats
                    
                    
                    # ------ Save the results ------
                    
                    with open(results_path + '.json', 'w') as f:
                        json.dump(json_results, f)
                        f.close()
                    best_model = sorted(new_population, key=lambda individual: individual.fitness, reverse=True)[0]
                    run_individual_model(best_model.model, num_episodes=EPISODES, render=True)
                    genetic_utils.plotStatistics(json_results, CARTPOLE, g, p, c, m)

    CartPole.end()