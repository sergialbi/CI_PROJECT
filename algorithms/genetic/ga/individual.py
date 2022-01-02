from abc import ABC
from constants.constants_genetic import * 
import numpy as np
import gym
import torch

### input layer 24 observations

class Individual(ABC):
    def __init__(self, hidden_layer: int):
        self.model = define_model(hidden_layer)


    def get_action(self, obs):
        return self.model(obs)

    def get_weights(self):
        parameters = self.model.state_dict().values()
        parameters = [x.flatten() for x in parameters]
        parameters = torch.cat(parameters, 0)
        return parameters

    def update_model(self, weights):
        pass

def define_model(hidden_layer):
    return torch.nn.Sequential(
        torch.nn.Linear(24, hidden_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer, 4),
        torch.nn.Sigmoid()
    )

def run_individual(ind: Individual):
    """
    Run the individual.
    """
    obs = env.reset()
    fitness = 0
    for i in range(EPISODES):
        action = ind.get_action(obs)
        env.render()
        obs, reward, done, _ = env.step(action)
        fitness += reward
        if done:
            break




def mutation():
    """
    Mutate the weights of the individual.
    """
    pass

def crossOver(child_weights: np.array, child2_weights):
    """
    Cross over the weights of the individual.
    """
    pass

def rank():
    """
    Rank the individual.
    """
    pass
