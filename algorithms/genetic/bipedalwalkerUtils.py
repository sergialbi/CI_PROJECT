import gym
import torch
from constants.constants_general import *
from constants.constants_genetic import EPISODES

def run_walker(path, episodes=EPISODES, render=False):
    """
    Execute the walker environment with best model
    @param path: path to the model
    @param episodes: number of episodes to play
    @param render: boolean to decide to render or not the game
    @return score obtained in game
    """
    #load weights for torch model
    model = torch.load(path)
    fitness = 0
    env = gym.make('BipedalWalker-v3')
    obs = env.reset()
    for _ in range(episodes):
        action = model(torch.from_numpy(obs).float())
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)
        fitness += reward
        if render:
            env.render()
        if done:
            break
        
    return reward


def plotStatistics(path):
    """
    Plot statistics of the evolution of the population
    """
    # canviar format sobre com es guarda la merda aquesta.    
    


