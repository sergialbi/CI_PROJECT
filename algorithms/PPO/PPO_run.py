from sys import path
from constants.constants_ppo import *
from constants.constants_general import CARTPOLE, WALKER, PATH_RESULTS_PPO_WALKER, PATH_RESULTS_PPO_CARTPOLE
from algorithms.PPO.PPO_agent import DiscretePPOAgent, ContinuousPPOAgent
from environments import BipedalWalker, MultiEnvironmentManager, CartPole
from algorithms.PPO.PPO_utils import PPOResults


def create_environments(environment_name, num_envs=NUM_ENVS, render=False):
    if environment_name is WALKER:
        env_params = {'reward_scale': REWARD_SCALE, 'render': render}
        env_function = BipedalWalker.BipedalWalker

    elif environment_name is CARTPOLE:
        env_params = {'reward_scale': REWARD_SCALE, 'render': render}
        env_function = CartPole.CartPole

    return MultiEnvironmentManager.MultiEnvironmentManager(env_function, num_envs, **env_params)


def get_models_path(environment_name):
    if environment_name is WALKER:
        return PATH_RESULTS_PPO_WALKER
    elif environment_name is CARTPOLE:
        return PATH_RESULTS_PPO_CARTPOLE
    else:
        raise KeyError(environment_name)


def create_agent(environment_name, buffer_size, state_shape, action_space):
    if environment_name is WALKER:
        agent = ContinuousPPOAgent(state_shape, action_space, buffer_size, NUM_ENVS, GAMMA, GAE_LAMBDA, EPSILON, EPOCHS, 
            LEARNING_RATE, GRADIENT_CLIPPING, MAX_KL_DIVERG)

    elif environment_name is CARTPOLE:
        num_actions = action_space.n
        agent = DiscretePPOAgent(state_shape, num_actions, buffer_size, NUM_ENVS, GAMMA, GAE_LAMBDA, EPSILON, 
            EPOCHS, LEARNING_RATE, GRADIENT_CLIPPING, MAX_KL_DIVERG)

    return agent


def train_PPO(environment_name):
    save_path = get_models_path(environment_name)
    environments = create_environments(environment_name)
    agent = create_agent(environment_name, BUFFER_SIZE, environments.get_state_shape(), environments.get_action_space())
    results = PPOResults(NUM_ENVS, REWARD_SCALE, save_path)

    states = environments.start()

    best_average = -10000

    for iteration in range(TRAIN_ITERATIONS):

        for _ in range(ITERATION_STEPS):

            actions = agent.step(states)
            
            next_states, rewards, terminals = environments.step(actions)

            results.add_transition_rewards(rewards)

            agent.store_transitions(states, rewards, terminals)
            states = next_states

            for index in range(len(terminals)):
                if terminals[index]:
                    episode_reward, average = results.end_episode(index)
                    results.plot_reward_curve(environment_name)
                    print(f'Iteration {iteration}/{TRAIN_ITERATIONS}: Env {index}, Reward: {episode_reward},' + 
                        f' Last 50 Average: {average:.2f}')

                    if average >= best_average:
                        best_average = average
                        agent.save_models(save_path)

        train_state = agent.train(BATCH_SIZE, states, iteration, TRAIN_ITERATIONS)
        results.write_metrics(train_state)

    environments.end()
 

def test_PPO(environment_name, render=False):
    environment = create_environments(environment_name, num_envs=1, render=render)
    models_path = get_models_path(environment_name)
    agent = create_agent(environment_name, BUFFER_SIZE, environment.get_state_shape(), environment.get_action_space())
    agent.load_models(models_path)

    results = PPOResults(1, REWARD_SCALE, models_path, test=True)
    
    average_episode_reward = 0

    state = environment.start()

    for episode in range(TEST_EPISODES):
        terminal = False
        total_reward = 0

        while not terminal:
            action = agent.step(state)
            state, reward, terminal = environment.step(action)

            results.add_transition_rewards(reward)

            total_reward += reward[0]/REWARD_SCALE
            average_episode_reward += total_reward

        episode_reward, average = results.end_episode(0)
        results.plot_reward_curve(environment_name, test=True)
        print(f'Test Episode {episode}/{TEST_EPISODES}: Reward: {episode_reward},' + 
            f' Last 50 Average: {average:.2f}')

    environment.end()

    return average_episode_reward