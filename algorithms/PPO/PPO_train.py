from constants.constants_ppo import *
from constants.constants_general import BREAKOUT, WALKER, PATH_RESULTS_PPO_BREAKOUT, PATH_RESULTS_PPO_WALKER
from algorithms.PPO.PPO_agent import DiscretePPOAgent, ContinuousPPOAgent
from environments import BipedalWalker, Breakout, MultiEnvironmentManager


def run_PPO(environment_name):
    
    if environment_name is BREAKOUT:
        agent_class = DiscretePPOAgent
        env_params = {'num_stacked' : FRAMES_STACKED, 'frame_resize' : FRAMES_RESIZE}
        env_function = Breakout.Breakout
        save_path = PATH_RESULTS_PPO_BREAKOUT
    
    elif environment_name is WALKER:
        agent_class = ContinuousPPOAgent
        env_params = {}
        env_function = BipedalWalker.BipedalWalker
        save_path = PATH_RESULTS_PPO_WALKER

    environment = MultiEnvironmentManager.MultiEnvironmentManager(env_function, NUM_ENVS, **env_params)

    agent = agent_class(environment.get_state_shape(), environment.get_action_space(), BUFFER_SIZE, NUM_ENVS, GAMMA, 
            GAE_LAMBDA, EPSILON, EPOCHS, LEARNING_RATE, GRADIENT_CLIPPING)

    states = environment.start()

    for iteration in range(ITERATIONS):

        for _ in range(ITERATION_STEPS):

            actions = agent.step(states)
            rewards, next_states, terminals = environment.step(actions)

            agent.store_transitions(states, rewards, terminals, next_states)

            states = next_states
        
        losses = agent.train(BATCH_SIZE)
        agent.reset_buffer()

        # TODO: Save best
        if iteration%10 == 0:
            agent.save_models(save_path)

        print("======== Iteration " + str(iteration) + " Finished ============")

        
    environment.end()
    # TODO: Save model and plot results
    agent.save_models(save_path)