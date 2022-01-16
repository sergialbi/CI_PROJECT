import numpy as np
from numpy import random
from algorithms.PPO.PPO_model import DiscretePPOModel, ContinuousPPOModel
from algorithms.PPO.PPO_buffer import DiscretePPOBuffer, ContinuousPPOBuffer
from constants.constants_ppo import MAX_KL_DIVERG

class __PPOAgent:

    def __init__(self, epochs):
        self.epochs = epochs
        self.last_values = None
        self.last_actions = None
        self.last_actions_log_prob = None


    def step(self, states):
        self.last_actions, self.last_actions_log_prob, self.last_values = self.model.forward(states)
        return self.last_actions


    def store_transitions(self, states, rewards, terminals):
        self.buffer.store_transitions(states, self.last_actions, rewards, terminals, self.last_values,
            self.last_actions_log_prob)


    def train(self, batch_size, last_next_states, current_iteration, total_iterations):
        _, _, bootstrapped_values = self.model.forward(last_next_states)

        states, actions, returns, advantages, actions_log_prob, values = self.buffer.get_transitions(bootstrapped_values)

        num_transitions = states.shape[0]
        indices = np.arange(num_transitions)
        num_batches = int(np.ceil(num_transitions/batch_size))

        annealing_fraction = 1 - current_iteration/total_iterations

        self.model.apply_annealing(annealing_fraction) 

        for _ in range(self.epochs):

            np.random.shuffle(indices)

            for i in range(num_batches):

                start_index = i*batch_size
                end_index = start_index+batch_size if start_index+batch_size < num_transitions else num_transitions
                indices_batch = indices[start_index:end_index]

                actor_loss, critic_loss, kl_divergence, learning_rate = self.model.update_model(states[indices_batch], 
                    actions[indices_batch], advantages[indices_batch], returns[indices_batch], 
                    actions_log_prob[indices_batch], values[indices_batch])

        return {'Actor Loss': actor_loss.numpy(), 'Critic Loss': critic_loss.numpy(), 
            'KL Divergence': kl_divergence.numpy(), 'Learning Rate': learning_rate}

    def reset_buffer(self):
        self.buffer.reset_buffer()

    def load_models(self, path):
        self.model.load_models(path)

    def save_models(self, path):
        self.model.save_models(path)


class DiscretePPOAgent(__PPOAgent):

    def __init__(self, state_shape, num_actions, buffer_size, num_envs, gamma, gae_lambda, epsilon, epochs, 
        learning_rate, gradient_clipping, max_kl_diverg):

        super().__init__(epochs)
        self.buffer = DiscretePPOBuffer(buffer_size, num_envs, state_shape, gamma, gae_lambda)
        self.model = DiscretePPOModel(state_shape, num_actions, epsilon, learning_rate, gradient_clipping, max_kl_diverg)


class ContinuousPPOAgent(__PPOAgent):

    def __init__(self, state_shape, action_space, buffer_size, num_envs, gamma, gae_lambda, epsilon, epochs, 
        learning_rate, gradient_clipping, max_kl_diverg):

        super().__init__(epochs)
        self.buffer = ContinuousPPOBuffer(buffer_size, num_envs, state_shape, action_space.shape[0], gamma, gae_lambda)
        self.model = ContinuousPPOModel(state_shape, action_space, epsilon, learning_rate, gradient_clipping, max_kl_diverg)