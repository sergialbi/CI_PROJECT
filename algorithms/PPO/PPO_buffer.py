import numpy as np

class __PPOBuffer:

    def __init__(self, buffer_size, num_envs, state_shape, gamma, gae_lambda):
        self.state_shape = state_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pointer = 0
        self.buffer_size = buffer_size

        self.states = np.zeros((num_envs, buffer_size, *self.state_shape))
        self.rewards = np.zeros((num_envs, buffer_size))
        self.terminals = np.zeros((num_envs, buffer_size), dtype=int)
        self.values = np.zeros((num_envs, buffer_size))
        self.actions_log_prob = np.zeros((num_envs, buffer_size), dtype=np.float32)


    def store_transitions(self, states, actions, rewards, terminals, values, actions_log_prob):
        self.states[:, self.pointer] = states
        self.actions[:, self.pointer] = actions
        self.rewards[:, self.pointer] = rewards
        self.terminals[:, self.pointer] = terminals
        self.values[:, self.pointer] = values
        self.actions_log_prob[:, self.pointer] = actions_log_prob
        self.pointer = (self.pointer + 1)%self.buffer_size


    def __discount(self, values, discount_factor, bootstrapped_values=0):
        next_values = bootstrapped_values
        for i in reversed(range(values.shape[1])):
            values[:, i] = values[:, i] + discount_factor*next_values*(1 - self.terminals[:, i])
            next_values = values[:, i]
        return values


    def __compute_returns(self, bootstrapped_values):
        rewards_copy = np.copy(self.rewards)
        returns = self.__discount(rewards_copy, self.gamma, bootstrapped_values)
        return np.reshape(returns, (-1))


    def __compute_advantages(self, bootstrapped_values):
        values = np.append(self.values, np.expand_dims(bootstrapped_values, axis=-1), axis=-1)
        td_errors = self.rewards + self.gamma*values[:, 1:]*(1 - self.terminals) - values[:, :-1]
        advantages = self.__discount(td_errors, self.gamma*self.gae_lambda)

        advantages = np.reshape(advantages, (-1))
        advantages = (advantages - np.mean(advantages))/(np.std(advantages) + 1e-8)

        return advantages


    def get_buffer_size(self):
        return self.states.shape[1]


    def get_last_next_states(self):
        return self.next_states[:, -1]


    def get_transitions(self, bootstrapped_values):
        states = np.reshape(self.states, (-1, *self.state_shape))
        actions = np.reshape(self.actions, (-1, *self.actions.shape[2:]))
        returns = self.__compute_returns(bootstrapped_values)
        advantages = self.__compute_advantages(bootstrapped_values)
        action_log_probs = np.reshape(self.actions_log_prob, (-1))
        values = np.reshape(self.values, (-1))
        return states, actions, returns, advantages, action_log_probs, values


class DiscretePPOBuffer(__PPOBuffer):

    def __init__(self, buffer_size, num_envs, state_shape, gamma, gae_lambda):
        super().__init__(buffer_size, num_envs, state_shape, gamma, gae_lambda)
        self.actions = np.zeros((num_envs, buffer_size), dtype=int)


class ContinuousPPOBuffer(__PPOBuffer):

    def __init__(self, buffer_size, num_envs, state_shape, action_size, gamma, gae_lambda):
        super().__init__(buffer_size, num_envs, state_shape, gamma, gae_lambda)
        self.actions = np.zeros((num_envs, buffer_size, action_size))