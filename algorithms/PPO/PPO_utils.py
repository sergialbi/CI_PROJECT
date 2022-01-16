import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import Categorical, MultivariateNormalDiag
import numpy as np
import matplotlib.pyplot as plt


class DiscreteActorCritic(keras.Model):
    
    def __init__(self, num_actions):
        super().__init__()
        self.dense_1 = keras.layers.Dense(64, activation='relu')
        self.batch_norm = keras.layers.BatchNormalization()
        self.dense_2 = keras.layers.Dense(32, activation='relu')

        self.logits = keras.layers.Dense(num_actions)
        self.state_value = keras.layers.Dense(1)

    def common(self, states):
        output_1 = self.dense_1(states)
        output_2 = self.batch_norm(output_1)
        output_3 = self.dense_2(output_2)
        logits = self.logits(output_3)
        state_values = self.state_value(output_3)
        return logits, tf.squeeze(state_values, axis=-1)

    def call(self, states):
        logits, state_values = self.common(states)
        prob_distribs = Categorical(logits=logits)
        actions = prob_distribs.sample()
        return actions, prob_distribs.log_prob(actions), state_values

    def call_update(self, states, actions):
        logits, state_values = self.common(states)
        prob_distribs = Categorical(logits=logits)
        return prob_distribs.log_prob(actions), state_values

    
class ContinuousActorCritic(keras.Model):

    def __init__(self, action_size, min_action, max_action):
        super().__init__()
        self.dense_1 = keras.layers.Dense(512, activation="relu")
        self.dense_2 = keras.layers.Dense(256, activation="relu")
        self.dense_3 = keras.layers.Dense(64, activation="relu")

        self.mu = keras.layers.Dense(action_size, activation='tanh')
        self.state_value = keras.layers.Dense(1)

        self.log_std = -0.5 * np.ones(action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

        self.min_action = min_action
        self.max_action = max_action

    def common(self, states):
        output_1 = self.dense_1(states)
        output_2 = self.dense_2(output_1)
        output_3 = self.dense_3(output_2)
        state_values = self.state_value(output_3)
        mus = self.mu(output_2)
        state_values = tf.squeeze(state_values, axis=-1)
        return mus, state_values

    def log_probs(self, actions, mus):
        # Diagonal gaussian likelihood
        pre_sum = -0.5 * (((actions - mus)/(self.std + 1e-8))**2 + 2*self.log_std + tf.math.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=-1)

    def get_actor_values(self, mus):
        actions = mus + np.random.uniform(self.min_action, self.max_action)*self.std
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        log_probs = self.log_probs(actions, mus)
        return actions, log_probs

    def call(self, states):
        mus, state_values = self.common(states)
        actions, log_probs = self.get_actor_values(mus)
        return actions, log_probs, state_values
        
    def call_update(self, states, actions):
        mus, state_values = self.common(states)
        log_probs = self.log_probs(actions, mus)
        return log_probs, state_values


class PPOResults:

    def __init__(self, num_envs, reward_scale, save_path, test=False):
        self.episode_rewards = []
        self.rewards_avg = []
        self.env_episode_reward = np.zeros((num_envs), dtype=float)
        self.env_episode_steps = np.zeros((num_envs), dtype=int)
        self.update_counter = 0
        self.reward_scale = reward_scale
        self.save_path = save_path

        if test == False:
            self.summary_writer = tf.summary.create_file_writer(save_path)


    def add_transition_rewards(self, rewards):
        self.env_episode_reward[:] += rewards
        self.env_episode_steps[:] += 1


    def plot_reward_curve(self, env_name, test=False):
        num_episodes = len(self.episode_rewards)
        range_episodes = range(num_episodes)
        plt.figure(figsize=(18, 9))
        plt.plot(range_episodes, self.episode_rewards, 'b', label='Episode reward')
        plt.plot(range_episodes, self.rewards_avg, 'r', label='Last 50 average')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        if test:
            title = 'Evolution of the episode reward during test'
            path = f'{self.save_path}/{env_name}_PPO_test_curve.png'
        else:
            title = 'Evolution of the episode reward during training'
            path = f'{self.save_path}/{env_name}_PPO_training_curve.png'

        plt.title(title)
        plt.savefig(path)
        plt.close()
    

    def end_episode(self, env_index):
        env_episode_reward = self.env_episode_reward[env_index]/self.reward_scale
        self.episode_rewards.append(env_episode_reward)

        last_50_avg = np.mean(self.episode_rewards[-50:])
        self.rewards_avg.append(last_50_avg)

        self.env_episode_reward[env_index] = 0
        self.env_episode_steps[env_index] = 0

        return env_episode_reward, last_50_avg


    def write_metrics(self, train_state):
        with self.summary_writer.as_default():
            for name, value in train_state.items():
                tf.summary.scalar(name, value, self.update_counter)

        self.update_counter += 1