import tensorflow as tf
from tensorflow import keras
from abc import ABC, abstractmethod
from algorithms.PPO.PPO_utils import *


class __PPOModel(ABC):
    
    def __init__(self, state_shape, action_space, epsilon, learning_rate, gradient_clipping):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)
        self.init_models(state_shape, action_space)


    def init_models(self, state_shape, action_space):
        self.actor = build_actor(state_shape, action_space)
        self.critic = build_critic(state_shape)


    def load_models(self, checkpoint_path):
        self.actor = None
        self.critic = None


    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def compute_actor_loss(self, tape, states, actions, advantages, actions_old_log_prob):
        pass


    def update_actor(self, states, actions, advantages, actions_old_log_prob):
        tape = tf.GradientTape()
        loss = self.compute_actor_loss(tape, states, actions, advantages, actions_old_log_prob)

        trainable_variables = self.actor.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        #gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)       
        self.actor_optimizer.apply_gradients(zip(gradients, trainable_variables))
 
        return loss


    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            values = self.critic(states)
            loss = keras.losses.MSE(returns, tf.squeeze(values, axis=-1))
        
        trainable_variables = self.critic.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        #gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)       
        self.critic_optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss


    def save_models(self, path):
        self.actor.save_weights(f'{path}_actor_weights')
        self.critic.save_weights(f'{path}_critic_weights')


class DiscretePPOModel(__PPOModel):

    def __init__(self, state_shape, action_space, epsilon, learning_rate, gradient_clipping):
        super().__init__(state_shape, action_space, epsilon, learning_rate, gradient_clipping)
    
    
    def forward(self, states):
        values = tf.squeeze(self.critic(states), axis=-1)
        prob_dists = self.actor(states)
        actions = sample_from_categoricals(prob_dists)
        actions_prob = select_values_of_2D_tensor(prob_dists, actions)
        actions_log_prob = compute_log_of_tensor(actions_prob)
        return values.numpy(), actions.numpy(), actions_log_prob.numpy()


    def compute_actor_loss(self, tape, states, actions, advantages, actions_old_log_prob):
        with tape:
            prob_dists = self.actor(states)
            actions_prob = select_values_of_2D_tensor(prob_dists, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            clip_surrogate = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)*advantages
            loss = tf.minimum(ratios*advantages, clip_surrogate)            
            loss = -tf.reduce_mean(loss)
        return loss


class ContinuousPPOModel(__PPOModel):

    def __init__(self, state_shape, action_space, epsilon, learning_rate, gradient_clipping):
        super().__init__(state_shape, action_space, epsilon, learning_rate, gradient_clipping)
        self.min_action = action_space.low
        self.max_action = action_space.high


    def forward(self, states):
        values = tf.squeeze(self.critic(states), axis = -1)
        mus, log_sigmas = self.actor(states)
        actions = tf.clip_by_value(sample_from_gaussians(mus, log_sigmas), self.min_action, self.max_action)
        actions_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
        actions_log_prob = compute_log_of_tensor(actions_prob)
        return values.numpy(), actions.numpy(), actions_log_prob.numpy()


    def compute_actor_loss(self, tape, states, actions, advantages, actions_old_log_prob):
        with tape:
            mus, log_sigmas = self.actor(states)
            actions_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, actions)
            actions_log_prob = compute_log_of_tensor(actions_prob)
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)
            clip_surrogate = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon)*advantages
            loss = tf.minimum(ratios*advantages, clip_surrogate)
            loss = -tf.reduce_mean(loss)
            return loss
