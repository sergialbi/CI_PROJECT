import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from algorithms.PPO.PPO_utils import DiscreteActorCritic, ContinuousActorCritic


class __PPOModel:
    
    def __init__(self, epsilon, learning_rate, gradient_clipping, max_kl_diverg):
        self.original_epsilon = epsilon
        self.original_learning_rate = learning_rate
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.max_kl_diverg = max_kl_diverg
            

    def forward(self, states):
        actions, actions_log_prob, state_values = self.model(states)
        return actions.numpy(), actions_log_prob.numpy(), state_values.numpy()

    
    def get_actions_log_prob(self, states, actions):
        return self.model.get_actions_log_probs(states, actions)


    def apply_annealing(self, annealing_fraction):
        self.epsilon = self.original_epsilon*annealing_fraction
        self.learning_rate = self.original_learning_rate*annealing_fraction
        K.set_value(self.optimizer.learning_rate, self.learning_rate)


    def compute_losses(self, tape, states, actions, advantages, returns, actions_old_log_prob, old_values):
        with tape:

            actions_log_prob, state_values = self.model.call_update(states, actions)
           
            ratios = tf.exp(actions_log_prob - actions_old_log_prob)

            clip_surrogate = tf.clip_by_value(ratios, 1-self.epsilon, 1+self.epsilon)*advantages
            actor_loss = tf.reduce_mean(tf.minimum(ratios*advantages, clip_surrogate))

            state_values_clipped = old_values + tf.clip_by_value(state_values - old_values, -self.epsilon, self.epsilon)
            critic_loss_unclipped = (returns - state_values)**2
            critic_loss_clipped = (returns - state_values_clipped)**2
            
            critic_loss = 0.5 * tf.reduce_mean(tf.maximum(critic_loss_clipped, critic_loss_unclipped))

            kl_diverg = tf.reduce_mean(actions_old_log_prob - actions_log_prob)
            loss = -actor_loss + critic_loss

            #loss = tf.where(kl_diverg > self.max_kl_diverg, tf.stop_gradient(loss), loss)

        return loss, actor_loss, critic_loss, kl_diverg


    def update_model(self, states, actions, advantages, returns, actions_old_log_prob, old_values):
        tape = tf.GradientTape()
        loss, actor_loss, critic_loss, kl_divergence = self.compute_losses(tape, states, actions, advantages, returns,
            actions_old_log_prob, old_values)

        trainable_variables = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        has_nans = tf.reduce_any([tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients])
        if has_nans:
            print("NANs!")
            print(loss, actor_loss, critic_loss, kl_divergence)

        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)       
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
 
        return actor_loss, critic_loss, kl_divergence, self.optimizer.get_config()['learning_rate']


    def are_models_saved(self):
        files = os.listdir(self.models_path)
        return len([file_name for file_name in files 
            if file_name.startswith('actor_weights') or file_name.startswith('critic_weights')]) > 0

    def load_models(self, models_path):
        if "checkpoint" in os.listdir(models_path):
            self.model.load_weights(f'{models_path}/models_weights') 

    def save_models(self, models_path):
        self.model.save_weights(f'{models_path}/models_weights')


class DiscretePPOModel(__PPOModel):

    def __init__(self, state_shape, num_actions, epsilon, learning_rate, gradient_clipping, max_kl_diverg):
        self.model = DiscreteActorCritic(num_actions)
        super().__init__(epsilon, learning_rate, gradient_clipping, max_kl_diverg)


class ContinuousPPOModel(__PPOModel):

    def __init__(self, state_shape, action_space, epsilon, learning_rate, gradient_clipping, max_kl_diverg):
        self.model = ContinuousActorCritic(action_space.shape[0], action_space.low, action_space.high)
        super().__init__(epsilon, learning_rate, gradient_clipping, max_kl_diverg)
