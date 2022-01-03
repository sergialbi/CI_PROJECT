import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import Categorical, MultivariateNormalDiag
from gym.spaces import Discrete, Box


def compute_log_of_tensor(values):
    offsets = tf.cast(values == 0, dtype = tf.float32)*1e-6
    values = values + offsets
    return tf.math.log(values)


def select_values_of_2D_tensor(tensor, second_dimension_indices):
    first_dimension = tensor.shape[0]
    first_dimension_indices = tf.constant(range(first_dimension), shape = (first_dimension, 1))
    indices = tf.concat([first_dimension_indices, tf.expand_dims(second_dimension_indices, axis = -1)], axis = -1)
    selected_values = tf.gather_nd(tensor, indices)
    return selected_values


def sample_from_categoricals(probability_distributions):
    categorical_distribs = Categorical(probs = probability_distributions)
    return categorical_distribs.sample()


def compute_pdf_of_gaussian_samples(mus, log_sigmas, samples):
    sigmas = tf.exp(log_sigmas)
    normal_distributions = MultivariateNormalDiag(mus, sigmas)
    return normal_distributions.prob(samples)


def sample_from_gaussians(mus, log_sigmas):
    sigmas = tf.exp(log_sigmas)
    normal_distribs = MultivariateNormalDiag(mus, sigmas)
    return normal_distribs.sample()


def build_actor(state_shape, action_space):
    if isinstance(action_space, Box):
        return build_bipedal_actor(state_shape, action_space.shape)
    else: # Discrete
        return build_breakout_actor(state_shape, action_space.n)


def build_critic(state_shape):
    if len(state_shape) == 3:
        return build_breakout_critic(state_shape)
    else:
        return build_bipedal_critic(state_shape)    


def build_breakout_actor(state_shape, num_actions):
    state_input = keras.Input(state_shape)

    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu')(state_input)
    avg_pool1 = keras.layers.AveragePooling2D()(conv1)

    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu')(avg_pool1)
    avg_pool2 = keras.layers.AveragePooling2D()(conv2)

    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool2)
    avg_pool3 = keras.layers.AveragePooling2D()(conv3)

    flatten = keras.layers.Flatten()(avg_pool3)
    dense_1 = keras.layers.Dense(128, activation = 'relu')(flatten)
    prob_dist = keras.layers.Dense(num_actions, activation = 'softmax')(dense_1)

    return keras.Model(state_input, prob_dist)


def build_bipedal_actor(state_shape, action_size):
    state_input = keras.Input(state_shape)

    dense_1 = keras.layers.Dense(256, activation = 'relu')(state_input)
    dense_2 = keras.layers.Dense(256, activation = 'relu')(dense_1)

    mean = keras.layers.Dense(units = action_size, activation = 'linear')(dense_2)
    log_std = keras.layers.Dense(units = action_size, activation = 'linear')(dense_2)

    return keras.Model(state_input, [mean, log_std])


def build_breakout_critic(state_shape):
    state_input = keras.Input(state_shape)

    conv1 = keras.layers.Conv2D(16, 3, activation = 'relu')(state_input)
    avg_pool1 = keras.layers.AveragePooling2D()(conv1)

    conv2 = keras.layers.Conv2D(32, 3, activation = 'relu')(avg_pool1)
    avg_pool2 = keras.layers.AveragePooling2D()(conv2)

    conv3 = keras.layers.Conv2D(64, 3, activation = 'relu')(avg_pool2)
    avg_pool3 = keras.layers.AveragePooling2D()(conv3)

    flatten = keras.layers.Flatten()(avg_pool3)
    dense_1 = keras.layers.Dense(128, activation = 'relu')(flatten)
    state_value = keras.layers.Dense(1, activation='linear')(dense_1)

    return keras.Model(state_input, state_value)


def build_bipedal_critic(state_shape):
    state_input = keras.Input(state_shape)

    dense_1 = keras.layers.Dense(256, activation = 'relu')(state_input)
    dense_2 = keras.layers.Dense(256, activation = 'relu')(dense_1)
    state_value = keras.layers.Dense(1, activation='linear')(dense_2)

    return keras.Model(state_input, state_value)