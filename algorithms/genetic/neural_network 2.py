import random

import numpy as np

no_inputs = 128
first_layer_size = 256
second_layer_size = 64
no_outputs = 4




# loša funkcija, definiraj ih još
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tangens_hiperbolni(x):
    return np.tanh(x)


# dodatne funkcije, mozda ce se pokazati boljom pri testiranju
# ako se koriste, namjestiti think funkciju
# Rectified Linear Activation Function
def relu(x):
    return np.maximum(x, 0)


# popravlja 'dying ReLu' problem
def leakyReLu(x):
    return np.where(x > 0, x, x * 0.01)


# dodatne opcije: Parametric ReLu, Exponential, Concatenated, ReLu-6


def activation_function(x):
    return leakyReLu(x)


class NeuralNet(object):
    # napravi neuronsku sa random tezinama
    def __init__(self, mutation_probability=0.001, mutation_deviation=1.5, chromosomes=None):
        self.fitness = 0
        self.mutation_probability = mutation_probability
        self.mutation_deviation = mutation_deviation
        if chromosomes is None:  # ako radimo novi organizam
            np.random.seed(None)
            # postavlja tezine neuronske mreze na random vrijednosti u rasponu [-1,1]
            self.weights_first_layer = 2 * np.random.random((no_inputs, first_layer_size)) - 1
            self.weights_second_layer = 2 * np.random.random((first_layer_size, second_layer_size)) - 1
            self.weights_output_layer = 2 * np.random.random((second_layer_size, no_outputs)) - 1
            self.chromosomes = self.__save_as_chromosomes()  # sprema tezine u 1D array
        else:  # ako radimo organizam sa zadanim kromosomima

            self.weights_first_layer, self.weights_second_layer, self.weights_output_layer = self.chromosomes_to_weights(
                chromosomes)
            self.chromosomes = chromosomes

        return

    def think(self, input):
        first_layer = activation_function(np.dot(input, self.weights_first_layer))
        second_layer = activation_function(np.dot(first_layer, self.weights_second_layer))
        output = activation_function(np.dot(second_layer, self.weights_output_layer))
        # print("output : ", output)

        maximum = -10000
        index = -1
        for i in range(no_outputs):
            if output[i] > maximum:
                maximum = output[i]
                index = i

        return index

    def crossover_random(self, parent2):
        child_chromosome = np.zeros(self.chromosomes.size)

        for i in range(self.chromosomes.size):
            if random.random() < 0.5:
                child_chromosome[i] = self.chromosomes[i]
            else:
                child_chromosome[i] = parent2.chromosomes[i]

        return NeuralNet(
            chromosomes=child_chromosome,
            mutation_probability=self.mutation_probability,
            mutation_deviation=self.mutation_deviation
        )

    def crossover_arithmetic(self, parent2):
        child_chromosome = np.zeros(self.chromosomes.size)

        for i in range(self.chromosomes.size):
            # crossover implementiran aritmetickom sredinom
            child_chromosome[i] = self.chromosomes[i] + parent2.chromosomes[i]
            child_chromosome[i] /= 2
        return NeuralNet(
            chromosomes=child_chromosome,
            mutation_probability=self.mutation_probability,
            mutation_deviation=self.mutation_deviation
        )

    def crossover_blx_alpha(self, parent2):
        child_chromosome = np.zeros(self.chromosomes.size)
        alpha = 0.5

        for i in range(self.chromosomes.size):
            if self.chromosomes[i] < parent2.chromosomes[i]:
                x1 = self.chromosomes[i]
                x2 = parent2.chromosomes[i]
            else:
                x1 = parent2.chromosomes[i]
                x2 = self.chromosomes[i]

            lower = x1 - alpha * (x2 - x1)
            upper = x2 + alpha * (x2 - x1)

            child_chromosome[i] = random.uniform(lower, upper)
        return NeuralNet(
            chromosomes=child_chromosome,
            mutation_probability=self.mutation_probability,
            mutation_deviation=self.mutation_deviation
        )

    def mutate(self):
        for i in range(self.chromosomes.size):
            if random.random() < self.mutation_probability:
                # normal distribution
                x = np.random.normal(0, self.mutation_deviation)

                self.chromosomes[i] += x

        self.weights_first_layer, self.weights_second_layer, self.weights_output_layer = self.chromosomes_to_weights(
            self.chromosomes)

        return

    def __save_as_chromosomes(self):
        first_flattened = np.matrix.flatten(self.weights_first_layer)
        second_flattened = np.matrix.flatten(self.weights_second_layer)
        output_flattened = np.matrix.flatten(self.weights_output_layer)

        all_arrays = np.concatenate([first_flattened, second_flattened, output_flattened])

        return all_arrays

    def chromosomes_to_weights(self, chromosomes: np.array):
        first_size = no_inputs * first_layer_size
        second_size = no_inputs * first_layer_size + first_layer_size * second_layer_size

        a = chromosomes[0: first_size].reshape(no_inputs, first_layer_size)
        b = chromosomes[first_size: second_size].reshape(first_layer_size, second_layer_size)
        c = chromosomes[second_size:].reshape(second_layer_size, no_outputs)

        return a, b, c

    def get_fitness(self):
        return self.fitness

    def random_positive_negative(self):
        return 1 if random.random() < 0.5 else -1

    def __repr__(self):
        return repr(self.fitness)

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness