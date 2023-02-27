from keras.datasets import mnist
from random import uniform
from math import e
from matplotlib import pyplot
import numpy as np


class Layer:

    def __init__(self, num_neurons, activation_func=None):

        self.num_neurons = num_neurons
        self.activations = np.ndarray((self.num_neurons, 1))
        self.activation_func = self.set_activation_func(activation_func)
        self.layer_type = 'hidden'
        self.weight_matrix = None
        self.biases = None
        self.z = None
        self.in_activations = None

    def init_vectors(self, prev_neurons):
        self.weight_matrix = np.ndarray((self.num_neurons, prev_neurons))
        self.biases = np.ndarray((self.num_neurons, 1))

        for row_index in range(self.num_neurons):
            for col_index in range(prev_neurons):
                self.weight_matrix[row_index][col_index] = uniform(-10, 10)

        for index in range(self.num_neurons):
            self.biases[index] = uniform(1, 5)

    def set_layer_type(self, layer_type):
        self.layer_type = layer_type

    def set_activation_func(self, activation_func):

        if activation_func == 'sigmoid':
            return lambda z: 1 / (1 + np.exp(-z))

        elif activation_func == 'softmax':
            return lambda z, i: (e ** z[i]) / np.sum(e ** z)

        elif activation_func == 'relu':
            return lambda z: z if(z > 0) else 0

        else:
            return lambda z: z

    def set_activations(self, in_activations):

        self.in_activations = in_activations

        if self.layer_type == 'input':
            for index in range(self.num_neurons):
                self.activations[index] = self.activation_func(
                                                         in_activations[index])

        else:
            self.z = self.weight_matrix.dot(in_activations) + self.biases
            self.activations = self.activation_func(self.z)

    def get_num_neurons(self):

        return self.num_neurons

    def get_activations(self):

        return self.activations

    def display_activations(self):

        for activation in self.activations:
            print(activation)
        print(f'The model predicts this is a {self.activations.argmax()}')

    def get_weight_matrix(self):

        return self.weight_matrix

    def get_biases(self):

        return self.biases


class Network:

    def __init__(self, layers=None):

        self.layers = layers
        self.num_layers = len(layers)
        self.set_layer_types()
        self.weight_error = np.ndarray((self.num_layers,), dtype=object)
        self.activation_error = np.ndarray((self.num_layers,), dtype=object)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            mnist.load_data()
        self.minibatch_size = 100

        self.init_layers()

    def init_layers(self):

        for index in range(len(self.layers) - 1):
            self.layers[index + 1].init_vectors(
                                          self.layers[index].get_num_neurons())

    def set_layer_types(self):

        self.layers[0].set_layer_type('input')
        self.layers[self.num_layers - 1].set_layer_type('output')

    def train(self):

        self.y_train = self.convert_y()
        counter = 0

        while counter < self.y_train.shape[0] / 5:
            print(f'Training with minibatch: {counter / self.minibatch_size}')
            self.backprop(counter)
            counter += self.minibatch_size

    def init_errors(self):

        for index in range(self.num_layers - 1):
            self.weight_error[index + 1] = np.zeros(self.layers[index + 1].
                                                    weight_matrix.shape,
                                                    dtype=float)
            self.activation_error[index + 1] = np.zeros(self.layers[index + 1].
                                                        activations.shape,
                                                        dtype=float)

    def backprop(self, counter):

        ending = counter + self.minibatch_size

        self.init_errors()

        while ending > counter:
            self.input_image(self.x_train[counter].flatten())
            self.backprop_helper(self.num_layers - 1, self.y_train[counter])
            counter += 1

        for index in range(self.num_layers - 1):
            self.layers[index + 1].weight_matrix -= \
                                                (self.weight_error[index + 1] /
                                                 self.minibatch_size)
            self.layers[index + 1].biases -= \
                (self.activation_error[index + 1] /
                 self.minibatch_size)

    def backprop_helper(self, index, y):
        if not self.layers[index].layer_type == 'input':
            if self.layers[index].layer_type == 'output':
                self.activation_error[index] += \
                                ((self.layers[index].activations - y) *
                                 (self.layers[index].activation_func
                                 (self.layers[index].z) *
                                 (1 - self.layers[index].activation_func(
                                  self.layers[index].z))))
            else:
                self.activation_error[index] += \
                        (self.layers[index + 1].weight_matrix.T.dot
                         (self.activation_error[index + 1]) *
                         (self.layers[index].activation_func
                         (self.layers[index].z) *
                         (1 - self.layers[index].activation_func(
                          self.layers[index].z))))

            for row_index in range(self.weight_error[index].shape[0]):
                for col_index in range(self.weight_error[index].shape[1]):
                    self.weight_error[index][row_index][col_index] += \
                                         (self.activation_error
                                          [index][row_index] *
                                          self.layers[index - 1].in_activations
                                          [col_index])

            return self.backprop_helper(index - 1, y)

        else:
            return

    def convert_y(self):

        return_arr = np.ndarray((self.y_train.shape[0], 10, 1))

        for index in range(self.y_train.shape[0]):
            temp_arr = np.zeros((10, 1), dtype=float)
            temp_arr[self.y_train[index]][0] = 1.0
            return_arr[index] = temp_arr

        return return_arr

    def input_image(self, in_data):

        for index in range(len(self.layers)):
            if index == 0:
                self.layers[0].set_activations(in_data)
            else:
                self.layers[index].set_activations(self.layers[index - 1].
                                                   get_activations())
        return self.layers[self.num_layers - 1].get_activations()

    def display_output(self):
        self.layers[self.num_layers - 1].display_activations()


def main():
    nn = Network(layers=[Layer(784, 'sigmoid'),
                 Layer(40, 'sigmoid'),
                 Layer(20, 'sigmoid'),
                 Layer(10, 'sigmoid')])

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    correct = 0
    nn.train()
    pyplot.subplot(330 + 1)
    pyplot.imshow(x_train[37], cmap=pyplot.get_cmap('gray'))
    pyplot.show()

    for index in range(x_test.shape[0]):
        results = nn.input_image(x_test[index].flatten())
        if results.argmax() == y_test[index]:
            correct += 1

    print(f"The model got {correct} right answers, out of {y_test.shape[0]}.")


if __name__ == "__main__":
    main()
