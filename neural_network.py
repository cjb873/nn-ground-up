from keras.datasets import mnist
from random import uniform, randrange
import matplotlib.pyplot as plt
from math import e
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
                self.weight_matrix[row_index][col_index] = uniform(-1, 1)

        for index in range(self.num_neurons):
            self.biases[index] = uniform(1, 5)

    def set_layer_type(self, layer_type):
        self.layer_type = layer_type

    def set_activation_func(self, activation_func):

        if activation_func == 'sigmoid':
            return lambda z: 1.0 / (1.0 + np.exp(-z))

        elif activation_func == 'softmax':
            return lambda z, i: (e ** z[i]) / np.sum(e ** z)

        elif activation_func == 'relu':
            return lambda z: z if(z > 0) else 0

        else:
            return lambda z: z

    def set_activations(self, in_activations, index):

        self.in_activations = in_activations
        if index == 0:
            for act_index in range(self.num_neurons):
                self.activations[act_index] = in_activations[act_index] / 255

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
        self.avg_loss = 0
        self.training_size = 0

        self.init_layers()

    def get_training_size(self):
        return self.training_size

    def set_training_size(self, new_size):
        self.training_size = new_size

    def init_layers(self):

        for index in range(1, self.num_layers):
            self.layers[index].init_vectors(
                                      self.layers[index - 1].get_num_neurons())

    def set_layer_types(self):

        self.layers[0].set_layer_type('input')
        self.layers[self.num_layers - 1].set_layer_type('output')

    def train(self):

        self.y_train = self.convert_y()

        for i in range(0, self.training_size):
            self.backprop(i)

    def init_errors(self):
        for index in range(1, self.num_layers):
            self.weight_error[index] = np.zeros(self.layers[index].
                                                weight_matrix.shape,
                                                dtype=float)
            self.activation_error[index] = np.zeros(self.layers[index].
                                                    activations.shape,
                                                    dtype=float)

    def backprop(self, counter):

        self.predict(self.x_train[counter].flatten())
        self.init_errors()
        y = self.y_train[counter]
        last = self.num_layers - 1
        z = self.layers[last].z
        activations = self.layers[last].activations
        activation_func = self.layers[last].activation_func
        derivative_sigmoid = (lambda z: activation_func(z) *
                              (1 - activation_func(z)))

        self.avg_loss += np.sum(np.square(y - activations))
        self.activation_error[last] = np.multiply((activations - y),
                                                  derivative_sigmoid(z))

        for index in range(self.num_layers - 2, 0, -1):
            activations = self.layers[index].activations
            activation_func = self.layers[index].activation_func
            z = self.layers[index].z
            self.activation_error[index] = np.multiply(self.layers[index + 1].
                                                       weight_matrix.T.dot
                                                       (self.activation_error
                                                       [index + 1]),
                                                       derivative_sigmoid(z))

        for index in range(1, self.num_layers):
            for row_index in range(self.weight_error[index].shape[0]):
                for col_index in range(self.weight_error[index].shape[1]):
                    self.weight_error[index][row_index][col_index] = \
                                          (self.activation_error
                                           [index][row_index] *
                                           self.layers[index].in_activations
                                           [col_index])

        for index in range(1, self.num_layers):
            self.layers[index].weight_matrix = \
                                              (self.layers[index].weight_matrix
                                               - self.weight_error[index])
            self.layers[index].biases = (self.layers[index].biases -
                                         self.activation_error[index])

        if counter % 1000 == 0 and counter != 0:
            print(f'Loss: {self.avg_loss / 1000}')
            self.avg_loss = 0

    def convert_y(self):

        return_arr = np.ndarray((self.y_train.shape[0], 10, 1))

        for index in range(self.y_train.shape[0]):
            temp_arr = np.zeros((10, 1), dtype=float)
            temp_arr[self.y_train[index]][0] = 1.0
            return_arr[index] = temp_arr

        return return_arr

    def predict(self, in_data):

        for index in range(self.num_layers):
            if index == 0:
                self.layers[0].set_activations(in_data, index)
            else:
                self.layers[index].set_activations(self.layers[index - 1].
                                                   get_activations(), index)
        return self.layers[self.num_layers - 1].get_activations()

    def display_output(self):
        self.layers[self.num_layers - 1].display_activations()


def main():
    nn = Network(layers=[Layer(784, 'sigmoid'),
                 Layer(16, 'sigmoid'),
                 Layer(16, 'sigmoid'),
                 Layer(10, 'sigmoid')])

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    correct = 0
    nn.training_size = int(input("Num images to train on: "))
    nn.train()

    for index in range(x_test.shape[0]):
        results = nn.input_image(x_test[index].flatten())
        if results.argmax() == y_test[index]:
            correct += 1

    print(f"When trained with {nn.training_size} examples, "
          f"the model's accuracy is {correct / y_test.shape[0]}.")
    test_img = randrange(y_test.shape[0])
    prediction = nn.input_image(x_test[test_img].flatten()).argmax()
    print(f"The model thinks image #{test_img} is a {prediction}."
          f" The label is {y_test[test_img]}.")
    plt.imshow(x_test[test_img], cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == "__main__":
    main()
