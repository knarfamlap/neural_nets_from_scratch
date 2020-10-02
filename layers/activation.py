from .layer import Layer
import numpy as np


class Activation(Layer):
    def __init__(self, activation="tanh"):
        if activation == "tanh":
            self.activation = self.tanh
            self.activation_prime = self.tanh_prime
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_prime = self.sigmoid_prime
        elif activation == 'relu':
            self.activation = self.relu
            self.activation_prime = self.relu_prime
        else:
            raise ValueError(
                "Only supported activations are: tanh, sigmoid, and relu")

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return np.heaviside(x, 1)
