from .layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation="relu"):
        if activation == "relu":
            self.activation = self.tanh
            self.activation_prime = self.tanh_prime
        else:
            raise ValueError("Only supported activations are: relu")

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