# Implementation of Basic RNN layer
import numpy as np
from layer import Layer


class RNN(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def forward_propagation(self, x, prev_s, U, W, V):
        self.u = np.dot(W, x)
        self.w = np.dot(W, prev_s)

        self.add = self.u + self.w

        self.v = np.dot(V, self.add)

    def backward_propagation(self):
        pass
