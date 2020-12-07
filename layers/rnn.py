# Implementation of Basic RNN layer
import numpy as np
from layer import Layer
from numpy.random import randn


class RNN(Layer):
    def __init__(self, input_size, output_size, hidden_size=64):
        # init weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward_propagation(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        y = self.Why @ h + self.by

        return y, h

    def backward_propagation(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)

        # Calculate dL/dWhy and dL/dby
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # init dL/dWhh, dL/dWxh, and dL/dbh to zero vects
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # calculate dL/dh for last h
        d_h = self.Why.T @ d_y

        # backprop
        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1]**2) * d_h)

            d_bh += temp

            d_Whh += temp @ self.last_hs[t].T

            d_Wxh += temp @ self.last_inputs[t].T

            d_h = self.Whh @ temp

        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # update weights and biases using gradient descent
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by
