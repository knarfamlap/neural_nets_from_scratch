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

    def forward(self, inputs):
        '''
        Forward pass for vanilla RNN given the inputs. 
        Returns the final output and its hidden state.
        - inputs must be an array where each column is a one-hot vector. 
            dimensions of each column are (input_size, 1). 
        '''
        # init the first hidden state
        h = np.zeros((self.Whh.shape[0], 1))
        # store the curr input so they can be referenced in future
        # time steps
        self.last_inputs = inputs
        # store the current hidden state for future reference
        self.last_hs = {0: h}
        # iterate through one-hot vectors
        for i, x in enumerate(inputs):
            # update hidden state
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            # save hidden state
            self.last_hs[i + 1] = h
        # calculate the output
        y = self.Why @ h + self.by
        # return the output and the last hidden state
        return y, h

    def backward(self, d_y, learn_rate=2e-2):
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
