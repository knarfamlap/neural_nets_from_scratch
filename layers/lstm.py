import numpy as np
from activations import sigmoid
from layer import Layer


class LSTM(Layer):
    def __init(self, input_size, output_size, hidden_size=64):

        # init weights
        self.Wf = randn(input_size + hidden_size,
                        hidden_size)  # forget gate weights
        self.Wi = randn(input_size + hidden_size,
                        hidden_size)  # input gate weights
        self.Wc = randn(input_size + hidden_size,
                        hidden_size)  # keep gate weights
        self.Wo = rand(input_size + hidden_size,
                       hidden_size)  # output gate weights

        self.bf = np.zeros((hidden_size, 1))  # forget gate bias
        self.bi = np.zeros((hidden_size, 1))  # input gate bias
        self.bc = np.zeros((hidden_size, 1))  # candidate gate bias
        self.bo = np.zeros((hidden_size, 1))  # output gate

    def forward(self, inputs):
        # init first hidden state
        h = np.zeros((self.Wf.shape[0], 1))
        # store inputs for reference
        self.last_inputs = inputs
        # history of hiddent states
        self.last_hs = {0: h}

        for i, x in enumerate(inputs):
            # concatenate inputs and last hidden state
            z = np.hstack(h, x))
            # forget gate
            f_t = sigmoid(self.Wf @ z + bf)
            i_t = sigmoid(self.Wi @ z + bi)
            c_t = np.tanh(self.Wc @ z + bc)

            self.Ct = f_t @ self.Ct + i_t @ c_t

            o_t = sigmoid(self.Wo @ z + bo)

            h = o_t @ np.tanh(self.Ct)
        
        return h

    def backward(self):
        pass
