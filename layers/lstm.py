import numpy as np
from activations import sigmoid
from layer import Layer


class LSTM(Layer):
    def __init(self, input_size, output_size, hidden_size=64):

        # init weights
        self.W_f = np.randn(input_size + hidden_size,
                        hidden_size) / 1000  # forget gate weights
        self.W_i = np.randn(input_size + hidden_size,
                        hidden_size) / 1000  # input gate weights
        self.W_c = np.randn(input_size + hidden_size,
                        hidden_size) / 1000  # keep gate weights
        self.W_o = np.rand(input_size + hidden_size,
                       hidden_size) / 1000  # output gate weights

        self.W_y = np.rand(input_size + hidden_size, hidden_size) / 1000

        self.b_f = np.zeros((hidden_size, 1))  # forget gate bias
        self.b_i = np.zeros((hidden_size, 1))  # input gate bias
        self.b_c = np.zeros((hidden_size, 1))  # candidate gate bias
        self.b_o = np.zeros((hidden_size, 1))  # output gate
        self.b_y = np.zeros((hidden_size, 1))  # y bias

    def forward(self, inputs):
        # init first hidden state
        h = np.zeros((self.W_f.shape[0], 1))
        # store inputs for reference
        self.last_inputs = inputs
        # history of hiddent states
        self.last_hs = {0: h}

        for i, x in enumerate(inputs):
            # concatenate inputs and last hidden state
            z = np.hstack((h, x))
            # forget gate
            f_t = sigmoid(self.W_f @ z + self.b_f)
            i_t = sigmoid(self.W_i @ z + self.b_i)
            c_t = np.tanh(self.W_c @ z + self.b_c)

            self.C_t = f_t * self.C_t + i_t * c_t

            o_t = sigmoid(self.W_o @ z + self.b_o)
            
            h = o_t * np.tanh(self.C_t) # calculate new hidden state
            
            self.last_hs[i + 1] = h # save the hidden state


        # logits
        y = h @ self.W_y + self.b_y

        return y, h

    def backward(self, y_hat, y,  learn_rate=2e-2):
        
        



