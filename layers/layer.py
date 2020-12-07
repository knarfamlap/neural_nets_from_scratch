class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes output of Y of a layer for give input X
    def forward(self, input):
        raise NotImplementedError

    # dE/dx for given dE/dY (updates params if any)
    def backward(self, output_error, learning_rate):
        raise NotImplementedError
