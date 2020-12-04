import numpy as np
from layer import Layer


class Conv2D(Layer):
    def __init__(self, filters):
        self.filters = np.random.randn(filters, 3, 3) / 9

    def iterate_regions(self, img):
        h, w = img.shape

        for i in range(h - 2):
            for j in range(w - 2):
                img_region = img[i:(i + 3), j:(j + 3)]
                yield img_region, i, j

    def forward_propagation(self, input):
        self.last_input = input
        h, w = input.shape

        output = np.zeros((h - 2, w - 2, self.filter.shape[0]))

        for img_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(img_region * self.filters, axis=(1, 2))

        return output

    def backward_propagation(self, output_error, learning_rate):
        grad_filters = np.zeros(self.filters.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.filters.shape[0]):
                grad_filters[f] += output_error[i, j, f] * img_region

        self.filters -= learning_rate * grad_filters

        return grad_filters
