import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - x * x
