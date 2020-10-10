import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_prime(x):
    jacobian_m = np.ones((len(x), len(x)))

    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                jacobian_m[i][j] = x[i] * (1 - x[j])
            else:
                jacobian_m[i][j] = -x[i] * x[j]

    return jacobian_m

x = [0.4975, 0.5024]

soft = softmax_prime(x)
print(soft)
# print(softmax_prime(soft)) 