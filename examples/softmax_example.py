import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_prime(x):
    jacobian_m = np.diag(x)

    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                jacobian_m[i][j] = x[i] * (1 - x[j])
            else:
                jacobian_m[i][j] = -x[i] * x[j]

    return jacobian_m

x = [1.0, 2.0, 3.0, 4.0]

soft = softmax(x)
print(soft)
print(softmax_prime(soft)) 