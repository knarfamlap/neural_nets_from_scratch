import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from layers.activation import Activation
from layers.dense import Dense
from model.model import Model



X_train = np.array([
    [[-2, -1]],
    [[25, 6]],
    [[17, 4]],
    [[-15, -6]]
])
y_train = np.array([
    [[1]],  # F
    [[0]],  # M
    [[0]],  # M
    [[1]]  # F
])

model = Model()

model.add(Dense(2, 2))
model.add(Activation("sigmoid"))
model.add(Dense(2, 1))
model.add(Activation("sigmoid"))
model.add(Activation("softmax"))

model.fit(X_train, y_train, epochs=1000, learning_rate=0.01)

frank = np.array([20,  2])  # 155 pounts, 68 inches

pred = model.predict(frank)

print(pred)
