import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from layers.activation import Activation
from layers.dense import Dense
from model.model import Model



X_train = np.array([
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[1, 1]]
])
y_train = np.array([
    [[0]],  # F
    [[1]],  # M
    [[1]],  # M
    [[0]]  # F
])

model = Model()

model.add(Dense(2, 3))
model.add(Activation('tanh'))
model.add(Dense(3, 2)) 
model.add(Activation('tanh'))
model.add(Activation('softmax'))


model.fit(X_train, y_train, epochs=1000, learning_rate=0.1)

frank = np.array(X_train)  # 155 pounts, 68 inches

pred = model.predict(frank)

print(np.array(pred))
