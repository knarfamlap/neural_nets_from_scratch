import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import Model
from layers.dense import Dense
from layers.activation import Activation

import numpy as np

X_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model = Model()

model.add(Dense(2, 3))
model.add(Activation())
model.add(Dense(3, 1))
model.add(Activation())

model.fit(X_train, y_train, epochs=1000, learning_rate=0.1)

pred = model.predict(X_train)

print(pred)

