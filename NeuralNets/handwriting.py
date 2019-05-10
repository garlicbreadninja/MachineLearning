import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = tf.keras.datasets.mnist
(xtrain, ytrain), (xtest, ytest) = data.load_data()

xtrain = tf.keras.utils.normalize(xtrain, axis=1)
xtest = tf.keras.utils.normalize(xtest, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=5)

valloss, valacc = model.evaluate(xtest, ytest)
print(valloss, valacc)

model.save('handwriting.model', include_optimizer=False)
