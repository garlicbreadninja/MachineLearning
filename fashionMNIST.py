from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(xtrain, ytrain), (xtest, ytest) = data.load_data()

xtrain = keras.utils.normalize(xtrain)
xtest = keras.utils.normalize(xtest)

classes = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneakers', 'Bag', 'Ankle boots']

model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(xtrain, ytrain, epochs=5)

val_loss, val_acc = model.evaluate(xtest, ytest)

print("Accuracy: {}%".format(val_acc*100))

model.save('fashion_mnist_nn.model')

