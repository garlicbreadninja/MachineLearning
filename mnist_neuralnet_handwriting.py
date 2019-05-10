# pretty self-explanatory, imports.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

#normalizing the data to make our NN run faster.
xtrain = tf.keras.utils.normalize(xtrain, axis=1)
xtest = tf.keras.utils.normalize(xtest, axis=1)

# defining our model, we've got a sequential NN, with a Flatten() for the input layer.
# Flatten() squishes the 28 x 28 2D image into a large array with 784 piel values.
# The handwriting images go in like this, lineraly into the NN.
# We then add 3 hidden layers, each with 1024, 512 and 128 nodes respectively.
# We pass each of the layers through ReLU to decide which neurons will fire
# The last layer has 10 nodes, we later use numpy's argmax() function to check for correction predctions, since we're using one hot
# We have a softmax - sigmoid function, for the activation for the last one because it's a probability of the image being a number from the label.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# We compile the model with an Adam optimizer and a categorical cross entropy loss function
# We also use the metrics argument to let it know that we want to track the model's accuracy
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit() trains the model with the data we give.
# we're running it for 5 epochs which does the job within a reasonable amout of time [on my laptop].
model.fit(xtrain, ytrain, epochs=5)

# we calculate the test set's accuracy and loss.
val_loss, val_acc = model.evaluate(xtest, ytest)

#we print the accuracy
print("Accuracy: ", val_acc*100, "%")

#saving the model for later
model.save('handwriting_mnist_nn.model')
