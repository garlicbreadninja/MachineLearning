# pretty self-explanatory, imports.
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# importing the mnist dataset into 'data'
data = keras.datasets.fashion_mnist

# splitting the dataset into train and test variables
(xtrain, ytrain), (xtest, ytest) = data.load_data()

# normalizing the x components to help the NN run faster (smaller values)
xtrain = keras.utils.normalize(xtrain)
xtest = keras.utils.normalize(xtest)

# defining the classes, we'll use these to print an output based on the one_hot predition from the NN
classes = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneakers', 'Bag', 'Ankle boots']

# defining our model, we've got a sequential NN, with a Flatten() for the input layer.
# Flatten squishes the 28 x 28 image into a 784 index 1D array
# We then add 3 Dense() hidden layers, using add() -> each with 1024, 512 and 128 nodes respectively.
# We pass each of the layers through a ReLU activation 
# The last layer is our output layer
# it had 10 nodes, we later use numpy's argmax() function to check for correction predctions, since we're running it in a 'one_hot' format
# We have a softmax - sigmoid function, for the activation for the last one because it's a probability of the image being related to that class so it would be a number between [0,1].
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# We compile the model with an Adam optimizer and a categorical cross entropy loss function
# We also use the metrics argument to let it know that we want to track the model's accuracy.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Finally, we fit the model using the fit() funtion, out xtrain and ytrain splits and we specify the number of life_cycles for the training(epochs).
# I've given it 5 epochs here, becuasse it runs fast and gets the job done. Too many will take too long.
model.fit(xtrain, ytrain, epochs=5)

# We calculate the validation loss and accuracy using evaluate() and out test splits for x, y.
# The accuracy for this model with 5 epochs was about 97%.
# Meaning it does good wiht generalized data, data that is not a part of the training set.
val_loss, val_acc = model.evaluate(xtest, ytest)

# we print the Accuracy
print("Accuracy: {}%".format(val_acc*100))

# I've saved the model. We can later import it in another project through keras.models.load_model() and work with it
# It will save the same weights for the NN after we've trained it. So the accuracy should be similar for other data too.
model.save('fashion_mnist_nn.model')

# We're using TensorFlow and Keras here. They're pretty high level and we use simple English like statements to get the job done.

