# auth: Anurag Akella / github.com/garlicbreadninja
# Testing the model I saved in the training .py file.
from tensorflow import keras as k
import numpy as np
import matplotlib.pyplot as plt
x = 8
mnist = k.datasets.mnist

(_x, _y), (xtest, ytest) = mnist.load_data()

model = k.models.load_model('handwriting_mnist_nn.model')
predicitons = model.predict(xtest)
x = int(input('Enter an array index: '))
print(np.argmax(predicitons[x]))
plt.imshow(xtest[x])
plt.show()
