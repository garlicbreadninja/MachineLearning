from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
x = 6
data = keras.datasets.fashion_mnist

(_x, _y), (xtest, ytest) = data.load_data()

xtest = keras.utils.normalize(xtest)

classes = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneakers', 'Bag', 'Ankle boots']

modelfashion = keras.models.load_model('fashion_mnist_nn.model')

preds = modelfashion.predict(xtest)
x = int(input("Enter an array index: "))
print(classes[np.argmax(preds[x])])
plt.imshow(xtest[x])
plt.show()