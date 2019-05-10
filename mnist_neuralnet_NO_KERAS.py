# auth: Anurag Akella / github.com/garlicbreadninja
# This is a Neural Net I wrote when I first started learning about Neural Nets. It doesn't use Keras.
# Although this doesn't look as simple as the Keras model, it stil works. I still have to figure out a way to save and reuse the model.
# Imports, self-explanatory
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 

# We load the MNIST data, with ONE_HOT ENABLED. !IMPORTANT
mnist = input_data.read_data_sets("mnist/", one_hot = True)

# We define the variables that hold the number nodes that each layer is going to have.
# For the input, we have a 28 x 28 image collapsed into a 1D 784 indices long array
# the hidden layers have 512, 1024, 256 arrays respectively.
# the output, since we're using one_hot has 10 nodes, each node corresponding to the class [0-9].
# We'll use argmax later to get a definitive output, instead of just probabilities.
no_input = 784
no_hidden_one = 512
no_hidden_two = 1024
no_hidden_three = 256
no_output = 10

# We can't train the whole dataset in one go, so we divide it into batches.
batch_size = 100

# We define placeholders for the inputs and outputs. 'float'.
x = tf.placeholder("float", [None, no_input])
y = tf.placeholder("float", [None, no_output])

# this function is the 'brains' of our Network, quite literally.
# We give random values to the weights and biases of each hidden layers, since we have to start somewhere to converge on some minima.
def neural_network(data):
    # Defining weights, biases
    lay_one = {
        'weights': tf.Variable(tf.random_normal([no_input, no_hidden_one])),
        'biases' : tf.Variable(tf.random_normal([no_hidden_one]))
    }
    lay_two = {
        'weights': tf.Variable(tf.random_normal([no_hidden_one, no_hidden_two])),
        'biases' : tf.Variable(tf.random_normal([no_hidden_two]))
    }
    lay_three = {
        'weights': tf.Variable(tf.random_normal([no_hidden_two, no_hidden_three])),
        'biases' : tf.Variable(tf.random_normal([no_hidden_three]))
    }
    lay_out = {
        'weights': tf.Variable(tf.random_normal([no_hidden_three, no_output])),
        'biases' : tf.Variable(tf.random_normal([no_output]))
    }
    # Using TensorFlow's matmul, add to multiply the weights of each layer and then add a bias to each node.
    layerone = tf.add(tf.matmul(data, lay_one['weights']), lay_one['biases'])
    layerone = tf.nn.relu(layerone)

    # we then pass each layer through an activation function.
    layertwo = tf.add(tf.matmul(layerone, lay_two['weights']), lay_two['biases'])
    layertwo = tf.nn.relu(layertwo)
    
    layerthree = tf.add(tf.matmul(layertwo, lay_three['weights']), lay_three['biases'])
    layerthree = tf.nn.relu(layerthree)
    
    #final layer that we'll return to where ever we use this function.
    layer_out =  tf.matmul(layerthree, lay_out['weights']) + lay_out['biases']

    return layer_out

def train_network(x):
    # the initial precition is stored here (SPOILER: it's probably pretty bad and garbage)
    prediction = neural_network(x)

    # We define out cost function, softmax cross entropy = summation of logs .. blah blah
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    # optimizer = we'll use Adam and tell him what we're going to minimize.
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Define the total number of epochs
    max_epochs = 5

    # We run a Session() and train the model. The session will close when we're done training.
    with tf.Session() as sess: 
        # initialize all tensorflow vars
        sess.run(tf.initialize_all_variables())

        #loop that runs through each epoch
        for epoch in range(1, max_epochs):
            epoch_loss = 0
            for _ in range(mnist.train.num_examples // batch_size):
                xl, yl = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: xl, y: yl})
                epoch_loss += c
            print('Epoch: ', epoch, '[', (epoch/max_epochs)*100, '% completed]')

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            # Evalutating, After training.
            print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_network(x)
