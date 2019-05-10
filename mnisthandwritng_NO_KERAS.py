import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 

mnist = input_data.read_data_sets("mnist/", one_hot = True)

no_input = 784
no_hidden_one = 512
no_hidden_two = 1024
no_hidden_three = 256
no_output = 10

batch_size = 100

x = tf.placeholder("float", [None, no_input])
y = tf.placeholder("float", [None, no_output])

def neural_network(data):

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
    layerone = tf.add(tf.matmul(data, lay_one['weights']), lay_one['biases'])
    layerone = tf.nn.relu(layerone)

    layertwo = tf.add(tf.matmul(layerone, lay_two['weights']), lay_two['biases'])
    layertwo = tf.nn.relu(layertwo)
    
    layerthree = tf.add(tf.matmul(layertwo, lay_three['weights']), lay_three['biases'])
    layerthree = tf.nn.relu(layerthree)
    
    layer_out =  tf.matmul(layerthree, lay_out['weights']) + lay_out['biases']

    return layer_out

def train_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    max_epochs = 5

    with tf.Session() as sess: 
        sess.run(tf.initialize_all_variables())

        for epoch in range(1, max_epochs):
            epoch_loss = 0
            for _ in range(mnist.train.num_examples // batch_size):
                xl, yl = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: xl, y: yl})
                epoch_loss += c
            print('Epoch: ', epoch, '[', (epoch/max_epochs)*100, '% completed]')

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_network(x)
