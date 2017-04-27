import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
'''
input> weighted > hidden l1 (activation function)>weighted
> hiddenl2 (activation function) > weight > output layer

compare output to intednded output > cost function(cross entropy)

optimizer > minimize the cost(AdamOptimizer...SGD, AdaGrad, ...)

backpropagation

feed forward + backprop = epoch
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100#one time deal with 100 picture as the same time

#height * width
x = tf.placeholder('float', [None, 784])   #flatten the picture
y = tf.placeholder('float')

def NN(data):
    hiddenl1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hiddenl2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hiddenl3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    outputlayer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hiddenl1['weights']), hiddenl1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hiddenl2['weights']), hiddenl2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hiddenl3['weights']), hiddenl3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, outputlayer['weights']) + outputlayer['biases']

    return output


def trainNN(x):
    prediction = NN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #cycles feed forward + backprop
    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c

            print("Epoch:", epoch, "completed out of", epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))



trainNN(x)

'''
('Epoch:', 0, 'completed out of', 10, 'loss:', 1917707.0849761963)
('Epoch:', 1, 'completed out of', 10, 'loss:', 436189.25308990479)
('Epoch:', 2, 'completed out of', 10, 'loss:', 241173.25928497314)
('Epoch:', 3, 'completed out of', 10, 'loss:', 141731.01331560386)
('Epoch:', 4, 'completed out of', 10, 'loss:', 88071.080716371536)
('Epoch:', 5, 'completed out of', 10, 'loss:', 53265.387680763786)
('Epoch:', 6, 'completed out of', 10, 'loss:', 36800.063167385117)
('Epoch:', 7, 'completed out of', 10, 'loss:', 27023.829311112015)
('Epoch:', 8, 'completed out of', 10, 'loss:', 23522.421069383621)
('Epoch:', 9, 'completed out of', 10, 'loss:', 22025.893316861613)
('Accuracy', 0.95020002)

'''

