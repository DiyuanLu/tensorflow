#excercise of tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data       #get mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import matplotlib as plt
import ipdb



pixel = 28   #the pixel your picture has
#X = tf.placeholder(tf.float32, [None, pixel, pixel, 1])   #here 1 is for the chanel, since it's gray scale so it's 1
X = tf.placeholder(tf.float32, [None, pixel*pixel])   #here 1 is for the chanel, since it's gray scale so it's 1
W = tf.Variable(tf.zeros([pixel*pixel, 10]))
b = tf.Variable(tf.zeros([10]))

#init = tf.initialize_all_variables()       #old version
init =tf.global_variables_initializer()             #new version

#model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)        #possibility vector
Y_pred_label = tf.argmax(Y, 1)                                      #winner takes all

#placeholder for true labels
Y_ = tf.placeholder(tf.float32, [None, 10])

#lost function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))     #without phd

#accuracy of correct answers found in this traning batch
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))   #tf.armax=winner takes all--one hot vector
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)   #compute the derivatives regarding W_ij and b_ij
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(1000):
    #load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)  #everytime get 100 pictures, reasons:1. GPU running time 2.get derivitives of 10 figs, have a global sense of the data and where to go
    train_data = {X: batch_X, Y_: batch_Y}

    #train
    sess.run(train_step, feed_dict=train_data)

    #
    #a, c = sess.run([accuracy, cross_entropy], feed=train_data)

    #Tips: do the following every 100 or 1000 steps, not every step
    if i%100 ==0:#success
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

        #success on test data
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        #train_accuracy = accuracy.eval(feed_dict={X: batch_X, Y_: batch_Y})
        print("step %d, accuracy=%r"%(i, sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels})))





'''
#New activation function
Y = tf.nn.relu(tf.matmul(X, W) + B)

#Dropout
pkeep = tf.placeholder(tf.float32)

Yf = tf.nn.relu(tf.matmul(X, W)+B)
Y = tf.nn.dropout(Yf, pkeep)
'''
