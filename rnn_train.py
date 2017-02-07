
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

start_with = False
cc_fips = 'EZ'
backward = True
path = "C:/Users/Antoine Didisheim/Dropbox/PyCharm/TensorFlowTutorials/saved_model/meta-model_"+cc_fips
test_path = 'test_'+cc_fips+'.p'
train_path = 'train_'+cc_fips+'.p'

if backward:
    path = "C:/Users/Antoine Didisheim/Dropbox/PyCharm/TensorFlowTutorials/saved_model/meta-model_backward_"+cc_fips
    test_path = 'test_backward_'+cc_fips+'.p'
    train_path = 'train_backward_'+cc_fips+'.p'

with open(test_path, 'rb') as file:
    data_test = pickle.load(file)

with open(train_path, 'rb') as file:
    data_train = pickle.load(file)

test_x = data_train[0]
print("x, ", test_x)
print("x sample, ", test_x[0])

test_y = data_train[1]



train_x = data_test[0]
train_y = data_test[1]



n_classes = 30
batch_size = 128
hm_epochs = 10


print(len(train_x[0]))
print(len(train_y[0]))


chunk_size = 31
n_chunk = 20
rnn_size = 128

test_x = np.array(test_x)
test_x = test_x.reshape(-1, n_chunk, chunk_size)


x = tf.placeholder(tf.float32, [None, n_chunk, chunk_size])
y = tf.placeholder(tf.float32, [None, len(train_y[0])])

layer = {'weight': tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'bias': tf.Variable(tf.random_normal([n_classes]))}
saver = tf.train.Saver()

def recurent_neural_network(x1):
    x1 = tf.transpose(x1, [1,0,2])
    x1 = tf.reshape(x1, [-1,chunk_size])
    x1 = tf.split(0,n_chunk,x1)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)

    output, states = rnn.rnn(lstm_cell, x1, dtype=tf.float32)

    output = tf.matmul(output[-1], layer['weight']) + layer['bias']

    return output


def train_neural_network(x):
    prediction = recurent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:

        if start_with:
            print("Load wip")
            saver_in_between = tf.train.Saver()
            saver_in_between.restore(sess, path)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
        else:
            sess.run(tf.global_variables_initializer())
            print("Start over")


        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):

                start = i
                end = i + batch_size
                if end<len(train_x):
                    batch_x = np.array(train_x[start:end])
                    batch_y = np.array(train_y[start:end])
                    batch_x = batch_x.reshape((batch_size, n_chunk, chunk_size))
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c
                i += batch_size


            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
            print()

            save_full_graph(sess=sess,
                            path2=path)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Final accuracy:', accuracy.eval({x: test_x, y: test_y}))
        #saver.save(sess, "C:/Users/Antoine Didisheim/Dropbox/PyCharm/TensorFlowTutorials/saved_model/model.ckpt")
        save_full_graph(sess=sess,
                        path2=path)


def save_full_graph(sess, path2):
  with sess.graph.as_default():
    saver2 = tf.train.Saver()
    saver2.save(sess, path2, meta_graph_suffix='meta', write_meta_graph=True)


train_neural_network(x)
#load_neural_network()