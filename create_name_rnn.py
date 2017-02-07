
import tensorflow as tf
import numpy as np
import featuresFunction as ff
import features_function_backward as ff_backward
from tensorflow.python.ops import rnn, rnn_cell
import copy
import random
import csv
import bz2
import base64
import pickle

ref_vector = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
              "v", "w", "x", "y", "z", "-", "'", " ", "NOTHING"]

n_classes = 30
batch_size = 128
hm_epochs = 10

chunk_size = 31
n_chunk = 20
rnn_size = 128

x = tf.placeholder(tf.float32, [None, n_chunk, chunk_size])
y = tf.placeholder(tf.float32, [None, 30])

layer = {'weight': tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'bias': tf.Variable(tf.random_normal([n_classes]))}

x = tf.placeholder(tf.float32, [None, n_chunk, chunk_size])
def neural_network_model(x):

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1,chunk_size])
    x = tf.split(0,n_chunk,x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)

    output, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(output[-1], layer['weight']) + layer['bias']

    return output




def create_percent_list(abs_list, number_to_choose):
    pos_list = np.maximum(abs_list,0)
    percent_list = pos_list / np.sum(pos_list)
    non_zero_list = percent_list[np.nonzero(percent_list)]
    sort_list = copy.copy(percent_list)
    sort_list.sort()
    threshold = sort_list[-number_to_choose]
    cleaned_list = percent_list*(percent_list >= threshold)
    cleaned_list = cleaned_list / np.sum(cleaned_list)
    return cleaned_list, percent_list



def forward_generation(prediction, sess, generated_name):
    saver = tf.train.Saver()
    saver.restore(sess, path)
    likelihood_vector = []
    for i in range(0, len(generated_name)):
        _, input_vector_inside, output_vector = ff.createsFeaturesForLetter(generated_name, i)
        output_index = output_vector.index(1)
        features = np.array(input_vector_inside)
        features = features.reshape(-1, n_chunk, chunk_size)
        eval = prediction.eval(feed_dict={x: features})[0]
        cleaned_list, percent_list = create_percent_list(eval, number_choice_after)
        likelihood = percent_list[output_index]
        likelihood_vector.append(likelihood)
        #print(ref_vector[output_index],likelihood)
        if likelihood <= percent_likelihood:
            letter = np.random.choice(ref_vector, 1, p=cleaned_list)[0]
            new = list(generated_name)
            new[i] = letter
            generated_name = ''
            for l in new:
                generated_name = generated_name + l
    likelihood_vector = np.array(likelihood_vector)
    return generated_name, np.mean(likelihood_vector)


def backward_generation(prediction,sess, generated_name):

    saver = tf.train.Saver()
    saver.restore(sess, path_backward)
    i = len(generated_name)-1
    likelihood_vector = []
    while i >0:
        _, input_vector_inside, output_index = ff_backward.createsFeaturesForLetter(generated_name, i)
        output_index = output_index.index(1)
        features = np.array(input_vector_inside)
        features = features.reshape(-1, n_chunk, chunk_size)
        eval = prediction.eval(feed_dict={x: features})[0]
        cleaned_list, percent_list = create_percent_list(eval, number_choice_after)
        likelihood = percent_list[output_index]
        likelihood_vector.append(likelihood)
        #print(ref_vector[output_index],likelihood)
        if likelihood <= percent_likelihood:
            letter = np.random.choice(ref_vector, 1, p=cleaned_list)[0]
            new = list(generated_name)
            new[i] = letter
            generated_name = ''
            for l in new:
                generated_name = generated_name + l
        i = i -1
    likelihood_vector = np.array(likelihood_vector)
    return generated_name, np.mean(likelihood_vector)


def use_neural_network(input_data,length_name,creation_batch_size, prediction):
    with tf.Session() as sess:
        saver = tf.train.Saver()

        saver.restore(sess, path)
        features=np.array(input_data)
        features = features.reshape(-1, n_chunk, chunk_size)
        #result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: features}), 1)))
        eval = prediction.eval(feed_dict={x: features})[0]
        cleaned_list, percent_list = create_percent_list(eval, number_choice)
        cleaned_list = cleaned_list+np.max(cleaned_list)
        cleaned_list[26:] = 0
        cleaned_list = cleaned_list/np.sum(cleaned_list)

        letter = np.random.choice(ref_vector, 1, p=cleaned_list)
        generated_name = letter[0]

        for i in range(0,length_name):
            _, input_vector_inside, output_vector = ff.createsFeaturesForLetter(generated_name, i)
            features = np.array(input_vector_inside)
            features = features.reshape(-1, n_chunk, chunk_size)
            eval = prediction.eval(feed_dict={x: features})[0]
            cleaned_list, percent_list = create_percent_list(eval, number_choice_after)
            letter = np.random.choice(ref_vector, 1, p=cleaned_list)[0]
            generated_name = generated_name+letter
        name_list = []
        name_list.append(generated_name)
        likelihood_list = []
        for i in range(0,creation_batch_size):
            generated_name, likelihood = backward_generation(prediction,sess,generated_name)
            name_list.append(generated_name)
            likelihood_list.append(likelihood)
            generated_name, likelihood = forward_generation(prediction,sess,generated_name)
            name_list.append(generated_name)
            likelihood_list.append(likelihood)
        print(name_list)
        print(likelihood_list)
        name_list = name_list[:-1]
        print(name_list)
        percentile_likelihood = np.percentile(a=likelihood_list, q=20)
        top_likelihood = (1*(likelihood_list >= percentile_likelihood))
        final_list = []
        for i in range(0, len(top_likelihood)):
            if top_likelihood[i] == 1:
                final_list.append(name_list[i])
        return final_list


def load_full_graph(path, input_data):
  sess = tf.Session('', tf.Graph())
  with sess.graph.as_default():

    saver = tf.train.import_meta_graph(path + '.meta')
    saver.restore(sess, path)

    features = np.array(input_data)
    features = features.reshape(-1, n_chunk, chunk_size)

    v = tf.get_collection('x')

    all_vars = tf.global_variables()


#The general parameters
cc_fips = 'seeran'
backward = False
path = "C:/Users/Antoine Didisheim/Dropbox/PyCharm/TensorFlowTutorials/saved_model/meta-model_"+cc_fips
path_backward = "C:/Users/Antoine Didisheim/Dropbox/PyCharm/TensorFlowTutorials/saved_model/meta-model_backward_" + cc_fips
test_path = 'test_'+cc_fips+'.p'
train_path = 'train_'+cc_fips+'.p'

#the precision parameters
number_choice = 27
number_choice_after = 5
percent_likelihood = 0.20


def generate_name_list(input_name, length_name, creation_batch_size, prediction):

    _, input_vector, output_vector = ff.createsFeaturesForLetter(input_name, 0)
    final_list = use_neural_network(input_vector, length_name, creation_batch_size,prediction)
    return final_list

name_list = []
size_creation = 1
pool_generator_size = 60
str_name = "abc"
prediction = neural_network_model(x)
p = base64.standard_b64encode(bz2.compress(pickle.dumps(prediction)))
p = base64.standard_b64encode(bz2.compress(pickle.dumps(prediction)))
for i in range(0, 6*size_creation):
    str_name = "abc"
    for j in range(4, 7):
        try:
            str_name = str_name+"a"
            print(j,str_name)
            temp_list = generate_name_list(str_name, j, pool_generator_size, prediction)
            print("temp list ", temp_list)
            name_list = name_list + temp_list
        except:
            print("Some error")

    print("////////////////// round 1 -", i+1, "/6 finished /////////////////////")

for i in range(0, 5*size_creation):
    str_name = "abcdefg"
    for j in range(8, 9):
        try:
            str_name = str_name+"a"
            print(j,str_name)
            temp_list = generate_name_list(str_name, j, pool_generator_size, prediction)
            print("temp list ", temp_list)
            name_list = name_list + temp_list
        except:
            print("Some error")
    print("////////////////// round 2 -", i+1, "/5 finished /////////////////////")

for i in range(0, 1*size_creation):
    str_name = "abcdefghijk"
    for j in range(10, 11):
        try:
            str_name = str_name+"a"
            print(j,str_name)
            temp_list = generate_name_list(str_name, j, pool_generator_size, prediction)
            print("temp list ", temp_list)
            name_list = name_list + temp_list
        except:
            print("Some error")
    print("////////////////// round 3 -", i+1, "/1 finished /////////////////////")


random.shuffle(name_list)
name_list = list(set(name_list))
with open('eggs.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    with open("Output_"+cc_fips+".txt", "w") as text_file:
        print("final length", len(name_list))
        for name in name_list:
            text_file.write(name+'\n')
