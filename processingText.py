import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from zipfile import ZipFile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

url = 'http://mattmahoney.net/dc/'
vocabularyLength = 5000;
identityMatrix = np.eye(vocabularyLength + 1);

#we eill use an embedding of 100 features
numFeatures = 100;
windowSize = 2;


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    else:
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with ZipFile(filename) as f:
        # get the first file from the archive
        # read the content
        # convert the content to a unicode string using tensor flow compat package
        # split the result by spaces and store the result in data
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# dictionary means to return association betwen word and index
# reverse dictiomat association betwen index and word

def build_dataset(words):
    dictionary = {};
    reverse_dictionary = {};

    listOfWords = collections.Counter(words).most_common(vocabularyLength)
    index = 0

    for el in listOfWords:
        dictionary[el[0]] = index;
        reverse_dictionary[index] = el[0];
        index += 1

    return dictionary, reverse_dictionary


# given a word build return the one hot label
def getWordLabel(words, index):
    global vocabularyLength
    global identityMatrix;
    global dictionary;
    result = 0;
    if word in dictionary:
        result =  identityMatrix[dictionary[word]];
    else:
        result = identityMatrix[vocabularyLength];

    result.shape = (1, vocabularyLength + 1);
    return result;


# the file contains a text file with phrases.
filename = maybe_download('text8.zip', 31344016)
words = read_data(filename)
print('Data size %d' % len(words))
dictionary, reverse_dictionary = build_dataset(words)
graph = tf.Graph()

with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.

    tf_train_dataset = tf.placeholder(tf.float32, shape=(2*windowSize, vocabularyLength + 1));
    tf_train_label = tf.placeholder(tf.float32, shape=(1, vocabularyLength + 1));


    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights1 = tf.Variable(tf.truncated_normal([vocabularyLength + 1, numFeatures]))
    weights2 = tf.Variable(tf.truncated_normal([numFeatures, vocabularyLength + 1]))
    biases1  = tf.Variable(tf.truncated_normal([ numFeatures]))
    biases2  = tf.Variable((tf.truncated_normal([ vocabularyLength + 1])))

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    y1 = tf.matmul(tf_train_dataset, weights1) + biases1
    y1_reduced = tf.reduce_mean(y1, 0);
    y1_reduced = tf.reshape(y1_reduced, shape = (1, y1_reduced.shape[0]))

    y2 = tf.matmul(y1_reduced, weights2) + biases2;

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_label, logits=y2);
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss);
    tf_train_prediction = tf.nn.softmax(y2);

with tf.Session(graph=graph) as session:
     tf.global_variables_initializer().run();
     for i in range(len(words)):
         batch_label = getWordLabel(batchWords[index1])
         batchWords = words[i:i+windowSize]
         for index1 in range(windowSize):

             batch_data = np.ndarray(dtype=np.float32, shape=(0, vocabularyLength + 1));

             for index2 in range(windowSize):
                 if index2 != index1:
                     batch_data = np.append(batch_data, getWordLabel(words[index2]))

             batch_data = batch_data.reshape((windowSize-1 ,vocabularyLength + 1))


             fd = {tf_train_dataset: batch_data, tf_train_label: batch_label}

             _, l, predictions = session.run([optimizer, loss, tf_train_prediction], feed_dict=fd);

             print("session run finished")




