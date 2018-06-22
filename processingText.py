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
maxIterations = 1000;

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
def getWordLabel(word):
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

def getLabelWord(label):
    index = np.argmax(label)
    if index < vocabularyLength:
        return reverse_dictionary[index];
    return "."


# the file contains a text file with phrases.
filename = maybe_download('text8.zip', 31344016)
words = read_data(filename)
totalNumberOfWords = (float)(len(set(words)))
print('Data size %d' % len(words))
dictionary, reverse_dictionary = build_dataset(words)
graph = tf.Graph()

words = [word for word in words if word in dictionary or random.random() < (float)(vocabularyLength * 0.5)/(totalNumberOfWords  - vocabularyLength)]


def softmax(x):
    return  np.exp(x) / np.sum(np.exp(x),axis = 0);

#train dataset has windowSize * 2
# it is a list of words
def model(weights1, biases1, weights2, biases2, train_dataset):

    train_dataset = train_dataset.split();
    assert(len(train_dataset) == 2 * windowSize);
    batch_data = np.ndarray(dtype=np.float32, shape=(0, vocabularyLength + 1));
    for word in train_dataset:
        batch_data = np.append(batch_data, getWordLabel(word))

    batch_data = batch_data.reshape((2 * windowSize, vocabularyLength + 1))

    y1 = batch_data @ weights1 + biases1
    y1_reduced = np.mean(y1, 0)
    y1_reduced = np.reshape(y1_reduced, (1, y1_reduced.shape[0]))
    y2 = y1_reduced @ weights2 + biases2;

    #skip softmax

    return getLabelWord(y2);


#This is cbow
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

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_label, logits=y2));
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss);
    tf_train_prediction = tf.nn.softmax(y2);

with tf.Session(graph=graph) as session:
     tf.global_variables_initializer().run();
     for i in range(0, min(len(words), maxIterations) ,windowSize):
         batch_label = getWordLabel(words[i])
         batch_data = np.ndarray(dtype=np.float32, shape=(0, vocabularyLength + 1));
         for index1 in [index2 for index2 in range(i - windowSize, i + windowSize + 1) if index2 != i]:
             if index1 < len(words) and index1 >= 0:
                 batch_data = np.append(batch_data, getWordLabel(words[index1]))
             else:
                 batch_data = np.append(batch_data, batch_label)

         batch_data = batch_data.reshape((2 * windowSize ,vocabularyLength + 1))
         fd = {tf_train_dataset: batch_data, tf_train_label: batch_label}
         _, returnedLoss, predictions = session.run([optimizer, loss, tf_train_prediction], feed_dict=fd);
         print("iteration " + str(i) + "/" + str(len(words)) + ", loss: " + str(returnedLoss));

     # if the results are not returned by session.run, they can be evalua
     print(model(weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval(), "I feel bad today "))
     print(model(weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval(), "you went mountain winter"))
     print(model(weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval(), "king power land queen"))


#This is skip-gram
with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.

    tf_train_dataset = tf.placeholder(tf.float32, shape=(1, vocabularyLength + 1));
    tf_train_labels = tf.placeholder(tf.float32, shape=(2 * windowSize, vocabularyLength + 1));


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
    y2 = tf.matmul(y1, weights2) + biases2;
    y2rep = tf.ones([2 * windowSize, 1]) *  y2;

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=y2rep));
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss);
    tf_train_prediction = tf.nn.softmax(y2);

with tf.Session(graph=graph) as session:
     tf.global_variables_initializer().run();
     for i in range(0, min(len(words), maxIterations) ,windowSize):
         batch_data = getWordLabel(words[i])
         batch_labels = np.ndarray(dtype=np.float32, shape=(0, vocabularyLength + 1));
         for index1 in [index2 for index2 in range(i - windowSize, i + windowSize + 1) if index2 != i]:
             if index1 < len(words) and index1 >= 0:
                 batch_labels = np.append(batch_labels, getWordLabel(words[index1]))
             else:
                 batch_labels = np.append(batch_labels, batch_data)

         batch_labels = batch_labels.reshape((2 * windowSize ,vocabularyLength + 1))
         fd = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
         _, returnedLoss, predictions = session.run([optimizer, loss, tf_train_prediction], feed_dict=fd);
         print("iteration " + str(i) + "/" + str(len(words)) + ", loss: " + str(returnedLoss));

     # if the results are not returned by session.run, they can be evalua
     # print(model(weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval(), "I feel bad today "))
     # print(model(weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval(), "you went mountain winter"))
     # print(model(weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval(), "king power land queen"))