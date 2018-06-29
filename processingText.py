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

import  learningAlgorithms
from learningAlgorithms import getWordLabel
from learningAlgorithms import vocabularyLength



url = 'http://mattmahoney.net/dc/'

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


# def getLabelWord(label, reverseDictionary):
#    return reverse_dictionary(np.argmax(label))

# the file contains a text file with phrases.
filename = maybe_download('text8.zip', 31344016)
# words = read_data(filename)
words = open('data.txt', 'r').read().split();

print('Data size %d' % len(words))

words = [word for word in words if len(word) > 3]
dictionary, reverse_dictionary = build_dataset(words)
words = [word for word in words if word in dictionary]
maxSamples = 50;
numberOfWords = len(words)


graph = tf.Graph()


weights1, biases1, _, _ = learningAlgorithms.trainCbow(graph, words, dictionary)

for i in range(maxSamples):
    i = random.randint(0, numberOfWords - 1)
    print(learningAlgorithms.getNsimilarWords(words[i], weights1, biases1, dictionary, 10))


# maxTestWord = 0;
# while True:
#     i = random.randint(0, numberOfWords - 1)
#     if len(words[i]) < 4 or not words[i] in dictionary:
#         continue;
#     maxTestWord += 1
#
#     print(words[i] + "[cbow]" + learningAlgorithms.modelCbow(weights1, biases1, weights2, biases2, words[i],dictionary, reverse_dictionary))
#
#     if maxTestWord > 10:
#         break


weights1, biases1, _, _ = learningAlgorithms.trainSkipGram(graph,words,dictionary)

for i in range(maxSamples):
    i = random.randint(0, numberOfWords - 1)
    print(learningAlgorithms.getNsimilarWords(words[i], weights1, biases1, dictionary, 10))

# maxTestWord = 0;
# while True:
#
#     i = random.randint(0, numberOfWords - 1)
#     if len(words[i]) < 4 or not words[i] in dictionary:
#         continue;
#     maxTestWord += 1
#     print(words[i] + "[skipgram]" + str(learningAlgorithms.modelSkipGram(weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval(), words[i], dictionary, reverse_dictionary)))
#
#     if maxTestWord > 10:
#         break