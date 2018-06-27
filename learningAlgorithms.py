import numpy as np
import tensorflow as tf;
import random

vocabularyLength = 2000;
identityMatrix = np.eye(vocabularyLength);
numFeatures = 100;
windowSize = 5;
maxIterations = vocabularyLength * 50;


def getWordLabel(word, dictionary, targetWord=""):
    global vocabularyLength
    global identityMatrix;

    if word in dictionary:
        result =  identityMatrix[dictionary[word]];
    elif len(targetWord) > 1:
        assert(targetWord in dictionary);
        result = identityMatrix[dictionary[targetWord]];
    else:
        #default the most common word label is returned, if the word is not in dictionary
        result = identityMatrix[0];

    result.shape = (1, vocabularyLength);
    return result;


def trainCbow(graph, words, dictionary):
    #This is cbow
    numberOfWords = len(words)
    with graph.as_default():


        tf_train_dataset = tf.placeholder(tf.float32, shape=(2*windowSize, vocabularyLength));
        tf_train_label = tf.placeholder(tf.float32, shape=(1, vocabularyLength));

        weights1 = tf.Variable(tf.truncated_normal([vocabularyLength, numFeatures]))
        weights2 = tf.Variable(tf.truncated_normal([numFeatures, vocabularyLength]))
        biases1  = tf.Variable(tf.truncated_normal([ numFeatures]))
        biases2  = tf.Variable((tf.truncated_normal([ vocabularyLength])))


        y1 = tf.matmul(tf_train_dataset, weights1) + biases1
        y1_reduced = tf.reduce_mean(y1, 0);
        y1_reduced = tf.reshape(y1_reduced, shape = (1, y1_reduced.shape[0]))

        y2 = tf.matmul(y1_reduced, weights2) + biases2;

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_label, logits=y2));
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss);
        tf_train_prediction = tf.nn.softmax(y2);

    with tf.Session(graph=graph) as session:
         tf.global_variables_initializer().run();


         for index in range(0, min(numberOfWords, maxIterations)):

             i = random.randint(0, numberOfWords - 1)
             if len(words[i]) < 4 or not words[i] in dictionary:
                 continue;

             batch_label = getWordLabel(words[i], dictionary)
             batch_data = np.ndarray(dtype=np.float32, shape=(0, vocabularyLength));
             for index1 in [index2 for index2 in range(i - windowSize, i + windowSize + 1) if index2 != i]:
                 if index1 < len(words) and index1 >= 0:
                     batch_data = np.append(batch_data, getWordLabel(words[index1], dictionary, words[i]))
                 else:
                     batch_data = np.append(batch_data, batch_label)

             batch_data = batch_data.reshape((2 * windowSize ,vocabularyLength));
             fd = {tf_train_dataset: batch_data, tf_train_label: batch_label}
             _, returnedLoss, predictions = session.run([optimizer, loss, tf_train_prediction], feed_dict=fd);
             if index % 50 == 0:
                print("iteration " + str(index) + "/" + str(maxIterations) + ", loss: " + str(returnedLoss));

         return weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval()

def trainSkipGram(graph, words, dictionary):

    numberOfWords = len(words)
    #This is skip-gram
    with graph.as_default():

        tf_train_dataset = tf.placeholder(tf.float32, shape=(1, vocabularyLength));
        tf_train_labels = tf.placeholder(tf.float32, shape=(2 * windowSize, vocabularyLength));

        weights1 = tf.Variable(tf.truncated_normal([vocabularyLength, numFeatures]))
        weights2 = tf.Variable(tf.truncated_normal([numFeatures, vocabularyLength]))
        biases1  = tf.Variable(tf.truncated_normal([ numFeatures]))
        biases2  = tf.Variable((tf.truncated_normal([ vocabularyLength])))


        y1 = tf.matmul(tf_train_dataset, weights1) + biases1
        y2 = tf.matmul(y1, weights2) + biases2;
        y2rep = tf.ones([2 * windowSize, 1]) *  y2;

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=y2rep));
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss);
        tf_train_prediction = tf.nn.softmax(y2);

    with tf.Session(graph=graph) as session:

         tf.global_variables_initializer().run();
         for index in range(0, min(numberOfWords, maxIterations)):

             i = random.randint(0, numberOfWords - 1)
             if len(words[i]) < 4 or not words[i] in dictionary:
                 continue;

             batch_data = getWordLabel(words[i], dictionary)
             batch_labels = np.ndarray(dtype=np.float32, shape=(0, vocabularyLength));
             for index1 in [index2 for index2 in range(i - windowSize, i + windowSize + 1) if index2 != i]:
                 if index1 < len(words) and index1 >= 0:
                     batch_labels = np.append(batch_labels, getWordLabel(words[index1], dictionary, words[i]))
                 else:
                     batch_labels = np.append(batch_labels, batch_data)

             batch_labels = batch_labels.reshape((2 * windowSize ,vocabularyLength))
             fd = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
             _, returnedLoss, predictions = session.run([optimizer, loss, tf_train_prediction], feed_dict=fd);
             if index % 50 == 0:
                print("iteration " + str(index) + "/" + str(maxIterations) + ", loss: " + str(returnedLoss));
         return   weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval()

def modelCbow(weights1, biases1, weights2, biases2, train_dataset, dictionary, reverse_dictionary):

    train_dataset = train_dataset.split();
    batch_data = np.ndarray(dtype=np.float32, shape=(0, vocabularyLength));
    for word in train_dataset:
        assert(word in dictionary);
        batch_data = np.append(batch_data, getWordLabel(word, dictionary))

    batch_data = batch_data.reshape((-1, vocabularyLength))

    y1 = batch_data @ weights1 + biases1
    y1_reduced = np.mean(y1, 0)
    y1_reduced = np.reshape(y1_reduced, (1, y1_reduced.shape[0]))
    y2 = y1_reduced @ weights2 + biases2;

    #skip softmax
    indexes = np.argsort(y2)

    train_datasetIndex = list(map((lambda x: dictionary[x]), train_dataset))

    countDown = 0
    while indexes[0, vocabularyLength-1-countDown] in train_datasetIndex:
        countDown+=1
    return reverse_dictionary[indexes[0, vocabularyLength-1-countDown]]


def modelSkipGram(weights1, biases1, weights2, biases2, train_dataset, dictionary, reverse_dictionary):

    train_dataset = train_dataset.split();
    assert(len(train_dataset) == 1);
    assert(train_dataset[0] in dictionary);

    inputLabel = getWordLabel(train_dataset[0], dictionary)

    y1 = inputLabel @ weights1 + biases1
    y2 = y1 @ weights2 + biases2;

    indexes = np.argsort(y2)

    labels = [];

    count = 0
    i = -1
    while True:

        i+= 1
        currIndex = indexes[0, vocabularyLength - 1 - i];

        if dictionary[train_dataset[0]] == currIndex:
            continue;

        labels.append(reverse_dictionary[currIndex]);

        count += 1
        if count == 2 * windowSize:
            break;
    return labels