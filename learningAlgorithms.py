import numpy as np
import tensorflow as tf;
import random

from sklearn.metrics.pairwise import cosine_similarity
from math import inf

vocabularyLength = 100;
identityMatrix = np.eye(vocabularyLength);
numFeatures = 100;
windowSize = 6;
maxIterations = vocabularyLength * 10;


def getWordLabel(word, dictionary):
    global vocabularyLength
    global identityMatrix;

    result = identityMatrix[dictionary[word]];
    result.shape = (1, vocabularyLength);

    return result;


def similarity(word1, word2, weights1, biases1, dictionary):
    inputLabel1 = getWordLabel(word1, dictionary)
    #word embedding
    y1 = inputLabel1 @ weights1 + biases1

    inputLabel2 = getWordLabel(word2, dictionary)
    # word embedding
    y2 = inputLabel2 @ weights1 + biases1


    res = np.mean(cosine_similarity(y1, y2))

    return res


def getNsimilarWords(word, weights1, biases1, dictionary, N):
    words =  [*dictionary]
    similarity_dataset= list(map((lambda x: similarity(x, word, weights1, biases1, dictionary)), words))
    similarity_Indexes = np.argsort(similarity_dataset);

    results = []
    for i in range(0,N):
        results.append(words[similarity_Indexes[i]])

    return word + "/////" + str(results);





def trainCbow(graph, words, dictionary, batchSize):
    #This is cbow
    val = list(map(lambda x : getWordLabel(x, dictionary), words))
    train_dataset = np.array(val)
    train_dataset.shape = (-1, vocabularyLength)
    
    
    train_labels = np.array(train_dataset)
    
    
    for i in range(len(words)):
        train_labels[i] = train_dataset[max(i - windowSize, 0)] + train_dataset[min(i + windowSize, vocabularyLength -1 )]
        
    
    
    offsetValidation =  (int) ( 0.7 * len(words))
    
    temp = np.copy(train_dataset)
    train_dataset = temp[:offsetValidation, :]
    train_labels = temp[:offsetValidation, :]
    
    valid_dataset = temp[offsetValidation:, :]
    valid_labels = temp[offsetValidation:, :]
    
    
    with graph.as_default():


        tf_train_dataset = tf.placeholder(dtype = tf.float32, shape=(batchSize, vocabularyLength));
        tf_train_label = tf.placeholder(dtype = tf.float32, shape=(batchSize, vocabularyLength));
        
        tf_valid_dataset = tf.constant(valid_dataset, dtype = tf.float32)
        tf_valid_label =  tf.constant(valid_labels, dtype = tf.float32)
        

        weights1 = tf.Variable(tf.truncated_normal([vocabularyLength, numFeatures]), dtype = tf.float32)
        weights2 = tf.Variable(tf.truncated_normal([numFeatures, vocabularyLength]), dtype = tf.float32)
        biases1  = tf.Variable(tf.truncated_normal([ numFeatures]), dtype = tf.float32)
        biases2  = tf.Variable((tf.truncated_normal([ vocabularyLength])), dtype = tf.float32)


        y1 = tf.matmul(tf_train_dataset, weights1) + biases1
        y1 = tf.nn.leaky_relu(y1)
        y2 = tf.matmul(y1, weights2) + biases2;

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_label, logits=y2));
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss);
        
        
        y1v = tf.matmul(tf_valid_dataset, weights1) + biases1
        y1v = tf.nn.leaky_relu(y1v)     
        y2v = tf.matmul(y1v, weights2) + biases2;
        
        lossV = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_valid_label, logits=y2v));
        
        
        

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run();
        index = 0
        lastVError = inf
        
        while True:
            offset = index * batchSize
            if (offset +  batchSize) > offsetValidation:
                offset = 0
                 
            batch_label = train_labels[offset:offset + batchSize]
            batch_data  = train_dataset[offset:offset + batchSize]
         
            fd = {tf_train_dataset: batch_data, tf_train_label: batch_label}
            _, returnedLoss, lossValidation = session.run([optimizer, loss, lossV], feed_dict=fd);
            if index % 5 == 0:
                print("iteration " + str(index) + "/" + str(maxIterations) + ", loss: " + str(returnedLoss) + ", validation  loss " + str(lossValidation))
                if lossValidation > lastVError:
                    break
                lastVError = lossValidation
                
            index+= 1

        return weights1.eval(), biases1.eval(), weights2.eval(), biases2.eval()




