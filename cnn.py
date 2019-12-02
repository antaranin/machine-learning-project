import numpy as np
import gensim
import tensorflow as tf
import data_importer


# Model consists of four layers:

# Embedding Layer
def create_embedding(tokenized_vocabulary, dimension, weights):
    pass


# Convolution Layer
def convolve(input, weights, bias):

    input = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    input = tf.nn.bias_add(bias)
    return tf.nn.relu(input)


# Max Pooling
def max_pool(x, kernel_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, kernel_size, kernel_size, 1],
                          padding='VALID')


# Dropout Layer
def dropout(pooled_values):
    return tf.nn.dropout(pooled_values)

# model = gensim.models.Word2Vec.load("./vectors/news_vectors.model")
# #print(model.wv.vectors[:10])
# data = np.load("./vectors/news_vectors.model.vectors.npy")
# print(data)
