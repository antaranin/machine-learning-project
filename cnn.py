from typing import Collection, Dict

import numpy as np
import tensorflow as tf


class CNN(object):

    def __init__(
            self, max_sentence_length: int, vocabulary_size: int, number_of_classes: int,
            filter_sizes: Collection[int], filter_count: int, embedding_size: int,
            embedding: np.ndarray = None
    ):
        self._setup_placeholders(max_sentence_length, number_of_classes)
        self._setup_embedding(vocabulary_size, embedding_size, embedding)
        self._setup_convolutional_pooling_layer(filter_sizes, embedding_size, filter_count,
                                                max_sentence_length)
        self._setup_dropout_layer()

        total_filter_count = filter_count * len(filter_sizes)
        self._setup_output_layer(total_filter_count, number_of_classes)
        self._setup_accuracy_calculation()
        self._setup_loss_calculation()

    def create_data_dict(self, data: np.ndarray, labels: np.ndarray,
                         dropout_keep_probability: float) -> Dict:
        return {
            self.placeholder_data: data,
            self.placeholder_labels: labels,
            self.placeholder_dropout: dropout_keep_probability
        }

    def _setup_placeholders(self, max_sentence_length: int, number_of_classes: int):
        self.placeholder_data = tf.placeholder(tf.int32, [None, max_sentence_length])
        self.placeholder_labels = tf.placeholder(tf.float32, [None, number_of_classes])
        self.placeholder_dropout = tf.placeholder(tf.float32)

    def _setup_embedding(self, vocabulary_size: int, embedding_size: int,
                         embedding: np.ndarray = None):
        embedding_to_use = embedding if embedding is not None \
            else tf.random_uniform((vocabulary_size, embedding_size), -1.0, 1.0)

        weights = tf.Variable(embedding_to_use, dtype='float32', trainable=False)
        embedded_vec = tf.nn.embedding_lookup(weights, self.placeholder_data)
        self.embedded_vectors = tf.expand_dims(embedded_vec, -1)

    def _setup_convolutional_pooling_layer(self, filter_sizes: Collection[int],
                                           embedding_size: int,
                                           filter_count: int, max_sentence_length: int):
        pooled_outputs = [self._setup_convolutional_pooling_layer_for_filter(
            filter_size,
            filter_count,
            embedding_size,
            max_sentence_length
        ) for filter_size in filter_sizes]

        total_filter_count = filter_count * len(filter_sizes)
        combined_pooled_features = tf.concat(pooled_outputs, axis=3)
        self.combined_pooled_features_flattened = tf.reshape(
            combined_pooled_features, [-1, total_filter_count]
        )

    def _setup_convolutional_pooling_layer_for_filter(self, filter_size: int,
                                                      filter_count: int,
                                                      embedding_size: int,
                                                      max_sentence_length: int):
        shape_of_filter = (filter_size, embedding_size, 1, filter_count)
        weights = tf.Variable(tf.truncated_normal(shape_of_filter, stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=(filter_count,)))
        convolution = tf.nn.conv2d(
            self.embedded_vectors,
            weights,
            strides=(1, 1, 1, 1),
            padding="VALID")

        convolution_with_bias = tf.nn.bias_add(convolution, bias)
        feature_map = tf.nn.relu(convolution_with_bias)

        '''
        window size - since every filter corresponds to one pooled value, therefore,
        there has to be max size of a sentence - filter size + 1 values in the pooling layer
        '''
        window_size = max_sentence_length - filter_size + 1
        pooling = tf.nn.max_pool(
            feature_map,
            ksize=(1, window_size, 1, 1),
            strides=(1, 1, 1, 1),
            padding='VALID')
        return pooling

    def _setup_dropout_layer(self):
        self.dropout = tf.nn.dropout(self.combined_pooled_features_flattened,
                                     self.placeholder_dropout)

    def _setup_output_layer(self, total_filter_count: int, number_of_classes: int):
        weights = tf.get_variable(
            "weights",
            shape=[total_filter_count, number_of_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[number_of_classes]))
        self.scores = tf.nn.xw_plus_b(self.dropout, weights, bias)
        self.prediction_indexes = tf.argmax(self.scores, axis=1)

    def _setup_accuracy_calculation(self):
        label_indexes = tf.argmax(self.placeholder_labels, axis=1)
        correct_predictions = tf.equal(self.prediction_indexes, label_indexes)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    def _setup_loss_calculation(self):
        loss_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                              labels=self.placeholder_labels)
        self.loss = tf.reduce_mean(loss_matrix)
