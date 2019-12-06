from typing import Tuple, Union, List

import numpy as np
import tensorflow as tf


class CNN(object):

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, embedding=None):
        self._setup_placeholders(sequence_length, num_classes)
        self._setup_embedding(vocab_size, embedding_size, embedding)
        self._setup_convolutional_pooling_layer(filter_sizes, embedding_size, num_filters,
                                                sequence_length)
        self._setup_dropout_layer()

        total_filter_count = num_filters * len(filter_sizes)
        self._setup_output_layer(total_filter_count, num_classes)
        self._setup_loss_calculation()
        self._setup_accuracy_calculation()

    def _setup_placeholders(self, sequence_length: int, number_of_classes: int):
        self.input_data = tf.placeholder(tf.int32, [None, sequence_length])
        self.input_labels = tf.placeholder(tf.float32, [None, number_of_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _setup_embedding(self, vocabulary_size: int, embedding_size: int,
                         embedding: np.ndarray = None):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding_to_use = embedding if embedding is not None \
                else tf.random_uniform((vocabulary_size, embedding_size), -1.0, 1.0)

            self.W = tf.Variable(embedding_to_use, dtype='float32')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_data)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    def _setup_convolutional_pooling_layer(self, filter_sizes: Union[List[int], Tuple[int]],
                                           embedding_size: int,
                                           number_of_filters: int, sequence_length: int):
        pooled_outputs = []
        for filter_size in filter_sizes:
            pooled = self._setup_convolutional_pooling_layer_for_filter(filter_size,
                                                                        number_of_filters,
                                                                        embedding_size,
                                                                        sequence_length)
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        total_filter_count = number_of_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filter_count])

    def _setup_convolutional_pooling_layer_for_filter(self, filter_size: int,
                                                      number_of_filters: int, embedding_size: int,
                                                      sequence_length: int):
        with tf.name_scope(f"convolutional-maxpool-for-filter-{filter_size}"):
            # Convolution Layer
            filter_shape = (filter_size, embedding_size, 1, number_of_filters)
            weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=(number_of_filters,)))
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                weights,
                strides=(1, 1, 1, 1),
                padding="VALID")
            # Apply nonlinearity

            feature_map = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                feature_map,
                ksize=(1, sequence_length - filter_size + 1, 1, 1),
                strides=(1, 1, 1, 1),
                padding='VALID')
            return pooled

    def _setup_dropout_layer(self):
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def _setup_output_layer(self, total_filter_count: int, number_of_classes: int):
        with tf.name_scope("output"):
            weights = tf.get_variable(
                "W",
                shape=[total_filter_count, number_of_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1, shape=[number_of_classes]))
            self.scores = tf.nn.xw_plus_b(self.h_drop, weights, bias)
            self.predictions = tf.argmax(self.scores, 1)

        # Calculate mean cross-entropy loss

    def _setup_loss_calculation(self):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                             labels=self.input_labels)
            self.loss = tf.reduce_mean(losses)

    def _setup_accuracy_calculation(self):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
