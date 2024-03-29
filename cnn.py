from typing import Collection, Dict

import numpy as np
import tensorflow as tf
from tensorflow import Tensor


class CNN(object):
    _placeholder_data: Tensor
    _placeholder_labels: Tensor
    _placeholder_dropout: Tensor
    _embedded_vectors: Tensor
    _combined_pooled_features: Tensor
    _dropout_result: Tensor
    _label_scores: Tensor
    _predicted_label_indexes: Tensor
    loss: Tensor
    accuracy: Tensor

    def __init__(
            self, max_sentence_length: int, vocabulary_size: int, number_of_classes: int,
            filter_sizes: Collection[int], filter_count: int, embedding_size: int,
            embedding: np.ndarray = None, trainable_embedding: bool = False,
            random_embedding_mean: float = 0, random_embedding_std: float = 0.2
    ):
        self._setup_placeholders(max_sentence_length, number_of_classes)
        self._setup_embedding(vocabulary_size, embedding_size, embedding, trainable_embedding,
                              random_embedding_mean, random_embedding_std)
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
            self._placeholder_data: data,
            self._placeholder_labels: labels,
            self._placeholder_dropout: dropout_keep_probability
        }

    def _setup_placeholders(self, max_sentence_length: int, number_of_classes: int):
        self._placeholder_data = tf.placeholder(tf.int32, (None, max_sentence_length))
        self._placeholder_labels = tf.placeholder(tf.float32, (None, number_of_classes))
        self._placeholder_dropout = tf.placeholder(tf.float32)

    def _setup_embedding(self, vocabulary_size: int, embedding_size: int,
                         embedding: np.ndarray = None, trainable_embedding: bool = False,
                         random_embedding_mean: float = 0, random_embedding_std: float = 0.2):
        embedding_to_use = embedding if embedding is not None \
            else tf.random_normal((vocabulary_size, embedding_size), mean=random_embedding_mean,
                                  stddev=random_embedding_std)
        # tf.random_uniform((vocabulary_size, embedding_size), -1.0, 1.0)
        weights = tf.Variable(embedding_to_use, dtype='float32', trainable=trainable_embedding)
        embedded_vec = tf.nn.embedding_lookup(weights, self._placeholder_data)
        # Add one dimension representing channels. Should be 1
        self._embedded_vectors = tf.expand_dims(embedded_vec, -1)

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
        # flatten features
        self._combined_pooled_features = tf.reshape(
            combined_pooled_features, (-1, total_filter_count)
        )

    def _setup_convolutional_pooling_layer_for_filter(self, filter_size: int,
                                                      filter_count: int,
                                                      embedding_size: int,
                                                      max_sentence_length: int):
        shape_of_filter = (filter_size, embedding_size, 1, filter_count)
        weights = tf.Variable(tf.truncated_normal(shape_of_filter, stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=(filter_count,)))
        # Input shape in form data, rows, columns, channels
        # In our case that is ?, 62(sentence length) 300(embedding size), 1
        convolution = tf.nn.conv2d(
            self._embedded_vectors,
            weights,
            strides=(1, 1, 1, 1),
            padding="VALID")

        convolution_with_bias = tf.nn.bias_add(convolution, bias)
        feature_map = tf.nn.relu(convolution_with_bias)

        window_size = max_sentence_length - filter_size + 1
        pooling = tf.nn.max_pool(
            feature_map,
            ksize=(1, window_size, 1, 1),
            strides=(1, 1, 1, 1),
            padding='VALID')

        return pooling

    def _setup_dropout_layer(self):
        self._dropout_result = tf.nn.dropout(self._combined_pooled_features,
                                             self._placeholder_dropout)

    def _setup_output_layer(self, total_filter_count: int, number_of_classes: int):
        weights = tf.get_variable(
            "weights",
            shape=(total_filter_count, number_of_classes),
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=(number_of_classes,)))
        self._label_scores = tf.nn.xw_plus_b(self._dropout_result, weights, bias)
        self._predicted_label_indexes = tf.argmax(self._label_scores, axis=1)

    def _setup_accuracy_calculation(self):
        # Gets the index of a label that is set (since the labels are set as binary values)
        label_indexes = tf.argmax(self._placeholder_labels, axis=1)
        matching_predictions = tf.equal(self._predicted_label_indexes, label_indexes)
        self.accuracy = tf.reduce_mean(tf.cast(matching_predictions, "float"))

    def _setup_loss_calculation(self):
        loss_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=self._label_scores,
                                                              labels=self._placeholder_labels)
        self.loss = tf.reduce_mean(loss_matrix)
