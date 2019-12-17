import datetime
import math
from typing import Iterable, Tuple, Collection

import numpy as np
import tensorflow as tf

from config import *
from test_cnn import CNN


class StepRunner:
    cnn: CNN
    sess: tf.Session

    def __init__(self, cnn: CNN, session: tf.Session, step_counter) -> None:
        super().__init__()
        self.cnn = cnn
        self.sess = session
        self.step_counter = step_counter

    def train_step(self, train_data, train_labels, train_op):
        self._step(train_data, train_labels, train_op=train_op, should_print=PRINT_STEPS)

    def test_step(self, test_data, test_labels):
        self._step(test_data, test_labels, 1.0)

    def _step(self, data: np.ndarray, labels: np.ndarray,
              dropout_keep_prob: float = DROPOUT_KEEP_PROBABILITY,
              train_op=None, should_print: bool = True) -> None:
        data_for_step = {
            self.cnn.input_data: data,
            self.cnn.input_labels: labels,
            self.cnn.dropout_keep_prob: dropout_keep_prob
        }
        if train_op is not None:
            res = self.sess.run([self.step_counter, self.cnn.loss, self.cnn.accuracy, train_op],
                                data_for_step)
        else:
            res = self.sess.run([self.step_counter, self.cnn.loss, self.cnn.accuracy],
                                data_for_step)

        if should_print:
            current_time = datetime.datetime.now().isoformat()
            current_step = res[0]
            loss = res[1]
            accuracy = res[2]
            print(f"{current_time}: step {current_step}, loss {loss}, acc {accuracy}")


def _split_data_into_batches(data: np.ndarray, labels: np.ndarray, batch_size: int,
                             should_shuffle=True) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    assert len(data) == len(labels)
    data_count = len(data)
    if should_shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_count))
        batch_data = data[shuffle_indices]
        batch_labels = labels[shuffle_indices]
    else:
        batch_data = data
        batch_labels = labels

    batch_count = int(math.ceil(data_count / batch_size))
    for batch_index in range(batch_count):
        start_index = batch_index * batch_size
        end_index = min(data_count, (batch_index + 1) * batch_size)
        yield batch_data[start_index:end_index], batch_labels[start_index:end_index]


def train(train_data: np.ndarray, train_labels: np.ndarray, vocabulary: Collection[str],
          test_data: np.ndarray,
          test_labels: np.ndarray, embedding_vectors: np.ndarray = None):
    with tf.Graph().as_default():
        session = tf.Session()
        with session.as_default():
            cnn = CNN(
                sequence_length=train_data.shape[1],
                number_of_classes=train_labels.shape[1],
                vocabulary_size=len(vocabulary),
                embedding_size=EMBEDDING_DIMENSION,
                filter_sizes=FILTER_SIZES,
                filter_count=FILTER_COUNT,
                embedding=embedding_vectors)

            global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            gradients = optimizer.compute_gradients(cnn.loss)
            training_operation = optimizer.apply_gradients(gradients, global_step=global_step)

            session.run(tf.global_variables_initializer())

            step_runner = StepRunner(cnn, session, global_step)

            for epoch in range(EPOCH_COUNT):
                if epoch % EVALUATE_EVERY == 0:
                    print(f"\nEvaluation, epoch {epoch}:")
                    step_runner.test_step(test_data, test_labels)
                batches = _split_data_into_batches(train_data, train_labels, BATCH_SIZE)
                for batch in batches:
                    data_batch, label_batch = batch
                    step_runner.train_step(data_batch, label_batch, training_operation)
