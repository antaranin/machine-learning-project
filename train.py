import datetime

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
        """
        A single training step
        """
        self._step(train_data, train_labels, train_op=train_op, should_print=PRINT_STEPS)

    def test_step(self, test_data, test_labels):
        """
        Evaluates model on a test set
        """
        self._step(test_data, test_labels, 1.0)

    def _step(self, data, labels, dropout_keep_prob=DROPOUT_KEEP_PROBABILITY,
              train_op=None, should_print=True):
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


def generate_batch_iterator(data, batch_size, epoch_count, should_shuffle=True):
    data = np.array(data)
    data_size = len(data)
    batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(epoch_count):
        # Shuffle the data at each epoch
        if should_shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_index in range(batches_per_epoch):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def train(train_data, train_labels, vocabulary, x_dev, y_dev, embedding_vectors=None):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = CNN(
                sequence_length=train_data.shape[1],
                num_classes=train_labels.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=EMBEDDING_DIMENSION,
                filter_sizes=FILTER_SIZES,
                num_filters=FILTER_COUNT,
                embedding=embedding_vectors)

            # Define Training procedure
            global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            step_runner = StepRunner(cnn, sess, global_step)

            # Generate batches
            batches = generate_batch_iterator(
                list(zip(train_data, train_labels)), BATCH_SIZE, EPOCH_COUNT)
            # Training loop. For each batch...
            for batch in batches:
                data_batch, label_batch = zip(*batch)
                step_runner.train_step(data_batch, label_batch, train_op)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % EVALUATE_EVERY == 0:
                    print("\nEvaluation:")
                    step_runner.test_step(x_dev, y_dev)
                    print("")
