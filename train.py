import datetime

import numpy as np
import tensorflow as tf

from config import *
from test_cnn import CNN


# Data loading params


# Parameters
# ==================================================




def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def train(x_train, y_train, vocabulary, x_dev, y_dev, embedding_vectors=None):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        # session_conf = tf.ConfigProto(
        #     allow_soft_placement=FLAGS.allow_soft_placement,
        #     log_device_placement=FLAGS.log_device_placement)
        # sess = tf.Session(config=session_conf)
        sess = tf.Session()
        with sess.as_default():
            cnn = CNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=EMBEDDING_DIMENSION,
                filter_sizes=FILTER_SIZES,
                num_filters=FILTER_COUNT,
                embedding=embedding_vectors)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(train_data, train_labels):
                """
                A single training step
                """
                step(train_data, train_labels, should_print=PRINT_STEPS)

            def test_step(test_data, test_labels):
                """
                Evaluates model on a test set
                """
                step(test_data, test_labels, 1.0, False)

            def step(data, labels, dropout_keep_prob=DROPOUT_KEEP_PROBABILITY,
                     should_train=True, should_print=True):
                data_for_step = {
                    cnn.input_data: data,
                    cnn.input_labels: labels,
                    cnn.dropout_keep_prob: dropout_keep_prob
                }
                if should_train:
                    res = sess.run([global_step, cnn.loss, cnn.accuracy, train_op], data_for_step)
                else:
                    res = sess.run([global_step, cnn.loss, cnn.accuracy], data_for_step)

                if should_print:
                    current_time = datetime.datetime.now().isoformat()
                    current_step = res[0]
                    loss = res[1]
                    accuracy = res[2]
                    print(f"{current_time}: step {current_step}, loss {loss}, acc {accuracy}")

            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), BATCH_SIZE, EPOCH_COUNT)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % EVALUATE_EVERY == 0:
                    print("\nEvaluation:")
                    test_step(x_dev, y_dev)
                    print("")


