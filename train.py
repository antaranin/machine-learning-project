import datetime
from typing import List, Dict

import numpy as np
import tensorflow as tf

import data_importer as data
from config import *
from test_cnn import CNN
# Data loading params
from word2vec_handler import load_mr_word_vectors


# Parameters
# ==================================================


def preprocessing():
    x_text, y = data.load_data_and_labels_mr()
    mapping, embedding_vectors = load_mr_word_vectors()
    y = np.array(y)
    print(x_text)
    print(y)

    mapping["<PAD>"] = len(embedding_vectors)
    embedding_vectors = np.append(embedding_vectors, np.zeros((1, 300)), axis=0)

    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    vocabulary = mapping.keys()
    x = pad_and_index_sentences(x_text, mapping)
    print(f"X => {x}")
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    print(f"X shuffled => {x_shuffled}")

    # Split train/test set
    dev_sample_index = -1 * int(TEST_SPLIT_RATIO * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocabulary)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocabulary, x_dev, y_dev, embedding_vectors


def pad_sentences(sentences: List[List[str]]):
    max_sentence_length = max([len(x) for x in sentences])
    for sentence in sentences:
        padding = ["<PAD>" for _ in range(max_sentence_length - len(sentence))]
        sentence += padding


def sentences_to_word_indexes(sentences: List[List[str]], word_mapping: Dict[str, int]) -> List[
    List[int]]:
    return [sentence_to_word_indexes(sentence, word_mapping) for sentence in sentences]


def sentence_to_word_indexes(sentence: List[str], word_mapping: Dict[str, int]) -> List[int]:
    return [word_mapping[word] for word in sentence]


def pad_and_index_sentences(sentences: List[List[str]], word_mapping: Dict[str, int]) -> \
        np.ndarray:
    pad_sentences(sentences)
    return np.array(sentences_to_word_indexes(sentences, word_mapping))


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
                step(train_data, train_labels)

            def dev_step(test_data, test_labels):
                """
                Evaluates model on a test set
                """
                step(test_data, test_labels, 1.0, False)

            def step(data, labels, dropout_keep_prob=DROPOUT_KEEP_PROBABILITY,
                     should_train=True):
                data_for_step = {
                    cnn.input_data: data,
                    cnn.input_labels: labels,
                    cnn.dropout_keep_prob: dropout_keep_prob
                }
                if should_train:
                    res = sess.run([global_step, cnn.loss, cnn.accuracy, train_op], data_for_step)
                else:
                    res = sess.run([global_step, cnn.loss, cnn.accuracy], data_for_step)

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
                    dev_step(x_dev, y_dev)
                    print("")


def main(argv=None):
    x_train, y_train, vocabulary, x_dev, y_dev, embedding_vectors = preprocessing()
    train(x_train, y_train, vocabulary, x_dev, y_dev, embedding_vectors)


if __name__ == '__main__':
    tf.app.run()
