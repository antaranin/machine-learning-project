from typing import List, Dict

import numpy as np

from config import EMBEDDING_DIMENSION, SHUFFLE_SEED, TEST_SPLIT_RATIO


def preprocess_data(data: List[List[str]], labels: List[List[int]],
                    word_to_index_mapping: Dict[str, int], data_vectors: np.ndarray):
    labels = np.array(labels)

    word_to_index_mapping, data_vectors = prepare_mapping_and_vectors_for_padding(
        word_to_index_mapping, data_vectors)

    vocabulary = word_to_index_mapping.keys()
    indexed_data = pad_and_index_sentences(data, word_to_index_mapping)
    label_count = len(labels)

    # Shuffle data randomly
    np.random.seed(SHUFFLE_SEED)
    shuffle_indices = np.random.permutation(np.arange(label_count))
    shuffled_data = indexed_data[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]

    # Split train/test data
    test_index = -1 * int(TEST_SPLIT_RATIO * float(label_count))
    train_data, test_data = shuffled_data[:test_index], shuffled_data[test_index:]
    train_labels, test_labels = shuffled_labels[:test_index], shuffled_labels[test_index:]

    #delete unnecessary data from memory
    del data, indexed_data, labels, shuffled_data, shuffled_labels, word_to_index_mapping

    return train_data, train_labels, vocabulary, test_data, test_labels, data_vectors


def prepare_mapping_and_vectors_for_padding(word_to_index_mapping: Dict[str, int],
                                            data_vectors: np.ndarray):
    word_to_index_mapping["<PAD>"] = len(data_vectors)
    data_vectors = np.append(data_vectors, np.zeros((1, EMBEDDING_DIMENSION)), axis=0)
    return word_to_index_mapping, data_vectors


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
