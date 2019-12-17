from typing import List, Dict, Tuple, Collection

import numpy as np

from config import EMBEDDING_DIMENSION, SHUFFLE_SEED, TRAIN_TO_TEST_SPLIT_RATIO

_PAD_SYMBOL = "<PAD>"


def preprocess_data(
        data: List[List[str]],
        labels: List[List[int]],
        word_to_index_mapping: Dict[str, int],
        data_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Collection[str], np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array(labels)

    word_to_index_mapping, data_vectors = _prepare_mapping_and_vectors_for_padding(
        word_to_index_mapping, data_vectors)

    vocabulary = list(word_to_index_mapping.keys())
    indexed_data = _pad_and_index_sentences(data, word_to_index_mapping)
    label_count = len(labels)

    # Shuffle data randomly using the SHUFFLE_SEED, for repeatability
    np.random.seed(SHUFFLE_SEED)
    shuffle_indices = np.random.permutation(np.arange(label_count))
    shuffled_data = indexed_data[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]

    # Finds the index matching the split ratio
    test_index = int(TRAIN_TO_TEST_SPLIT_RATIO * float(label_count))
    train_data, test_data = shuffled_data[:test_index], shuffled_data[test_index:]
    train_labels, test_labels = shuffled_labels[:test_index], shuffled_labels[test_index:]

    # Delete unnecessary data from memory
    del data, indexed_data, labels, shuffled_data, shuffled_labels, word_to_index_mapping

    return train_data, train_labels, vocabulary, test_data, test_labels, data_vectors


def _prepare_mapping_and_vectors_for_padding(word_to_index_mapping: Dict[str, int],
                                             data_vectors: np.ndarray):
    word_to_index_mapping[_PAD_SYMBOL] = len(data_vectors)
    data_vectors = np.append(data_vectors, np.zeros((1, EMBEDDING_DIMENSION)), axis=0)
    return word_to_index_mapping, data_vectors


def _pad_sentences(sentences: List[List[str]]):
    max_sentence_length = max([len(x) for x in sentences])
    for sentence in sentences:
        padding = [_PAD_SYMBOL for _ in range(max_sentence_length - len(sentence))]
        sentence += padding


def _sentences_to_word_indexes(sentences: List[List[str]], word_mapping: Dict[str, int]) -> List[
    List[int]]:
    return [_sentence_to_word_indexes(sentence, word_mapping) for sentence in sentences]


def _sentence_to_word_indexes(sentence: List[str], word_mapping: Dict[str, int]) -> List[int]:
    return [word_mapping[word] for word in sentence]


def _pad_and_index_sentences(sentences: List[List[str]], word_mapping: Dict[str, int]) -> \
        np.ndarray:
    _pad_sentences(sentences)
    return np.array(_sentences_to_word_indexes(sentences, word_mapping))
