import csv
import os
from typing import Dict, Iterable, Tuple

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors

from data_importer import load_vocabulary_mr


def get_vectors_for_words(model: FastTextKeyedVectors, words: Iterable[str]) -> \
        Tuple[Dict[str, int], np.ndarray]:
    words_to_indexes = {}
    vectors = []
    index = 0
    for word in words:
        words_to_indexes[word] = index
        if word in model.vocab:
            vectors.append(model.vectors[model.vocab[word].index])
        else:
            vectors.append(np.random.normal(0, 0.20, 300))
        index += 1
    return words_to_indexes, np.array(vectors)


def save_vectors_and_index_mapping(word_to_index_mapping: Dict[str, int], vectors: np.ndarray,
                                   title: str):
    dir_name = f"vectors/processed/{title}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(f'{dir_name}/mapping.csv', mode='w+') as mapping_file:
        writer = csv.writer(mapping_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for word in word_to_index_mapping:
            writer.writerow([word, word_to_index_mapping[word]])
    np.save(f"vectors/processed/{title}/vectors.npy", vectors)


def save_mr_word_vectors():
    mr_vocabulary = load_vocabulary_mr()
    words = mr_vocabulary.keys()
    model = KeyedVectors.load("vectors/news_vectors.model")
    mapping, vectors = get_vectors_for_words(model, words)
    save_vectors_and_index_mapping(mapping, vectors, "mr")


def load_mr_word_vectors() -> Tuple[Dict[str, int], np.ndarray]:
    return load_vectors_and_index_mapping("mr")


def load_vectors_and_index_mapping(title: str) -> Tuple[Dict[str, int], np.ndarray]:
    dir_name = f"vectors/processed/{title}"
    words_to_indexes = {}
    with open(f'{dir_name}/mapping.csv', mode='r') as mapping_file:
        mappings = csv.reader(mapping_file, delimiter=' ', quotechar='"')
        for entry in mappings:
            words_to_indexes[entry[0]] = entry[1]
    vectors = np.load(f"{dir_name}/vectors.npy")
    return words_to_indexes, vectors


