import os
from typing import List, Dict, Iterable, Tuple

from gensim.models import KeyedVectors
import csv
import numpy as np

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
            vectors.append(np.random.uniform(-0.25, 0.25, 300))
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



#save_mr_word_vectors()

