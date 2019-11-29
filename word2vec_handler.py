import os
from typing import List, Dict, Iterable, Tuple

from gensim.models import KeyedVectors
import csv
import numpy as np

from gensim.models.keyedvectors import FastTextKeyedVectors


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


model = KeyedVectors.load("vectors/news_vectors.model")
mapping, vec = get_vectors_for_words(model, ("word", "name", "blah", "grekarjekfdj", "!", "'s"))
print(vec)
print(mapping)
save_vectors_and_index_mapping(mapping, vec, "TestRun")
# restrict_w2v(model, ("word", "name", "blah", "grekarjekfdj", "!", "'s"))
