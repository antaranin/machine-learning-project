from collections import defaultdict
from typing import Dict, List, DefaultDict, Tuple

import regex as re


def load_word_counts(filename: str, encoding="utf-8") -> DefaultDict[str, int]:
    words = defaultdict(int)
    with open(filename, mode="r", encoding=encoding) as file:
        for line in file:
            line_words = clean_line(line).split(" ")
            for word in line_words:
                words[word] += 1
    return words


def load_sentences(filename: str, encoding="utf-8") -> List[List[str]]:
    sentences = []
    with open(filename, mode="r", encoding=encoding) as file:
        for line in file:
            line_words = clean_line(line).split(" ")
            sentences.append(line_words)
    return sentences


def clean_line(string: str) -> str:
    cleaned = string.lower()
    cleaned = insert_spaces_around_chars(cleaned)
    return cleaned.strip()


def insert_spaces_around_chars(string: str) -> str:
    char_pattern = r"[\?\.,)(\]\[{}!\"]"
    word_pattern = r"\w+"
    left_spaces = insert_space_between_patterns(word_pattern, char_pattern, string)
    right_spaces = insert_space_between_patterns(char_pattern, word_pattern, left_spaces)
    possessives = insert_space_between_patterns(word_pattern, r"\'s", right_spaces)
    return possessives


def insert_space_between_patterns(left_pattern: str, right_pattern: str, string: str) -> str:
    left = r"<left>"
    right = r"<right>"
    grouped_left = f"(?P{left}{left_pattern})"
    grouped_right = f"(?P{right}{right_pattern})"
    return re.sub(grouped_left + grouped_right, f"\g{left} \g{right}", string)


def merge_vocabularies(dict1: Dict[str, int], dict2: Dict[str, int]) -> Dict[str, int]:
    dict3 = {**dict1, **dict2}
    for key in dict3:
        if key in dict1 and key in dict2:
            dict3[key] = dict1[key] + dict2[key]

    return dict3


def load_vocabulary_positive_mr() -> DefaultDict[str, int]:
    return load_word_counts("data/mr/rt-polarity.pos.txt", "iso-8859-1")


def load_vocabulary_negative_mr() -> DefaultDict[str, int]:
    return load_word_counts("data/mr/rt-polarity.neg.txt", "iso-8859-1")


def load_vocabulary_mr():
    positive = load_vocabulary_positive_mr()
    negative = load_vocabulary_negative_mr()
    return merge_vocabularies(positive, negative)


def load_sentences_positive_mr():
    return load_sentences("data/mr/rt-polarity.pos.txt", "iso-8859-1")


def load_sentences_negative_mr():
    return load_sentences("data/mr/rt-polarity.pos.txt", "iso-8859-1")


def load_data_and_labels_mr() -> Tuple[List[List[str]], List[int]]:
    positive = load_sentences_positive_mr()
    negative = load_sentences_negative_mr()
    labels = [1 for _ in positive]
    labels += [0 for _ in negative]
    data = positive + negative
    return data, labels

# print(clean_line("So, this-is a, (somewhat, maybe) cleaned [1] string? Wohoo!"))
