from collections import defaultdict
from typing import Dict

import regex as re


def load_words(filename: str, encoding="utf-8") -> Dict[str, int]:
    words = defaultdict(int)
    with open(filename, mode="r", encoding=encoding) as file:
        for line in file:
            line_words = clean_line(line).split(" ")
            for word in line_words:
                words[word] += 1
    return words


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


def load_positive_mr():
    return load_words("data/mr/rt-polarity.pos.txt", "iso-8859-1")


def load_negative_mr():
    return load_words("data/mr/rt-polarity.neg.txt", "iso-8859-1")

# print(clean_line("So, this-is a, (somewhat, maybe) cleaned [1] string? Wohoo!"))

