import data_importer
import word2vec_handler as w2v
import preprocessor as pre
import train as t


def run_mr(use_word_to_vec=True):
    data, labels = data_importer.load_data_and_labels_mr()
    word_to_index_mapping, vectors = w2v.load_mr_word_vectors()
    train_data, train_labels, vocabulary, test_data, test_labels, embedding_vectors = \
        pre.preprocess_data(data, labels, word_to_index_mapping, vectors)
    if not use_word_to_vec:
        embedding_vectors = None
    t.train(train_data, train_labels, vocabulary, test_data, test_labels, embedding_vectors)


run_mr(True)
