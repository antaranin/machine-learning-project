TRAIN_TO_TEST_SPLIT_RATIO = 0.9
# Only modifiable if not using word2vec
EMBEDDING_DIMENSION = 300
FILTER_SIZES = [3, 4, 5]
FILTER_COUNT = 128
DROPOUT_KEEP_PROBABILITY = 0.5
# Should be a multiplication of 2, for efficiency.
BATCH_SIZE = 64
EPOCH_COUNT = 50
# How often (every how many epochs) to perform and print evaluation of cnn with test data
EVALUATE_EVERY = 1
SHUFFLE_SEED = 42
# Whether to print each individual training step or not
PRINT_STEPS = False
LEARNING_RATE = 1e-3
# The clip norm used for the L2 constraint
CLIP_NORM = 3
# Whether embedded vectors should be trained as well, or not
TRAINABLE_EMBEDDING = True
# Mean used for initializing random word vectors
RANDOM_NORMAL_VECTOR_MEAN = 0.0
# Standard deviation used for initializing random word vectors
RANDOM_NORMAL_VECTOR_STANDARD_DEV = 0.2
