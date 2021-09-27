
import numpy as np

from typing import List


def get_features_labels(tweets: List[List], n_vocab, seq_length, embedding_matrix):
    """
    Get the features/label pairs from the list of tweets. Each sequence of ``seq_length`` words from the flattened
    corpus of tweets will be considered as the input for the following word.


    :param tweets:
    :param n_vocab:
    :param seq_length: the number of word to be considered as input to predict the next word
    :param embedding_matrix:
    :return:
    """

    tweets_flat = [token for tweet in tweets for token in tweet]

    # prepare the dataset of input to output pairs encoded as integers
    X, y = [], []
    for i in range(0, len(tweets_flat) - seq_length, 1):
        X_batch = tweets_flat[i:i + seq_length]
        y_batch = tweets_flat[i + seq_length]
        X.append(X_batch)
        y.append(y_batch)

    # reshape X to be [samples, time steps, features]
    X = np.reshape(X, (len(X), seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    # MemoryError: Unable to allocate 988. GiB for an array with shape (1962233, 135094) and data type float32
    # y = np_utils.to_categorical(y, num_classes=vocab_size)
    y = np.array([embedding_matrix[i] for i in y])

    return X, y