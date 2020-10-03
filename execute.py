"""

"""

import numpy as np
import pandas as pd
import logging
import pickle

from os.path import join, dirname
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.models import load_model

from preprocessing import filter_tweets, encode_tweets, get_t128_italiannlp_embedding
from model import train_validate_model


# todo env variables
INPUT_DIR = join(dirname(__file__), 'shared')
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.json')
EMBEDDING_MATRIX_PATH = join(INPUT_DIR, 'embedding.pickle')
MODEL_PATH = join(INPUT_DIR, 'model.h5')

RANDOM_STATE = 1337

SEQ_LENGTH = 4
TRAIN_PERC = 0.7

logger = logging.getLogger(__name__)


def import_data(fn, n_max=None):
    logger.info("Importing")
    if n_max:
        return pd.read_csv(fn, usecols=['full_text'], nrows=n_max)
    else:
        return pd.read_csv(fn, usecols=['full_text'])
    # for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE, usecols=['full_text']):
    #      print(chunk)


def get_features_labels(tweets, n_vocab, seq_length):

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
    y = np_utils.to_categorical(y, num_classes=vocab_size)

    return X, y


def partition_tweets(processed_tweets, train_pct: float):
    """
    Perform holdout partitioning by shuffling tweets and splitting according to the train:test ratio.

    :param processed_tweets:
    :param train_pct:
    :return:
    """
    logger.info("Shuffling tweets...")
    tot_tweets_n = len(processed_tweets)
    train_tweets_n = int(train_pct * tot_tweets_n)
    # valid_tweets_n = tot_tweets_n - train_tweets_n
    shuffled_processed_tweets = shuffle(processed_tweets)
    return shuffled_processed_tweets[:train_tweets_n], \
           shuffled_processed_tweets[train_tweets_n:]


def partion_dataset(features, labels, train_perc=0.7, random_state=None, training_length=10):

    # it mantains the same shuffling order for both arrays
    shuffled_features, shuffled_labels = shuffle(features, labels, random_state=random_state)

    train_index = int(len(shuffled_features) * train_perc)
    X_train = shuffled_features[:train_index]
    y_train = shuffled_labels[:train_index]
    X_valid = shuffled_features[train_index:]
    y_valid = shuffled_labels[train_index:]

    return X_train, y_train, X_valid, y_valid


if __name__ == '__main__':

    # Import
    tweets: pd.DataFrame = import_data(INPUT_PATH, 1000)
    logger.info(tweets.shape)

    # Preprocessing
    tweets, max_words = filter_tweets(tweets)
    tweets, tokenizer = encode_tweets(tweets)

    vocab_size = len(tokenizer.word_index) + 1
    # embedding_matrix = get_t128_italiannlp_embedding(tokenizer=tokenizer, n_words=vocab_size)
    # with open(EMBEDDING_PATH, 'wb') as fout:
    #     pickle.dump(embedding_matrix, fout)
    with open(EMBEDDING_MATRIX_PATH, 'rb') as fin:
        embedding_matrix = pickle.load(fin)

    # Partitioning
    tweets_train, tweets_valid = partition_tweets(tweets, TRAIN_PERC)

    # Get labels
    X_train, y_train = get_features_labels(tweets_train, vocab_size, SEQ_LENGTH)
    X_valid, y_valid = get_features_labels(tweets_valid, vocab_size, SEQ_LENGTH)

    # Train and validate model
    model = train_validate_model(X_train, y_train, X_valid, y_valid, embedding_matrix=embedding_matrix)

    # Save model
    model.save(MODEL_PATH)

    # # Evaluate on validation data
    # scores = model.evaluate(X_valid, y_valid, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Load model
    model = load_model(MODEL_PATH)

    # todo score model


