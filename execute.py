"""

"""

import numpy as np
import pandas as pd
import logging
import pickle

from os.path import join, dirname

from keras.utils import np_utils
from sklearn.utils import shuffle
from model import train_validate_model, test_model
from keras.models import load_model
from preprocessing import preprocess, get_t128_italiannlp_embedding

from keras_preprocessing.text import Tokenizer


INPUT_DIR = join(dirname(__file__), 'shared')
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
EMBEDDING_PATH = join(INPUT_DIR, 'embedding.pickle')
MODEL_PATH = join(INPUT_DIR, 'model.h5')
TRAIN_PERC = 0.7
RANDOM_STATE = 1337

SEQ_LENGTH = 15

logger = logging.getLogger(__name__)


def import_data(fn, n_max=None):
    logger.info("Importing")
    if n_max:
        return pd.read_csv(fn, usecols=['full_text'], nrows=n_max)
    else:
        return pd.read_csv(fn, usecols=['full_text'])
    # for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE, usecols=['full_text']):
    #      print(chunk)


def get_features_labels(proc_tweets, n_vocab, seq_length):
    proc_tweets_flat = [token for tweet in proc_tweets for token in tweet]

    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for i in range(0, len(proc_tweets_flat) - seq_length, 1):
        seq_in = proc_tweets_flat[i:i + seq_length]
        seq_out = proc_tweets_flat[i + seq_length]
        dataX.append(seq_in)
        dataY.append(seq_out)
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY, num_classes=vocab_size)

    return X, y

    # X = proc_tweets_flat[:-1]  # 100 parole
    # y = proc_tweets_flat[1:]   # 101 esima
    # return X, y


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


if __name__ == '__main__':

    # Import
    tweets: pd.DataFrame = import_data(INPUT_PATH)
    logger.info(tweets.shape)

    # Preprocessing
    processed_tweets, tokenizer, vocab_size, max_words = preprocess(tweets)

    # embedding_matrix = get_t128_italiannlp_embedding(tokenizer=tokenizer, vocab_size=vocab_size)
    # with open(EMBEDDING_PATH, 'wb') as fout:
    #     pickle.dump(embedding_matrix, fout)
    with open(EMBEDDING_PATH, 'rb') as fin:
        embedding_matrix = pickle.load(fin)

    # Partitioning
    processed_tweets_train, processed_tweets_valid = partition_tweets(processed_tweets, TRAIN_PERC)

    # Get labels
    X_train, y_train = get_features_labels(processed_tweets_train, vocab_size, SEQ_LENGTH)
    X_valid, y_valid = get_features_labels(processed_tweets_valid, vocab_size, SEQ_LENGTH)

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
    score_output = test_model(model, X_valid[:SEQ_LENGTH])


def partion_dataset(features, labels, train_perc=0.7, random_state=None, training_length=10):

    # aggregator = cleaning()
    # # features = features_labels_with_strings(aggregator, training_length)
    # # fixed_length_shorter_makes_sense = 998 # 500
    # # aggregator = aggregator[0:fixed_length_shorter_makes_sense]
    #
    # aggregator_integers = integers_conversion(aggregator)
    # features, labels = features_labels(aggregator_integers,
    #                                    training_length)  # check if the features should be like these
    # print(features, labels)
    #
    # label_array = create_one_hot_encoding(features, labels)

    # it mantains the same shuffling order for both arrays
    shuffled_features, shuffled_labels = shuffle(features, labels, random_state=random_state)

    train_index = int(len(shuffled_features) * train_perc)
    X_train = shuffled_features[:train_index]
    y_train = shuffled_labels[:train_index]
    X_valid = shuffled_features[train_index:]
    y_valid = shuffled_labels[train_index:]

    return X_train, y_train, X_valid, y_valid