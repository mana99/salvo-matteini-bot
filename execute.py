"""

"""
import string

import numpy as np
import pandas as pd
import logging
import pickle

from os.path import join, dirname
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.models import load_model

from preprocessing import filter_tweets, encode_tweets, get_t128_italiannlp_embedding, tweet_parsing, preprocess
from model import compile_model

# from scipy.spatial import cKDTree  # distance between vectors
# distances, labels = cKDTree(centers).query(points, 1)

# todo env variables
INPUT_DIR = join(dirname(__file__), 'shared')
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.json')
EMBEDDING_MATRIX_PATH = join(INPUT_DIR, 'embedding.pickle')
MODEL_PATH = join(INPUT_DIR, 'model.h5')

RANDOM_STATE = 1337

SEQ_LENGTH = 52
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


def get_features_labels(tweets, n_vocab, seq_length, embedding_matrix):

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
    shuffled_processed_tweets = shuffle(processed_tweets)  # todo add random seed
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
    print("Importing")
    tweets = import_data(INPUT_PATH)

    # Preprocessing
    print("Preprocessing")
    tweets_iterator, tokenizer = preprocess(tweets['full_text'])
    vocab_size = len(tokenizer.word_index) + 1
    tweets = list(tweets_iterator)

    # Create embedding matrix
    print("Creating Embedding Matrix")
    embedding_matrix = get_t128_italiannlp_embedding(tokenizer=tokenizer, n_words=vocab_size)

    with open(EMBEDDING_PATH, 'wb') as fout:
        pickle.dump((tweets, tokenizer, embedding_matrix), fout)
    # with open(EMBEDDING_MATRIX_PATH, 'rb') as fin:
    #     tweets, tokenizer, embedding_matrix = pickle.load(fin)

    # Partitioning
    print("Partitioning")
    # todo: train valid test
    tweets_train, tweets_valid = partition_tweets(tweets, TRAIN_PERC)

    # Get labels
    print("Generating labels")
    X_train, y_train = get_features_labels(tweets_train, vocab_size, SEQ_LENGTH, embedding_matrix)
    X_valid, y_valid = get_features_labels(tweets_valid, vocab_size, SEQ_LENGTH, embedding_matrix)

    # Compile model
    # print("Build model")
    # model, callbacks = compile_model(X_train, y_train, embedding_matrix=embedding_matrix)

    # # Train and validate model
    # print("Train and validate model")
    # model.fit(X_train, y_train, epochs=20, batch_size=128, callbacks=callbacks,
    #           validation_data=(X_valid, y_valid))

    # # Save model
    # print("Saving final model to file")
    # model.save(MODEL_PATH)

    # Load model
    model = load_model(MODEL_PATH)

    # # Evaluate on validation data
    # todo evaluate on test
    # scores = model.evaluate(X_valid, y_valid, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # todo score model

    # # Score model
    # X_score = [tokenizer.word_index[w] for w in "salvini ha detto che".split()]
    # predicted_result = model.predict(X_score)
    # print(tokenizer.index_word[predicted_result.argmax()])
    #
    # def predict_sentence():
    #     sentence = []
    #
    #     def predictor(sequence=None):
    #         nonlocal sentence
    #         if not sequence and not sentence:
    #             raise Exception
    #         elif sequence:
    #             if sentence:
    #                 print('Restarting')
    #             sentence = sequence
    #             prediction = model.predict(sequence)
    #             sentence.append(prediction)
    #         elif sentence:
    #             prediction = model.predict(sentence[-4:])
    #             sentence.append(prediction)
    #         return sentence
    #
    #     return predictor
    #
    # predict_sentence(X_score)
    # predict_sentence()
    # predict_sentence()
    # predict_sentence()
    # sentence = predict_sentence()
