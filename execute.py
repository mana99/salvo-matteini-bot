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
import tensorflow as tf
from tensorflow.python.keras.utils.losses_utils import ReductionV2


from preprocessing import get_t128_italiannlp_embedding, preprocess
from model import compile_model

# from scipy.spatial import cKDTree  # distance between vectors
# distances, labels = cKDTree(centers).query(points, 1)

# todo env variables
INPUT_DIR = join(dirname(__file__), 'shared')
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.json')
EMBEDDING_MATRIX_PATH = join(INPUT_DIR, 'embedding.pickle')
MODEL_PATH = join(INPUT_DIR, 'model.h5')
PREPROCESSED_TWEETS_PATH = join(INPUT_DIR, 'preprocessed-tweets.pickle')
TOKENIZER_PATH = join(INPUT_DIR, 'tokenizer.pickle')
SCORE_PATH = join(INPUT_DIR, 'score.txt')
SPLITTED_DATASETS_PATH = join(INPUT_DIR, 'splitted_datasets.pickle')

RANDOM_STATE = 1337

SEQ_LENGTH = 52

NUM_EPOCHS = 5
TRAIN_VALID_TEST_RATIO = (0.55, 0.3, 0.15)

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


def partition_tweets(tweets, train_valid_test):
    """
    Perform holdout partitioning by shuffling tweets and splitting according to the train:test ratio.

    :param tweets:
    :param train_valid_test:
    :return:
    """

    train, valid, test = train_valid_test

    if train + valid + test != 1:
        train, valid, test = tuple(x / (train + valid + test) for x in (train, valid, test))

    logger.info("Shuffling tweets...")
    shuffled_tweets = shuffle(tweets)  # todo add random seed
    # 25 tweets for scoring
    score_tweets = []
    for i in range(25):
        score_tweets.append(shuffled_tweets.pop())
    # train, validation, test sets
    train_n, test_n, valid_n = tuple(int(x * len(shuffled_tweets)) for x in (train, valid, test))

    return shuffled_tweets[:train_n], \
           shuffled_tweets[train_n:train_n+valid_n], \
           shuffled_tweets[train_n+valid_n:], \
           score_tweets

if __name__ == '__main__':

    # with open(TOKENIZER_PATH, 'rb') as fin:
    #     tokenizer = pickle.load(fin)
    # with open(PREPROCESSED_TWEETS_PATH, 'rb') as fin:
    #     tweets = pickle.load(fin)
    # with open(EMBEDDING_MATRIX_PATH, 'rb') as fin:
    #     embedding_matrix = pickle.load(fin)
    # with open(SPLITTED_DATASETS_PATH, 'rb') as fin:
    #     X_train, y_train, X_valid, y_valid, X_test, y_test, tweets_score = pickle.load(fin)
    # model = load_model(MODEL_PATH)

    # Import
    print("Importing")
    tweets = import_data(INPUT_PATH)

    # Preprocessing
    print("Preprocessing")
    tweets_iterator, tokenizer = preprocess(tweets['full_text'])
    with open(TOKENIZER_PATH, 'wb') as fout:
        pickle.dump(tokenizer, fout)
    vocab_size = len(tokenizer.word_index) + 1
    tweets = list(tweets_iterator)
    with open(PREPROCESSED_TWEETS_PATH, 'wb') as fout:
        pickle.dump(tweets, fout)

    # Create embedding matrix
    print("Creating Embedding Matrix")
    embedding_matrix = get_t128_italiannlp_embedding(tokenizer=tokenizer, n_words=vocab_size)
    with open(EMBEDDING_PATH, 'wb') as fout:
        pickle.dump(embedding_matrix, fout)

    # Partitioning
    print("Partitioning")
    tweets_train, tweets_valid, tweets_test, tweets_score = partition_tweets(tweets, TRAIN_VALID_TEST_RATIO)

    # Get labels
    print("Generating labels")
    X_train, y_train = get_features_labels(tweets_train, vocab_size, SEQ_LENGTH, embedding_matrix)
    X_valid, y_valid = get_features_labels(tweets_valid, vocab_size, SEQ_LENGTH, embedding_matrix)
    X_test, y_test = get_features_labels(tweets_test, vocab_size, SEQ_LENGTH, embedding_matrix)

    with open(SPLITTED_DATASETS_PATH, 'wb') as fout:
        pickle.dump((X_train, y_train, X_valid, y_valid, X_test, y_test, tweets_score), fout)

    # Compile model
    print("Build model")
    model, callbacks = compile_model(X_train, y_train, embedding_matrix=embedding_matrix)

    # # Train and validate model
    print("Train and validate model")
    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=128, callbacks=callbacks,
              validation_data=(X_valid, y_valid))

    # # Save model
    print("Saving final model to file")
    model.save(MODEL_PATH)

    # Evaluate on validation data
    metrics_values = model.evaluate(X_test, y_test, verbose=0)
    if len(model.metrics_names) == 1:
        print("%s: %.2f%%" % (model.metrics_names[0], metrics_values*100))
    else:
        print(', '.join(["%s: %.2f%%" % (x, y) for x, y in zip(model.metrics_names, metrics_values)]))

    # Score model
    # todo: optimize
    X_score = [word
               for tweet in tweets_score
               for word in tweet][:SEQ_LENGTH]

    def get_word(index):
        if index == 0:
            return '\\START'
        else:
            return tokenizer.index_word[index]

    score_words = [get_word(x) for x in X_score]
    embedding_matrix_float32 = tf.cast(embedding_matrix, float).numpy()
    embedding_matrix_df = pd.DataFrame(embedding_matrix_float32)
    dist = lambda x: tf.losses.CosineSimilarity(reduction=ReductionV2.AUTO)(x, predicted_result[-1]).numpy()
    MAX = 30
    for i in range(1, MAX):
        print(f"Predicting word {i}/{MAX}...")
        # predict
        predicted_result = model.predict(X_score)
        # nearest word in embedding vector
        dist_df = embedding_matrix_df.apply(dist, axis=1)
        predicted_embedding = abs(dist_df).argmin()
        # index to word
        predicted_word = get_word(predicted_embedding)
        print(predicted_word)
        score_words.append(predicted_word)
        # add predicted word to the input
        X_score = X_score[1:] + [predicted_embedding]

    with open(SCORE_PATH, 'w') as fout:
        fout.write(" ".join(score_words))

    # todo: closure implementation
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
